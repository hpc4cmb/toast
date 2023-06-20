# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections.abc import Mapping, MutableMapping
from typing import NamedTuple

import numpy as np
from astropy import units as u
from pshmem import MPIShared

from .accelerator import (
    AcceleratorObject,
    accel_data_create,
    accel_data_delete,
    accel_data_present,
    accel_data_reset,
    accel_data_update_device,
    accel_data_update_host,
    accel_enabled,
    use_accel_jax,
    use_accel_omp,
)
from .intervals import IntervalList
from .mpi import MPI, comm_equivalent
from .utils import Logger, dtype_to_aligned

if use_accel_jax:
    import jax

    from .jax.mutableArray import MutableJaxArray, _zero_out_jitted


class DetectorData(AcceleratorObject):
    """Class representing a logical collection of co-sampled detector data.

    This class works like an array of detector data where the first dimension is the
    number of detectors and the second dimension is the data for that detector.  The
    data for a particular detector may itself be multi-dimensional, with the first
    dimension the number of samples.

    The data in this container may be sliced by both detector indices and names, as
    well as by sample range.

    Example:
        Imagine we have 3 detectors and each has 10 samples.  We want to store a
        4-element value at each sample using 4-byte floats.  We would do::

            detdata = DetectorData(["d01", "d02", "d03"], (10, 4), np.float32)

        and then we can access the data for an individual detector either by index
        or by name with::

            detdata["d01"] = np.ones((10, 4), dtype=np.float32)
            firstdet = detdata[0]

        slicing by index and by a list of detectors is possible::

            array_view = detdata[0:-1, 2:4]
            array_view = detdata[["d01", "d03"], 3:8]

    Args:
        detectors (list):  A list of detector names in exactly the order you wish.
            This order is fixed for the life of the object.
        shape (tuple):  The shape of the data *for each detector*.  The first element
            of this shape should be the number of samples.
        dtype (numpy.dtype):  A numpy-compatible dtype for each element of the detector
            data.  The only supported types are 1, 2, 4, and 8 byte signed and unsigned
            integers, 4 and 8 byte floating point numbers, and 4 and 8 byte complex
            numbers.
        units (Unit):  Optional scalar unit associated with this data.
        view_data (array):  (Internal use only) This makes it possible to create
            DetectorData instances that act as a view on an existing array.

    """

    def __init__(
        self, detectors, shape, dtype, units=u.dimensionless_unscaled, view_data=None
    ):
        log = Logger.get()
        super().__init__()

        self._set_detectors(detectors)
        self._units = units

        (
            self._storage_class,
            self.itemsize,
            self._dtype,
            self._shape,
            self._flatshape,
        ) = self._data_props(detectors, shape, dtype)

        self._fullsize = 0
        self._memsize = 0
        self._raw = None
        self._flatdata = None
        self._data = None

        if view_data is None:
            # Allocate the data
            self._allocate()
            self._is_view = False
        else:
            # We are provided the data
            if self._shape != view_data.shape:
                msg = (
                    "view data shape ({}) does not match constructor shape ({})".format(
                        view_data.shape, self._shape
                    )
                )
                log.error(msg)
                raise RuntimeError(msg)
            self._data = view_data
            self._is_view = True

    def _set_detectors(self, detectors):
        log = Logger.get()
        self._detectors = detectors
        if len(self._detectors) == 0:
            msg = "You must specify a list of at least one detector name"
            log.error(msg)
            raise ValueError(msg)
        self._name2idx = {y: x for x, y in enumerate(self._detectors)}

    def _data_props(self, detectors, detshape, dtype):
        log = Logger.get()
        dt = np.dtype(dtype)
        storage_class, itemsize = dtype_to_aligned(dtype)

        # Verify that our shape contains only integral values
        flatshape = len(detectors)
        for d in detshape:
            if not isinstance(d, (int, np.integer)):
                msg = "input shape contains non-integer values"
                log.error(msg)
                raise ValueError(msg)
            flatshape *= d

        shp = [len(detectors)]
        shp.extend(detshape)
        shp = tuple(shp)
        return (storage_class, itemsize, dt, shp, flatshape)

    def _allocate(self):
        log = Logger.get()
        self._fullsize = self._flatshape
        self._memsize = self.itemsize * self._fullsize

        # First delete potential device data
        # FIXME we might be able to recycle existing buffers to improve performance
        create_accel = False
        on_accel = False
        if self.accel_exists():
            # There is a buffer on the accelerator
            create_accel = True
            if self.accel_in_use():
                # The accelerator copy is the one in use
                on_accel = True
                msg = "Reallocation of DetectorData which is staged to accelerator- "
                msg += "Deleting device copy and re-allocating."
                log.verbose(msg)
            self.accel_delete()

        # Delete existing wrapper and buffer
        if self._data is not None:
            del self._data
        if self._flatdata is not None:
            del self._flatdata
        if self._raw is not None:
            self._raw.clear()
            del self._raw

        # Allocate new host buffer
        self._raw = self._storage_class.zeros(self._fullsize)

        # Wrap _raw
        self._flatdata = self._raw.array()[: self._flatshape]
        self._data = self._flatdata.reshape(self._shape)

        # Restore device buffer if needed
        if create_accel:
            self._accel_create(zero_out=True)
        if on_accel:
            self.accel_used(True)

    @property
    def detectors(self):
        return list(self._detectors)

    def keys(self):
        return list(self._detectors)

    def indices(self, names):
        """Return the detector indices of the specified detectors.

        Args:
            names (iterable):  The detector names.

        Returns:
            (array):  The detector indices.

        """
        return np.array([self._name2idx[x] for x in names], dtype=np.int32)

    @property
    def dtype(self):
        return self._dtype

    @property
    def units(self):
        return self._units

    @property
    def shape(self):
        return self._shape

    @property
    def detector_shape(self):
        return tuple(self._shape[1:])

    def memory_use(self):
        return self._memsize

    @property
    def data(self):
        if (not hasattr(self, "_data")) or self._data is None:
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        return self._data

    @property
    def flatdata(self):
        if (not hasattr(self, "_flatdata")) or self._flatdata is None:
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        return self._flatdata

    def update_units(self, new_units):
        """Update the detector data units."""
        self._units = new_units

    def change_detectors(self, detectors):
        """Modify the list of detectors.

        This attempts to re-use the underlying memory and just change the detector
        mapping to that memory.  This is useful if memory allocation is expensive.
        If the new list of detectors is longer than the original, a new memory buffer
        is allocated.  If the new list of detectors is shorter than the original, the
        buffer is kept and only a subset is used.

        The return value indicates whether the underlying memory was re-allocated.

        Args:
            detectors (list):  A list of detector names in exactly the order you wish.

        Returns:
            (bool):  True if the data was re-allocated, else False.

        """
        log = Logger.get()
        if self._is_view:
            msg = "Cannot resize a DetectorData view"
            log.error(msg)
            raise RuntimeError(msg)

        if detectors == self._detectors:
            # No-op
            return

        # Get the new data properties
        (storage_class, itemsize, dt, shp, flatshape) = self._data_props(
            detectors, self._shape[1:], self._dtype
        )

        self._set_detectors(detectors)

        if flatshape > self._fullsize:
            # We have to reallocate...
            self.clear()
            self._shape = shp
            self._flatshape = flatshape
            self._allocate()
            realloced = True
        else:
            # We can re-use the existing memory
            self._shape = shp
            self._flatshape = flatshape

            # Check if we have data on device before touching any buffer
            does_accel_exist = self.accel_exists()
            if does_accel_exist and use_accel_jax:
                # set aside the JAX device array for later recycling
                previous_device_data = self._data.data

            # Adjust the size of the data wrapper and reset underlying buffer to zero.
            self._flatdata = self._raw.array()[: self._flatshape]
            self._flatdata[:] = 0
            # this might trash a MutableJaxArray falsifying further self.accel_exists() tests
            self._data = self._flatdata.reshape(self._shape)

            if does_accel_exist:
                # We also have a copy on the device
                if use_accel_jax:
                    # creates a device buffer filled with zeroes
                    # we call _zero_out_jitted with the previous device buffer and a new shape into order to recycle the memory
                    # accel_reset cannot be used as there is a change in shape
                    device_data = _zero_out_jitted(
                        previous_device_data, output_shape=self._shape
                    )
                    self._data = MutableJaxArray(
                        cpu_data=self._data, gpu_data=device_data
                    )
                else:
                    # Set device copy to zero
                    self.accel_reset()

            realloced = False
        return realloced

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        # first delete potential GPU data
        if self.accel_exists():
            log = Logger.get()
            msg = "clear() of DetectorData which is staged to accelerator- "
            msg += "Deleting device copy."
            log.verbose(msg)
            self.accel_delete()
        # then apply clear
        if hasattr(self, "_data"):
            del self._data
            self._data = None
        if hasattr(self, "_is_view") and not self._is_view:
            if hasattr(self, "_flatdata"):
                del self._flatdata
                self._flatdata = None
            if hasattr(self, "_raw") and not self._raw:
                self._raw.clear()
                del self._raw
                self._raw = None

    def reset(self, dets=None):
        """Zero the current memory.

        The data buffer currently in use on either the host or accelerator is set
        to zero.

        Args:
            dets (list):  Only zero the data for these detectors.

        Returns:
            None

        """
        if self.accel_in_use():
            self.accel_reset()
        elif dets is None:
            # Zero the whole thing
            self.data[:] = 0
        else:
            for d in dets:
                self[d, :] = 0

    def __del__(self):
        self.clear()

    def _det_axis_view(self, key):
        if isinstance(key, (int, np.integer)):
            # Just one detector by index
            view = key
        elif isinstance(key, str):
            # Just one detector by name
            view = self._name2idx[key]
        elif isinstance(key, slice):
            # We are slicing detectors by index
            view = key
        else:
            # Assume that our key is at least iterable
            try:
                test = iter(key)
                view = tuple([self._name2idx[k] for k in key])
            except TypeError:
                log = Logger.get()
                msg = "Detector indexing supports slice, int, string or "
                msg += f"iterable, not '{key}'"
                log.error(msg)
                raise TypeError(msg)
        return view

    def _get_view(self, key):
        if isinstance(key, (tuple, Mapping)):
            # We are slicing in both detector and sample dimensions
            if len(key) > len(self._shape):
                log = Logger.get()
                msg = "DetectorData has only {} dimensions".format(len(self._shape))
                log.error(msg)
                raise TypeError(msg)
            view = [self._det_axis_view(key[0])]
            for k in key[1:]:
                view.append(k)
            # for s in range(len(self._shape) - len(key)):
            #     view += (slice(None, None, None),)
            return tuple(view)
        else:
            # Only detector slice
            view = self._det_axis_view(key)
            # for s in range(len(self._shape) - 1):
            #     view += (slice(None, None, None),)
            return view

    def __getitem__(self, key):
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        view = self._get_view(key)
        return self._data[view]

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete individual elements")
        return

    def __setitem__(self, key, value):
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        view = self._get_view(key)
        self._data[view] = value

    def view(self, key):
        """Create a new DetectorData instance that acts as a view of the data.

        Args:
            key (tuple/slice):  This is an indexing on detector or both detector and
                sample, the same as you would use to access data elements.

        Returns:
            (DetectorData):  A new instance whose data is a view of the current object.

        """
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        full_view = self._get_view(key)
        view_dets = self.detectors[full_view[0]]
        return DetectorData(
            view_dets,
            self._data[full_view].shape[1:],
            self._dtype,
            view_data=self._data[full_view],
        )

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._detectors)

    def __repr__(self):
        val = None
        if self._is_view:
            val = "<DetectorData (view)"
        else:
            val = "<DetectorData"
        val += " {} detectors each with shape {}, type {}, units {}:".format(
            len(self._detectors), self._shape[1:], self._dtype, self._units
        )
        if self._shape[1] <= 4:
            for d in self._detectors:
                vw = self.data[self._get_view(d)]
                val += "\n  {} = [ ".format(d)
                for i in range(self._shape[1]):
                    val += "{} ".format(vw[i])
                val += "]"
        else:
            for d in self._detectors:
                vw = self.data[self._get_view(d)]
                val += "\n  {} = [ {} {} ... {} {} ]".format(
                    d, vw[0], vw[1], vw[-2], vw[-1]
                )
        val += "\n>"
        return val

    def __eq__(self, other):
        # Please leave the commented-out lines for ease of future
        # debugging.
        if self.detectors != other.detectors:
            # msg = f"DetectorData dets not equal {self.detectors} != {other.detectors}"
            # print(msg)
            return False
        if self.dtype.char != other.dtype.char:
            # msg = f"DetectorData dtype not equal {self.dtype.char} "
            # msg += f"!= {other.dtype.char}"
            # print(msg)
            return False
        if self.shape != other.shape:
            # msg = f"DetectorData shape not equal {self.shape} != {other.shape}"
            # print(msg)
            return False
        if self.units != other.units:
            # msg = f"DetectorData dets not equal {self.units} != {other.units}"
            # print(msg)
            return False
        if self.dtype == np.dtype(np.float64):
            drange = np.amax(self.data) - np.amin(self.data)
            rtol = 1.0e-12
            atol = 10.0 * drange * 1.0e-15
            if not np.allclose(self.data, other.data, rtol=rtol, atol=atol):
                # indx = np.logical_not(
                #     np.isclose(self.data, other.data, rtol=rtol, atol=atol)
                # )
                # msg = f"DetectorData array not close rtol={rtol}, atol={atol}:"
                # for d in np.arange(self.data.shape[0]):
                #     dname = self.detectors[d]
                #     for s in np.arange(self.data.shape[1])[indx[d]]:
                #         msg += f"\n {dname}, {s}:  {self.data[d, s]}"
                #         msg += f" != {other.data[d, s]}"
                # print(msg)
                return False
        elif self.dtype == np.dtype(np.float32):
            drange = np.amax(self.data) - np.amin(self.data)
            rtol = 1.0e-6
            atol = 10.0 * drange * 1.0e-6
            if not np.allclose(self.data, other.data, rtol=rtol, atol=atol):
                # indx = np.logical_not(
                #     np.isclose(self.data, other.data, rtol=rtol, atol=atol)
                # )
                # msg = f"DetectorData array not close rtol={rtol}, atol={atol}:"
                # for d in np.arange(self.data.shape[0]):
                #     dname = self.detectors[d]
                #     for s in np.arange(self.data.shape[1])[indx[d]]:
                #         msg += f"\n {dname}, {s}:  {self.data[d, s]}"
                #         msg += f" != {other.data[d, s]}"
                # print(msg)
                return False
        elif not np.array_equal(self.data, other.data):
            # indx = np.where(self.data != other.data)[0]
            # msg = "DetectorData array not equal:"
            # for d in np.arange(self.data.shape[0]):
            #     dname = self.detectors[d]
            #     for s in np.arange(self.data.shape[1])[indx[d]]:
            #         msg += f"\n {dname}, {s}:  {self.data[d, s]}"
            #         msg += f" != {other.data[d, s]}"
            # print(msg)
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _accel_exists(self):
        if self._raw is None:
            # We have a view.
            # FIXME: eventually we could check the state of the underlying
            # object and use that.
            return False
        else:
            if use_accel_omp:
                return accel_data_present(self._raw, self._accel_name)
            elif use_accel_jax:
                return accel_data_present(self._data)
            else:
                return False

    def _accel_create(self, zero_out=False):
        if use_accel_omp:
            self._raw = accel_data_create(
                self._raw, self._accel_name, zero_out=zero_out
            )
        elif use_accel_jax:
            self._data = accel_data_create(self._data, zero_out=zero_out)

    def _accel_update_device(self):
        if use_accel_omp:
            self._raw = accel_data_update_device(self._raw, self._accel_name)
        elif use_accel_jax:
            self._data = accel_data_update_device(self._data)

    def _accel_update_host(self):
        if use_accel_omp:
            self._raw = accel_data_update_host(self._raw, self._accel_name)
        elif use_accel_jax:
            self._data = accel_data_update_host(self._data)

    def _accel_delete(self):
        if use_accel_omp:
            self._raw = accel_data_delete(self._raw, self._accel_name)
        elif use_accel_jax:
            self._data = accel_data_delete(self._data)

    def _accel_reset(self):
        if use_accel_omp:
            accel_data_reset(self._raw, self._accel_name)
        elif use_accel_jax:
            self._data = accel_data_reset(self._data)


class DetDataManager(MutableMapping):
    """Class used to manage DetectorData objects in an Observation.

    New objects can be created several ways.  The "create()" method:

        ob.detdata.create(name, sample_shape=None, dtype=None, detectors=None)

    gives full control over creating the named object and specifying the shape of
    each detector sample.  The detectors argument can be used to restrict the object
    to include only a subset of detectors.

    You can also create a new object by assignment from an existing DetectorData
    object or a dictionary of detector arrays.  For example:

        ob.detdata[name] = DetectorData(ob.local_detectors, ob.n_local_samples, dtype)

        ob.detdata[name] = {
            x: np.ones((ob.n_local_samples, 2), dtype=np.int16)
                for x in ob.local_detectors
        }

    Where the right hand side object must have only detectors that are included in
    the ob.local_detectors and the first dimension of shape must be the number of
    local samples.

    It is also possible to create a new object by assigning an array.  In that case
    the array must either have the full size of the DetectorData object
    (n_det x n_sample x sample_shape) or must have dimensions
    (n_sample x sample_shape), in which case the array is copied to all detectors.
    For example:

        ob.detdata[name] = np.ones(
            (len(ob.local_detectors), ob.n_local_samples, 4), dtype=np.float32
        )

        ob.detdata[name] = np.ones(
            (ob.n_local_samples,), dtype=np.float32
        )

    After creation, you can access a given DetectorData object by name with standard
    dictionary syntax:

        ob.detdata[name]

    And delete it as well:

        del ob.detdata[name]

    """

    def __init__(self, dist):
        self.samples = dist.samps[dist.comm.group_rank].n_elem
        self.detectors = dist.dets[dist.comm.group_rank]
        self._internal = dict()

    def _data_shape(self, sample_shape):
        dshape = None
        if sample_shape is None or len(sample_shape) == 0:
            dshape = (self.samples,)
        elif len(sample_shape) == 1 and sample_shape[0] == 1:
            dshape = (self.samples,)
        else:
            dshape = (self.samples,) + sample_shape
        return dshape

    def create(
        self,
        name,
        sample_shape=None,
        dtype=np.float64,
        detectors=None,
        units=u.dimensionless_unscaled,
    ):
        """Create a local DetectorData buffer on this process.

        This method can be used to create arrays of detector data for storing signal,
        flags, or other timestream products on each process.

        If the named detector data already exists in an observation, then additional
        checks are done that the sample_shape and dtype match the existing object.
        If so, then the DetectorData.change_detectors() method is called to re-use
        this existing memory buffer if possible.

        Args:
            name (str): The name of the detector data (signal, flags, etc)
            sample_shape (tuple): Use this shape for the data of each detector sample.
                Use None or an empty tuple if you want one element per sample.
            dtype (np.dtype): Use this dtype for each element.
            detectors (list):  Only construct a data object for this set of detectors.
                This is useful if creating temporary data within a pipeline working
                on a subset of detectors.
            units (Unit):  Optional scalar unit associated with this data.

        Returns:
            None

        """
        log = Logger.get()

        if detectors is None:
            detectors = self.detectors
        else:
            for d in detectors:
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)

        data_shape = self._data_shape(sample_shape)

        if name in self._internal:
            msg = "detdata '{}' already exists".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        # Create the data object
        self._internal[name] = DetectorData(detectors, data_shape, dtype, units=units)

        return

    def ensure(
        self,
        name,
        sample_shape=None,
        dtype=np.float64,
        detectors=None,
        create_units=u.dimensionless_unscaled,
        accel=False,
    ):
        """Ensure that the observation has the named detector data.

        If the named detdata object does not exist, it is created.  If it does exist
        and the sample shape and dtype are compatible, then it is checked whether the
        specified detectors are already included.  If not, it calls the
        DetectorData.change_detectors() method to re-use this existing memory buffer if
        possible.

        The return value is true if the data already exists and includes the specified
        detectors.

        The create_units option is used if the detector data does not yet exist, in
        which case the units will be set to this.

        Args:
            name (str): The name of the detector data (signal, flags, etc)
            sample_shape (tuple): Use this shape for the data of each detector sample.
                Use None or an empty tuple if you want one element per sample.
            dtype (np.dtype): Use this dtype for each element.
            detectors (list):  Ensure that these detectors exist in the object.
            create_units (Unit):  Optional scalar unit associated with this data.
                Only used if creating a new detdata object.
            accel (bool):  If True, make sure the device copy is in use, else use
                the host copy.

        Returns:
            (bool):  True if the data exists.

        """
        log = Logger.get()

        if detectors is None:
            detectors = self.detectors
        else:
            for d in detectors:
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)

        data_shape = self._data_shape(sample_shape)

        existing = True

        if name in self._internal:
            # The object already exists.  Check properties.
            dt = np.dtype(dtype)
            if dt != self._internal[name].dtype:
                msg = "Detector data '{}' already exists with dtype {}.".format(
                    name, self._internal[name].dtype
                )
                log.error(msg)
                raise RuntimeError(msg)
            if data_shape != self._internal[name].detector_shape:
                msg = "Detector data '{}' already exists with det shape {}.".format(
                    name, self._internal[name].detector_shape
                )
                log.error(msg)
                raise RuntimeError(msg)
            # Ok, we can re-use this.  Are the detectors already included in the data?
            internal_dets = set(self._internal[name].detectors)
            for test_det in detectors:
                if test_det not in internal_dets:
                    # At least one detector is not included.  In this case we change
                    # detectors and set the units.  The change_detectors() method
                    # Resets memory to zero.
                    existing = False
                    _ = self._internal[name].change_detectors(detectors)
                    self._internal[name].update_units(create_units)
                    break
        else:
            # Create the data object.  This zeros the memory.
            existing = False
            self.create(
                name,
                sample_shape=sample_shape,
                dtype=dtype,
                detectors=detectors,
                units=create_units,
            )
        if existing:
            # The data object exists with correct detectors, however if
            # we need to move data then we set existing to False so that
            # calling code knows it needs to generate the data.
            if accel:
                # We want the data on the device
                if not self.accel_in_use(name):
                    # The device copy needs to be updated
                    if not self.accel_exists(name):
                        # The device buffer does not exist, create it
                        self.accel_create(name)
                    # Update device copy
                    self.accel_update_device(name)
            else:
                # We want the data on the host
                if self.accel_in_use(name):
                    # The host copy needs to be updated
                    self.accel_update_host(name)
        else:
            # The data was either created or the existing object was re-used with
            # the detector list changed.  In either case, both host and device copies
            # (if they exist) have been zeroed out.  However, if changed_detectors
            # was called to re-use memory, the host/device in use flag is intentionally
            # not changed.
            if accel:
                # We want the data on the device.  If the data already exists, then
                # it was already zeroed out.
                if not self.accel_in_use(name):
                    if not self.accel_exists(name):
                        # Create the buffer and zero
                        self.accel_create(name, zero_out=True)
                    # Mark buffer as in-use
                    self.accel_used(name, True)
            else:
                # We want the data on the host.  The host buffer was already zeroed,
                # we just need to mark that as the one in use.
                if self.accel_in_use(name):
                    if use_accel_omp:
                        self.accel_used(name, False)
                    elif use_accel_jax:
                        # We need to convert jax array back to numpy
                        self.accel_update_host(name)
                    else:
                        msg = f"Should never get here: newly created detdata "
                        msg += "using neither openmp nor jax"
                        raise RuntimeError(msg)
        return existing

    def accel_exists(self, key):
        """Check if the named detector data exists on the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            (bool):  True if the data is present.

        """
        if not accel_enabled():
            return False
        log = Logger.get()
        result = self._internal[key].accel_exists()
        msg = f"DetDataMgr {key} type = {type(self._internal[key])} "
        msg += f"accel_exists = {result}"
        log.verbose(msg)
        return result

    def accel_in_use(self, key):
        """Check if the detector data device copy is the one currently in use.

        Args:
            key (str):  The object name.

        Returns:
            (bool):  True if the accelerator device copy is being used.

        """
        return self._internal[key].accel_in_use()

    def accel_used(self, key, state):
        """Set the in-use state of the detector data device copy.

        Setting the state to `True` is only possible if the data exists
        on the device.

        Args:
            key (str):  The object name.
            state (bool):  True if the device copy is in use, else False.

        Returns:
            None

        """
        self._internal[key].accel_used(state)

    def accel_create(self, key, zero_out=False):
        """Create the named detector data on the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        log.verbose(f"DetDataMgr {key} type = {type(self._internal[key])} accel_create")
        self._internal[key].accel_create(key, zero_out=zero_out)

    def accel_update_device(self, key):
        """Copy the named detector data to the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        log.verbose(
            f"DetDataMgr {key} type = {type(self._internal[key])} accel_update_device"
        )
        self._internal[key].accel_update_device()

    def accel_update_host(self, key):
        """Copy the named detector data from the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        log.verbose(
            f"DetDataMgr {key} type = {type(self._internal[key])} accel_update_host"
        )
        self._internal[key].accel_update_host()

    def accel_delete(self, key):
        """Delete the named detector data from the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        log = Logger.get()
        if not accel_enabled():
            return
        if not self._internal[key].accel_exists():
            msg = f"Detector data '{key}' type = {type(self._internal[key])} "
            msg += f"is not present on device, cannot delete"
            log.error(msg)
            raise RuntimeError(msg)
        log.verbose(f"DetDataMgr {key} type = {type(self._internal[key])} accel_delete")
        self._internal[key].accel_delete()

    def accel_reset(self, key):
        """Reset the named detector data from the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        log = Logger.get()
        if not accel_enabled():
            return
        if not self._internal[key].accel_exists():
            msg = f"Detector data '{key}' type = {type(self._internal[key])} "
            msg += f"is not present on device, cannot reset"
            log.error(msg)
            raise RuntimeError(msg)
        log.verbose(f"DetDataMgr {key} type = {type(self._internal[key])} accel_reset")
        self._internal[key].accel_reset()

    def accel_clear(self):
        """Clear all data from accelerators

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        for key in self._internal:
            if self._internal[key].accel_exists():
                log.verbose(
                    f"DetDataMgr {key} type = {type(self._internal[key])} accel_delete"
                )
                self._internal[key].accel_delete()

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        if key in self._internal:
            self._internal[key].clear()
            del self._internal[key]

    def __setitem__(self, key, value):
        if isinstance(value, DetectorData):
            # We have an input detector data object.  Verify properties.
            for d in value.detectors:
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)
            if value.shape[1] != self.samples:
                msg = f"Assignment DetectorData object has {value.shape[1]} samples "
                msg += "instead of {self.samples} in the observation"
                raise ValueError(msg)
            if key not in self._internal:
                # Create it first
                self.create(
                    key,
                    sample_shape=value.detector_shape[1:],
                    dtype=value.dtype,
                    detectors=value.detectors,
                    units=value.units,
                )
            else:
                if value.detector_shape != self._internal[key].detector_shape:
                    msg = "Assignment value has wrong detector shape"
                    raise ValueError(msg)
                if value.units != self._internal[key].units:
                    msg = "Assignment value has wrong units"
                    raise ValueError(msg)
            for d in value.detectors:
                self._internal[key][d] = value[d]
        elif isinstance(value, Mapping):
            # This is a dictionary of detector arrays
            sample_shape = None
            dtype = None
            dunits = None
            for d, ddata in value.items():
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)
                if ddata.shape[0] != self.samples:
                    msg = f"Assigment dictionary detector {d} has {ddata.shape[0]} "
                    msg += f"samples instead of {self.samples} in the observation"
                    raise ValueError(msg)

                # Check consistent units
                cur_units = u.dimensionless_unscaled
                if isinstance(ddata, u.Quantity):
                    cur_units = ddata.unit
                if dunits is None:
                    dunits = cur_units
                else:
                    if dunits != cur_units:
                        msg = f"Assignment dictionary detector {d} has "
                        msg += f"units '{cur_units}' instead of '{dunits}'"
                        raise ValueError(msg)
                # Check sample shape
                if sample_shape is None:
                    sample_shape = ddata.shape[1:]
                    dtype = ddata.dtype
                else:
                    if sample_shape != ddata.shape[1:]:
                        msg = "All detector arrays must have the same shape"
                        raise ValueError(msg)
                    if dtype != ddata.dtype:
                        msg = "All detector arrays must have the same type"
                        raise ValueError(msg)
            if key not in self._internal:
                self.create(
                    key,
                    sample_shape=sample_shape,
                    dtype=dtype,
                    detectors=sorted(value.keys()),
                    units=dunits,
                )
            else:
                if (self.samples,) + sample_shape != self._internal[key].detector_shape:
                    msg = "Assignment value has wrong detector shape"
                    raise ValueError(msg)
            for d, ddata in value.items():
                if isinstance(value, u.Quantity):
                    self._internal[key][d] = ddata.value
                else:
                    self._internal[key][d] = ddata
        else:
            # This must be just an array- verify the dimensions
            val_units = u.dimensionless_unscaled
            if isinstance(value, u.Quantity):
                val_units = value.unit
            shp = value.shape
            if shp[0] == self.samples:
                # This is a single detector array, being assigned to all detectors
                sample_shape = None
                if len(shp) > 1:
                    sample_shape = shp[1:]
                if key not in self._internal:
                    self.create(
                        key,
                        sample_shape=sample_shape,
                        dtype=value.dtype,
                        detectors=self.detectors,
                        units=val_units,
                    )
                else:
                    fullshape = (self.samples,)
                    if sample_shape is not None:
                        fullshape += sample_shape
                    if fullshape != self._internal[key].detector_shape:
                        msg = "Assignment value has wrong detector shape"
                        raise ValueError(msg)
                    if val_units != self._internal[key].units:
                        msg = "Assignment value has wrong units"
                        raise ValueError(msg)
                if isinstance(value, u.Quantity):
                    for d in self.detectors:
                        self._internal[key][d] = value.value
                else:
                    for d in self.detectors:
                        self._internal[key][d] = value
            elif shp[0] == len(self.detectors):
                # Full sized array
                if shp[1] != self.samples:
                    msg = "Assignment value has wrong number of samples"
                    raise ValueError(msg)
                sample_shape = None
                if len(shp) > 2:
                    sample_shape = shp[2:]
                if key not in self._internal:
                    self.create(
                        key,
                        sample_shape=sample_shape,
                        dtype=value.dtype,
                        detectors=self.detectors,
                        units=val_units,
                    )
                else:
                    fullshape = (self.samples,)
                    if sample_shape is not None:
                        fullshape += sample_shape
                    if fullshape != self._internal[key].detector_shape:
                        msg = "Assignment value has wrong detector shape"
                        raise ValueError(msg)
                    if val_units != self._internal[key].units:
                        msg = "Assignment value has wrong units"
                        raise ValueError(msg)
                if isinstance(value, u.Quantity):
                    self._internal[key][:] = value.value
                else:
                    self._internal[key][:] = value
            else:
                # Incompatible
                msg = "Assignment of detector data from an array only supports full "
                msg += "size or single detector"
                raise ValueError(msg)

    def memory_use(self):
        bytes = 0
        for k in self._internal.keys():
            bytes += self._internal[k].memory_use()
        return bytes

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        for k in self._internal.keys():
            self._internal[k].clear()

    def __repr__(self):
        val = "<DetDataManager {} local detectors, {} samples".format(
            len(self.detectors), self.samples
        )
        for k in self._internal.keys():
            val += "\n    {}: shape={}, dtype={}, units='{}'".format(
                k,
                self._internal[k].shape,
                self._internal[k].dtype,
                self._internal[k].units,
            )
        val += ">"
        return val

    def __eq__(self, other):
        log = Logger.get()
        if self.detectors != other.detectors:
            log.verbose(f"  detectors {self.detectors} != {other.detectors}")
            return False
        if self.samples != other.samples:
            log.verbose(f"  samples {self.samples} != {other.samples}")
            return False
        if set(self._internal.keys()) != set(other._internal.keys()):
            log.verbose(f"  keys {self._internal.keys()} != {other._internal.keys()}")
            return False
        for k in self._internal.keys():
            if self._internal[k] != other._internal[k]:
                msg = f"  detector data {k} not equal:  "
                msg += f"{self._internal[k]} != {other._internal[k]}"
                log.verbose(msg)
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class SharedDataType(NamedTuple):
    """The shared data object and a string specifying the comm type."""

    shdata: MPIShared
    type: str


class SharedDataManager(MutableMapping):
    """Class used to manage shared data objects in an Observation.

    New objects can be created with the "create_*()" methods:

        obs.shared.create_group(name, shape=None, dtype=None)
        obs.shared.create_row(name, shape=None, dtype=None)
        obs.shared.create_column(name, shape=None, dtype=None)

    You can also create shared objects by assignment from an existing MPIShared object
    or an array on one process.  In the case of creating from an array assignment, an
    extra communication step is required to determine what process is sending the data
    (all processes except for one should pass 'None' as the data).  For example:

        timestamps = None
        if obs.comm_col_rank == 0:
            # Input data only exists on one process
            timestamps = np.arange(obs.n_local_samples, dtype=np.float32)

        # Explicitly create the shared data and assign:
        obs.shared.create_column(
            "times",
            shape=(obs.n_local_samples,),
            dtype=np.float32
        )
        obs.shared["times"].set(timestamps, offset=(0,), fromrank=0)

        # Create from existing MPIShared object on the column communicator:
        sharedtime = MPIShared((obs.n_local_samples,), np.float32, obs.comm_col)
        sharedtime[:] = timestamps
        obs.shared["times"] = (sharedtime, "column")

        # Create from array on one process, pre-communication needed:
        obs.shared["times"] = (timestamps, "column")

    If you are creating data products shared over the whole group communicator, you
    may leave off the "group" communicator type:

        if obs.comm_col_rank == 0:
            obs.shared["stuff"] = np.ones(100)
        else:
            obs.shared["stuff"] = None

    After creation, you can access a given object by name with standard dictionary
    syntax:

        obs.shared[name]

    And delete it as well:

        del obs.shared[name]

    """

    def __init__(self, dist):
        self.n_detectors = len(dist.dets[dist.comm.group_rank])
        self.n_samples = dist.samps[dist.comm.group_rank].n_elem
        self.dist = dist
        # The internal dictionary stores tuples containing the shared
        # data object and a string specifying which communicator it
        # is distributed over:  "group", "row", or "column".
        self._internal = dict()
        self._accel_used = dict()

    def create_group(self, name, shape, dtype=None):
        """Create a shared memory buffer on the group communicator.

        This buffer will be replicated across all nodes used by the processes owning
        the observation.  This uses the MPIShared class, which falls back to a simple
        numpy array if MPI is not being used.

        Args:
            name (str): Name of the shared memory object.
            shape (tuple): The shape of the new buffer.
            dtype (np.dtype): Use this dtype for each element.

        Returns:
            None

        """
        log = Logger.get()
        if name in self._internal:
            msg = "Observation data with name '{}' already exists.".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        shared_comm = self.dist.comm.comm_group
        shared_comm_node = self.dist.comm.comm_group_node
        shared_comm_rank_node = self.dist.comm.comm_group_node_rank

        shared_dtype = dtype

        # Use defaults for dtype if not set
        if shared_dtype is None:
            shared_dtype = np.float64

        # Create the data object
        self._internal[name] = SharedDataType(
            MPIShared(
                shape,
                shared_dtype,
                shared_comm,
                comm_node=shared_comm_node,
                comm_node_rank=shared_comm_rank_node,
            ),
            "group",
        )
        self._accel_used[name] = False

        return

    def create_row(self, name, shape, dtype=None):
        """Create a shared memory buffer on the process row communicator.

        This buffer will be replicated across all nodes used by the processes in the
        process grid row.  This uses the MPIShared class, which falls back to a simple
        numpy array if MPI is not being used.

        Args:
            name (str): Name of the shared memory object (e.g. "beams").
            shape (tuple): The shape of the new buffer.
            dtype (np.dtype): Use this dtype for each element.

        Returns:
            None

        """
        log = Logger.get()
        if name in self._internal:
            msg = "Observation data with name '{}' already exists.".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        # This is shared over the row communicator, so the leading
        # dimension must be the number of local detectors.
        if shape[0] != self.n_detectors:
            msg = f"When creating shared data '{name}' on the row communicator, "
            msg += f"the leading dimension should be the number of local "
            msg += f"detectors ({self.n_detectors}).  Shape given = {shape}."
            log.error(msg)
            raise RuntimeError(msg)

        shared_comm = self.dist.comm_row
        shared_comm_node = self.dist.comm_row_node
        shared_comm_rank_node = self.dist.comm_row_rank_node

        shared_dtype = dtype

        # Use defaults for dtype if not set
        if shared_dtype is None:
            shared_dtype = np.float64

        # Create the data object
        self._internal[name] = SharedDataType(
            MPIShared(
                shape,
                shared_dtype,
                shared_comm,
                comm_node=shared_comm_node,
                comm_node_rank=shared_comm_rank_node,
            ),
            "row",
        )
        self._accel_used[name] = False

        return

    def create_column(self, name, shape, dtype=None):
        """Create a shared memory buffer on the process column communicator.

        This buffer will be replicated across all nodes used by the processes in the
        process grid column.  This uses the MPIShared class, which falls back to a
        simple numpy array if MPI is not being used.

        Args:
            name (str): Name of the shared memory object (e.g. "boresight").
            shape (tuple): The shape of the new buffer.
            dtype (np.dtype): Use this dtype for each element.

        Returns:
            None

        """
        log = Logger.get()
        if name in self._internal:
            msg = "Observation data with name '{}' already exists.".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        # This is shared over the column communicator, so the leading
        # dimension must be the number of local samples.
        if shape[0] != self.n_samples:
            msg = f"When creating shared data '{name}' on the column communicator, "
            msg += f"the leading dimension should be the number of local "
            msg += f"samples ({self.n_samples}).  Shape given = {shape}"
            log.error(msg)
            raise RuntimeError(msg)
        shared_comm = self.dist.comm_col
        shared_comm_node = self.dist.comm_col_node
        shared_comm_rank_node = self.dist.comm_col_rank_node

        shared_dtype = dtype

        # Use defaults for dtype if not set
        if shared_dtype is None:
            shared_dtype = np.float64

        # Create the data object
        self._internal[name] = SharedDataType(
            MPIShared(
                shape,
                shared_dtype,
                shared_comm,
                comm_node=shared_comm_node,
                comm_node_rank=shared_comm_rank_node,
            ),
            "column",
        )
        self._accel_used[name] = False

        return

    def create_type(self, commtype, name, shape, dtype=None):
        """Create a shared memory buffer of the specified type.

        This is a convenience function that calls `create_group()`, `create_row()`,
        or `create_column()` based on the value of commtype.

        Args:
            commtype (str):  "group", "row", or "column".
            name (str): Name of the shared memory object (e.g. "boresight").
            shape (tuple): The shape of the new buffer.
            dtype (np.dtype): Use this dtype for each element.

        Returns:
            None

        """
        log = Logger.get()
        if name in self._internal:
            msg = "Observation data with name '{}' already exists.".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        if commtype == "row":
            self.create_row(name, shape, dtype=dtype)
        elif commtype == "column":
            self.create_column(name, shape, dtype=dtype)
        elif commtype == "group":
            self.create_group(name, shape, dtype=dtype)
        else:
            raise ValueError(f"Invalid communicator type '{commtype}'")

    # Get the comm type for a key

    def comm_type(self, key):
        """Return the communicator type for a key.

        The valid types are "group", "row", and "column".

        Args:
            key (str):  The object name.

        Returns:
            (str):  The communicator name over which it is shared.

        """
        return self._internal[key].type

    # Accelerator access

    # FIXME:  These objects are in MPI shared memory, so we should think more
    # carefully about whether we want each process doing these operations or
    # having one process copy in and other processes attach to the device ptr.

    # FIXME:  We should add a public parameter to MPIShared to access the
    # flat-packed data.

    def accel_exists(self, key):
        """Check if the named shared data exists on the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            (bool):  True if the data is present.

        """
        if not accel_enabled():
            return False
        log = Logger.get()
        if key not in self._internal:
            msg = f"Cannot check accelerator status of non-existent data {key}"
            log.error(msg)
            raise RuntimeError(msg)

        if use_accel_omp:
            result = accel_data_present(self._internal[key].shdata._flat, key)
        elif use_accel_jax:
            result = accel_data_present(self._internal[key].shdata.data)
        else:
            result = False

        log.verbose(f"SharedDataMgr {key} accel_exists = {result}")
        return result

    def accel_in_use(self, key):
        """Check if the shared device copy is the one currently in use.

        Args:
            key (str):  The object name.

        Returns:
            (bool):  True if the accelerator device copy is being used.

        """
        return self._accel_used[key]

    def accel_used(self, key, state):
        """Set the in-use state of the shared device copy.

        Setting the state to `True` is only possible if the data exists
        on the device.

        Args:
            key (str):  The object name.
            state (bool):  True if the device copy is in use, else False.

        Returns:
            None

        """
        if state and not self.accel_exists(key):
            log = Logger.get()
            msg = f"Data is not present on device, cannot set state to in-use"
            log.error(msg)
            raise RuntimeError(msg)
        self._accel_used[key] = state

    def accel_create(self, key):
        """Create the named shared data on the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        if key not in self._internal:
            msg = f"Cannot create non-existent data {key} on accelerator"
            log.error(msg)
            raise RuntimeError(msg)
        if self.accel_exists(key):
            log = Logger.get()
            msg = f"Data already exists on device, cannot create"
            log.error(msg)
            raise RuntimeError(msg)

        log.verbose(f"SharedDataMgr {key} accel_create")
        if use_accel_omp:
            _ = accel_data_create(self._internal[key].shdata._flat, key)
        elif use_accel_jax:
            self._internal[key].shdata.data = MutableJaxArray(
                self._internal[key].shdata.data
            )

    def accel_update_device(self, key):
        """Copy the named shared data to the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        if key not in self._internal:
            msg = f"Cannot copy non-existent data {key} to accelerator"
            log.error(msg)
            raise RuntimeError(msg)
        if not self.accel_exists(key):
            msg = f"Shared data '{key}' is not present on device, cannot update"
            log.error(msg)
            raise RuntimeError(msg)
        if self._accel_used[key]:
            # The active copy is on the device
            log = Logger.get()
            msg = f"Active data is already on device, cannot update"
            log.error(msg)
            raise RuntimeError(msg)

        log.verbose(f"SharedDataMgr {key} accel_update_device")
        if use_accel_omp:
            _ = accel_data_update_device(self._internal[key].shdata._flat, key)
        elif use_accel_jax:
            self._internal[key].shdata.data = MutableJaxArray(
                self._internal[key].shdata.data
            )

        self._accel_used[key] = True

    def accel_update_host(self, key):
        """Copy the named shared data from the accelerator to the host.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        log = Logger.get()
        if not accel_enabled():
            return
        if key not in self._internal:
            msg = f"Cannot copy non-existent data {key} from accelerator"
            log.error(msg)
            raise RuntimeError(msg)
        if not self.accel_exists(key):
            msg = f"Shared data '{key}' is not present on device, cannot copy to host"
            log.error(msg)
            raise RuntimeError(msg)
        if not self._accel_used[key]:
            # The active copy is on the host
            log = Logger.get()
            msg = f"Active data is already on host, cannot update"
            log.error(msg)
            raise RuntimeError(msg)

        log.verbose(f"SharedDataMgr {key} accel_update_host")
        if use_accel_omp:
            _ = accel_data_update_host(self._internal[key].shdata._flat, key)
        elif use_accel_jax:
            self._internal[key].shdata.data = accel_data_update_host(
                self._internal[key].shdata.data
            )

        self._accel_used[key] = False

    def accel_delete(self, key):
        """Delete the named data object on the device

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        log = Logger.get()
        if not accel_enabled():
            return
        if key not in self._internal:
            msg = f"Cannot delete non-existent data {key} on accelerator"
            log.error(msg)
            raise RuntimeError(msg)
        if not self.accel_exists(key):
            msg = f"Shared data '{key}' is not present on device, cannot delete"
            log.error(msg)
            raise RuntimeError(msg)

        log.verbose(f"SharedDataMgr {key} accel_delete")
        if use_accel_omp:
            _ = accel_data_delete(self._internal[key].shdata._flat, key)
        elif use_accel_jax:
            self._internal[key].shdata.data = accel_data_delete(
                self._internal[key].shdata.data
            )

        self._accel_used[key] = False

    def accel_clear(self):
        """Clear all data from accelerators

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        for key in self._internal:
            if self.accel_exists(key):
                log.verbose(f"SharedDataMgr {key} accel_delete")
                if use_accel_omp:
                    _ = accel_data_delete(self._internal[key].shdata._flat, key)
                elif use_accel_jax:
                    self._internal[key].shdata.data = accel_data_delete(
                        self._internal[key].shdata.data
                    )
            self._accel_used[key] = False

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key].shdata

    def __delitem__(self, key):
        if key in self._internal:
            self._internal[key].shdata.close()
            del self._internal[key]

    def _comm_from_type(self, commstr):
        """Get the comm from the type string"""
        if commstr == "row":
            return self.dist.comm_row
        elif commstr == "column":
            return self.dist.comm_col
        elif commstr == "group":
            return self.dist.comm.comm_group
        else:
            raise ValueError(f"Invalid communicator type '{commstr}'")

    def _valid_commtype(self, commstr, comm):
        """Helper function to check that a comm matches the specified type."""
        shcomm = self._comm_from_type(commstr)
        if comm_equivalent(shcomm, comm):
            return True
        else:
            return False

    def _assign_array(self, key, value, commtype):
        """Helper function to assign an array on one process to a shared object."""
        if key in self._internal:
            # Object already exists, so force the commtype
            commtype = self._internal[key].type

        # Get the communicator for this object
        shcomm = self._comm_from_type(commtype)

        # Verify that we only have incoming data on one process
        fromrank = 0
        myrank = 0
        if shcomm is not None:
            myrank = shcomm.rank
            nproc = shcomm.size
            check_rank = np.zeros(nproc, dtype=np.int32)
            check_result = np.zeros(nproc, dtype=np.int32)
            if value is not None:
                check_rank[myrank] = 1
            shcomm.Allreduce(check_rank, check_result, op=MPI.SUM)
            tot = np.sum(check_result)
            if tot > 1:
                msg = "When assigning an array to a shared object, only one process"
                msg += " should have a non-None value"
                raise ValueError(msg)
            fromrank = np.where(check_result == 1)[0][0]

        # Create the object if needed
        if key not in self._internal:
            shshape = None
            shdtype = None
            if myrank == fromrank:
                shshape = value.shape
                shdtype = value.dtype
            if shcomm is not None:
                shshape = shcomm.bcast(shshape, root=fromrank)
                shdtype = shcomm.bcast(shdtype, root=fromrank)
            self.create_type(commtype, key, shshape, dtype=shdtype)

        # Assign
        off = None
        if myrank == fromrank:
            if value.shape != self._internal[key].shdata.shape:
                msg = "When assigning directly to a shared object, the value "
                msg += "must have the same dimensions"
                raise ValueError(msg)
            off = tuple([0 for x in self._internal[key].shdata.shape])
        self._internal[key].shdata.set(value, offset=off, fromrank=fromrank)

    def assign_mpishared(self, key, value, commtype):
        """Helper function to assign an existing MPIShared data object."""
        log = Logger.get()
        if key not in self._internal:
            # Create the object, and check that value comm is correct.
            if not self._valid_commtype(commtype, value.comm):
                msg = f"Assignment value for '{key}' ('{commtype}') "
                msg += "has incorrect communicator."
                log.error(msg)
                raise RuntimeError(msg)
            self.create_type(commtype, key, value.shape, dtype=value.dtype)
        else:
            # Verify that communicators and dimensions match
            if not self._valid_commtype(self._internal[key].type, value.comm):
                msg = "Direct assignment object must have equivalent communicator."
                log.error(msg)
                raise RuntimeError(msg)
            if value.shape != self._internal[key].shdata.shape:
                msg = "Direct assignment of shared object must have the same shape."
                log.error(msg)
                raise RuntimeError(msg)
            if value.dtype != self._internal[key].shdata.dtype:
                msg = "Direct assignment of shared object must have the same dtype."
                log.error(msg)
                raise RuntimeError(msg)

        # Assign data from just one process.
        offset = None
        dval = None
        if value.comm is None or value.comm.rank == 0:
            offset = tuple([0 for x in self._internal[key].shdata.shape])
            dval = value.data
        self._internal[key].shdata.set(dval, offset=offset, fromrank=0)

    def __setitem__(self, key, value):
        log = Logger.get()
        if key in self._internal:
            # This is an existing shared object.
            if isinstance(value, MPIShared):
                self.assign_mpishared(key, value, self._internal[key].type)
            else:
                # This must be an array on exactly one process.
                self._assign_array(key, value, self._internal[key].type)
        else:
            # Creating a new object
            if isinstance(value, (tuple, list, SharedDataType)) and len(value) == 2:
                # We are specifying the data and the comm type
                dvalue = value[0]
                tvalue = value[1]
                if isinstance(dvalue, MPIShared):
                    # The passed value is already an MPIShared object
                    self.assign_mpishared(key, dvalue, tvalue)
                else:
                    # This is a simple array on one process
                    self._assign_array(key, dvalue, tvalue)
            else:
                # We are specifying just the data, so the comm type
                # is assumed to be the group comm
                if isinstance(value, MPIShared):
                    self.assign_mpishared(key, value, "group")
                else:
                    # This is a simple array on one process
                    self._assign_array(key, value, "group")

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        self.accel_clear()
        for k in self._internal.keys():
            self._internal[k].shdata.close()

    def __del__(self):
        if hasattr(self, "_internal"):
            self.clear()

    def __eq__(self, other):
        log = Logger.get()
        if self.n_detectors != other.n_detectors:
            log.verbose(f"  n_detectors {self.n_detectors} != {other.n_detectors}")
            return False
        if self.n_samples != other.n_samples:
            log.verbose(f"  n_detectors {self.n_samples} != {other.n_samples}")
            return False
        if not comm_equivalent(self.dist.comm.comm_group, other.dist.comm.comm_group):
            log.verbose(f"  comm_group not equivalent")
            return False
        if not comm_equivalent(self.dist.comm_row, other.dist.comm_row):
            log.verbose(f"  comm_row not equivalent")
            return False
        if not comm_equivalent(self.dist.comm_col, other.dist.comm_col):
            log.verbose(f"  comm_col not equivalent")
            return False
        if set(self._internal.keys()) != set(other._internal.keys()):
            log.verbose(
                f"  keys {set(self._internal.keys())} != {set(other._internal.keys())}"
            )
            return False
        for k in self._internal.keys():
            if self._internal[k].shdata.shape != other._internal[k].shdata.shape:
                log.verbose(
                    f"  key {k} shape {self._internal[k].shdata.shape} != {other._internal[k].shdata.shape}"
                )
                return False
            if self._internal[k].shdata.dtype != other._internal[k].shdata.dtype:
                log.verbose(
                    f"  key {k} dtype {self._internal[k].shdata.dtype} != {other._internal[k].shdata.dtype}"
                )
                return False
            if not comm_equivalent(
                self._internal[k].shdata.comm, other._internal[k].shdata.comm
            ):
                log.verbose(f"  key {k} comms not equivalent")
                return False
            if self._internal[k].type != other._internal[k].type:
                log.verbose(f"  key {k} comm types not equal")
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def memory_use(self):
        bytes = 0
        for k in self._internal.keys():
            shared_bytes = 0
            node_bytes = 0
            node_rank = 0
            if self._internal[k].shdata.nodecomm is not None:
                node_rank = self._internal[k].shdata.nodecomm.rank
            if node_rank == 0:
                node_elems = 1
                for d in self._internal[k].shdata.shape:
                    node_elems *= d
                node_bytes += node_elems * self._internal[k].shdata.data.itemsize
            if self._internal[k].shdata.comm is None:
                shared_bytes = node_bytes
            else:
                shared_bytes = self._internal[k].shdata.comm.allreduce(
                    node_bytes, op=MPI.SUM
                )
            bytes += shared_bytes
        return bytes

    def __repr__(self):
        val = "<SharedDataManager"
        for k in self._internal.keys():
            val += f"\n    {k} ({self._internal[k].type}): "
            val += f"shape={self._internal[k].shdata.shape}, "
            val += f"dtype={self._internal[k].shdata.dtype}"
        val += ">"
        return val


class IntervalsManager(MutableMapping):
    """Class for creating and storing interval lists in an observation.

    Named lists of intervals are accessed by dictionary style syntax ([] brackets).
    When making new interval lists, these can be added directly on each process, or
    some helper functions can be used to create the appropriate local interval lists
    given a global set of ranges.

    Args:
        dist (DistDetSamp):  The observation data distribution.

    """

    # This could be anything, just has to be unique
    all_name = "ALL_OBSERVATION_SAMPLES"

    def __init__(self, dist, local_samples):
        self.comm = dist.comm
        self.comm_col = dist.comm_col
        self.comm_row = dist.comm_row
        self._internal = dict()
        self._del_callbacks = dict()
        self._local_samples = local_samples
        # Trigger creation of the internal interval list for all samples
        _ = self._real_key(None)

    def create_col(self, name, global_timespans, local_times, fromrank=0):
        """Create local interval lists on the same process column.

        Processes within the same column of the observation data distribution have the
        same local time range.  This function takes the global time ranges provided,
        computes the intersection with the local time range of this process column,
        and creates a local named interval list on each process in the column.

        Args:
            name (str):  The key to use in the local intervals dictionary.
            global_times (list):  List of start, stop tuples containing time ranges
                within the observation.
            local_times (array):  The local timestamps on this process.
            fromrank (int):  Get the list from this process rank of the observation
                column communicator.  Input arguments on other processes are ignored.

        """
        if self.comm_col is not None:
            # Broadcast to all processes in this column
            n_global = 0
            if global_timespans is not None:
                n_global = len(global_timespans)
            n_global = self.comm_col.bcast(n_global, root=fromrank)
            if n_global == 0:
                global_timespans = list()
            else:
                global_timespans = self.comm_col.bcast(global_timespans, root=fromrank)
        # Every process creates local intervals
        lt = local_times
        if isinstance(lt, MPIShared):
            lt = local_times.data
        self._internal[name] = IntervalList(lt, timespans=global_timespans)

    def create(self, name, global_timespans, local_times, fromrank=0):
        """Create local interval lists from global time ranges on one process.

        In some situations, a single process has loaded data from the disk, queried a
        database, etc and has information about some time spans that are global across
        the observation.  This function automatically creates the named local interval
        list consisting of the intersection of the local sample range with these global
        intervals.

        Args:
            name (str):  The key to use in the local intervals dictionary.
            global_timespans (list):  List of start, stop tuples containing time ranges
                within the observation.
            local_times (array):  The local timestamps on this process.
            fromrank (int):  Get the list from this process rank of the observation
                communicator.  Input arguments on other processes are ignored.

        """
        send_col_rank = 0
        send_row_rank = 0
        if self.comm.comm_group is not None:
            col_rank = 0
            if self.comm_col is not None:
                col_rank = self.comm_col.rank
            # Find the process grid ranks of the incoming data
            if self.comm.group_rank == fromrank:
                if self.comm_col is not None:
                    send_col_rank = self.comm_col.rank
                if self.comm_row is not None:
                    send_row_rank = self.comm_row.rank
            send_col_rank = self.comm.comm_group.bcast(send_col_rank, root=0)
            send_row_rank = self.comm.comm_group.bcast(send_row_rank, root=0)
            # Broadcast data along the row
            if col_rank == send_col_rank:
                if self.comm_row is not None:
                    n_global = 0
                    if global_timespans is not None:
                        n_global = len(global_timespans)
                    n_global = self.comm_row.bcast(n_global, root=send_row_rank)
                    if n_global == 0:
                        global_timespans = list()
                    else:
                        global_timespans = self.comm_row.bcast(
                            global_timespans, root=send_row_rank
                        )
        # Every process column creates their local intervals
        self.create_col(name, global_timespans, local_times, fromrank=send_col_rank)

    def register_delete_callback(self, key, fn):
        self._del_callbacks[key] = fn

    # Mapping methods

    def _real_key(self, key):
        if key is None or key == self.all_name:
            if self.all_name not in self._internal:
                # Create fake intervals
                faketimes = -1.0 * np.ones(self._local_samples, dtype=np.float64)
                self._internal[self.all_name] = IntervalList(
                    faketimes, samplespans=[(0, self._local_samples - 1)]
                )
            return self.all_name
        else:
            return key

    def __getitem__(self, key):
        key = self._real_key(key)
        return self._internal[key]

    def __delitem__(self, key):
        key = self._real_key(key)
        if key in self._del_callbacks:
            try:
                self._del_callbacks[key](key)
                del self._del_callbacks[key]
            except:
                pass
        if key in self._internal:
            del self._internal[key]

    def __setitem__(self, key, value):
        if not isinstance(value, IntervalList):
            raise ValueError("Value must be an IntervalList instance.")
        self._internal[key] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        self._internal.clear()

    def __del__(self):
        if hasattr(self, "_internal"):
            self.clear()

    def __repr__(self):
        val = "<IntervalsManager {} lists".format(len(self._internal))
        for k in self._internal.keys():
            if k != self.all_name:
                val += "\n  {}: {} intervals".format(k, len(self._internal[k]))
        val += ">"
        return val

    def __eq__(self, other):
        log = Logger.get()
        if not comm_equivalent(self.comm.comm_group, other.comm.comm_group):
            log.verbose(f"  comm not equivalent")
            return False
        if not comm_equivalent(self.comm_row, other.comm_row):
            log.verbose(f"  comm_row not equivalent")
            return False
        if not comm_equivalent(self.comm_col, other.comm_col):
            log.verbose(f"  comm_col not equivalent")
            return False
        this_set = set(self._internal.keys())
        try:
            this_set.remove(self.all_name)
        except KeyError:
            pass
        other_set = set(other._internal.keys())
        try:
            other_set.remove(self.all_name)
        except KeyError:
            pass

        if this_set != other_set:
            log.verbose(f"  keys {self._internal.keys()} != {other._internal.keys()}")
            return False
        for k in this_set:
            if self._internal[k] != other._internal[k]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def accel_exists(self, key):
        """Check if the named interval list exists on the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            (bool):  True if the data is present.

        """
        if not accel_enabled():
            return False
        log = Logger.get()
        result = self[key].accel_exists()
        log.verbose(f"IntervalsManager {key} accel_exists = {result}")
        return result

    def accel_in_use(self, key):
        """Check if the interval list device copy is the one currently in use.

        Args:
            key (str):  The object name.

        Returns:
            (bool):  True if the accelerator device copy is being used.

        """
        return self[key].accel_in_use()

    def accel_used(self, key, state):
        """Set the in-use state of the interval list device copy.

        Setting the state to `True` is only possible if the data exists
        on the device.

        Args:
            key (str):  The object name.
            state (bool):  True if the device copy is in use, else False.

        Returns:
            None

        """
        self[key].accel_used(state)

    def accel_create(self, key):
        """Create the named interval list on the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        log.verbose(f"IntervalsManager {key} accel_create")
        if key is None:
            # This is the special interval list of the full range
            self[key].accel_create("None")
        else:
            self[key].accel_create(key)

    def accel_update_device(self, key):
        """Copy the named interval list to the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        log.verbose(f"IntervalsManager {key} accel_update_device")
        self[key].accel_update_device()

    def accel_update_host(self, key):
        """Copy the named interval list from the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        log.verbose(f"IntervalsManager {key} accel_update_host")
        self[key].accel_update_host()

    def accel_delete(self, key):
        """Delete the named interval list from the accelerator.

        Args:
            key (str):  The object name.

        Returns:
            None

        """
        log = Logger.get()
        if not accel_enabled():
            return
        if not self[key].accel_exists():
            msg = f"Intervals list '{key}' is not present on device, cannot delete"
            log.error(msg)
            raise RuntimeError(msg)
        log.verbose(f"IntervalsManager {key} accel_delete")
        self[key].accel_delete()

    def accel_clear(self):
        """Clear all interval lists from accelerators

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()
        for key in self._internal:
            if self[key].accel_exists():
                log.verbose(f"IntervalsManager {key} accel_delete")
                self[key].accel_delete()
