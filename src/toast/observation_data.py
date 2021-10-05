# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys

from collections.abc import MutableMapping, Mapping

import numpy as np

from astropy import units as u

from pshmem import MPIShared

from .mpi import MPI, comm_equivalent, comm_equal

from .utils import (
    Logger,
    AlignedI8,
    AlignedU8,
    AlignedI16,
    AlignedU16,
    AlignedI32,
    AlignedU32,
    AlignedI64,
    AlignedU64,
    AlignedF32,
    AlignedF64,
    dtype_to_aligned,
)

from .intervals import IntervalList


class DetectorData(object):
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
        self._detectors = detectors
        if len(self._detectors) == 0:
            msg = "You must specify a list of at least one detector name"
            log.error(msg)
            raise ValueError(msg)
        self._name2idx = {y: x for x, y in enumerate(self._detectors)}

    def _data_props(self, detectors, detshape, dtype):
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
        self._fullsize = self._flatshape
        self._memsize = self.itemsize * self._fullsize
        self._raw = self._storage_class.zeros(self._fullsize)
        self._flatdata = self._raw.array()[: self._flatshape]
        self._data = self._flatdata.reshape(self._shape)

    @property
    def detectors(self):
        return list(self._detectors)

    def keys(self):
        return list(self._detectors)

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
        if not hasattr(self, "_data"):
            raise RuntimeError("Cannot use DetectorData object after clearing memory")
        return self._data

    @property
    def flatdata(self):
        return self._flatdata

    def change_detectors(self, detectors):
        """Modify the list of detectors.

        This attempts to re-use the underlying memory and just change the detector
        mapping to that memory.  This is useful if memory allocation is expensive.
        If the new list of detectors is longer than the original, a new memory buffer
        is allocated.  If the new list of detectors is shorter than the original, the
        buffer is kept and only a subset is used.

        Args:
            detectors (list):  A list of detector names in exactly the order you wish.

        Returns:
            None

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
        else:
            # We can re-use the existing memory
            self._shape = shp
            self._flatshape = flatshape
            self._flatdata = self._raw.array()[: self._flatshape]
            self._flatdata[:] = 0
            self._data = self._flatdata.reshape(self._shape)

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        if hasattr(self, "_data"):
            del self._data
        if not self._is_view:
            if hasattr(self, "_flatdata"):
                del self._flatdata
            if hasattr(self, "_raw"):
                self._raw.clear()
                del self._raw

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
                view = list()
                for k in key:
                    view.append(self._name2idx[k])
                view = tuple(view)
            except TypeError:
                log = Logger.get()
                msg = "Detector indexing supports slice, int, string or iterable, not '{}'".format(
                    key
                )
                log.error(msg)
                raise TypeError(msg)
        return view

    def _get_view(self, key):
        if isinstance(key, (tuple, Mapping)):
            # We are slicing in both detector and sample dimensions
            if len(key) > len(self._shape):
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
        val += " {} detectors each with shape {} and type {}:".format(
            len(self._detectors), self._shape[1:], self._dtype
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
        if self.detectors != other.detectors:
            return False
        if self.dtype.char != other.dtype.char:
            return False
        if self.shape != other.shape:
            return False
        if self.units != other.units:
            return False
        if not np.allclose(self.data, other.data):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


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
        units=u.dimensionless_unscaled,
    ):
        """Ensure that the observation has the named detector data.

        If the named detdata object does not exist, it is created.  If it does exist
        and the sample shape and dtype are compatible, then it is checked whether the
        specified detectors are already included.  If not, it calls the
        DetectorData.change_detectors() method to re-use this existing memory buffer if
        possible.

        Args:
            name (str): The name of the detector data (signal, flags, etc)
            sample_shape (tuple): Use this shape for the data of each detector sample.
                Use None or an empty tuple if you want one element per sample.
            dtype (np.dtype): Use this dtype for each element.
            detectors (list):  Ensure that these detectors exist in the object.
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
            if units != self._internal[name].units:
                msg = "Detector data '{}' already exists with units {}.".format(
                    name, self._internal[name].units
                )
                log.error(msg)
                raise RuntimeError(msg)
            # Ok, we can re-use this.  Are the detectors already included in the data?
            change = False
            for d in detectors:
                if d not in self._internal[name].detectors:
                    change = True
            if change:
                self._internal[name].change_detectors(detectors)
        else:
            # Create the data object
            self.create(
                name,
                sample_shape=sample_shape,
                dtype=dtype,
                detectors=detectors,
                units=units,
            )

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        if key in self._internal:
            self._internal[key].clear()
            del self._internal[key]

    def __setitem__(self, key, value):
        if isinstance(value, DetectorData):
            # We have an input detector data object.  Verify dimensions
            for d in value.detectors:
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)
            if value.shape[1] != self.samples:
                msg = "Assignment DetectorData object has {} samples instead of {} in the observation".format(
                    value.shape[1], self.samples
                )
                raise ValueError(msg)
            if key not in self._internal:
                # Create it first
                self.create(
                    key,
                    sample_shape=value.detector_shape[1:],
                    dtype=value.dtype,
                    detectors=value.detectors,
                )
            else:
                if value.detector_shape != self._internal[key].detector_shape:
                    msg = "Assignment value has wrong detector shape"
                    raise ValueError(msg)
            for d in value.detectors:
                self._internal[key][d] = value[d]
        elif isinstance(value, Mapping):
            # This is a dictionary of detector arrays
            sample_shape = None
            dtype = None
            for d, ddata in value.items():
                if d not in self.detectors:
                    msg = "detector '{}' not in this observation".format(d)
                    raise ValueError(msg)
                if ddata.shape[0] != self.samples:
                    msg = "Assigment dictionary detector {} has {} samples instead of {} in the observation".format(
                        d, ddata.shape[0], self.samples
                    )
                    raise ValueError(msg)
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
                )
            else:
                if (self.samples,) + sample_shape != self._internal[key].detector_shape:
                    msg = "Assignment value has wrong detector shape"
                    raise ValueError(msg)
            for d, ddata in value.items():
                self._internal[key][d] = ddata
        else:
            # This must be just an array- verify the dimensions
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
                    )
                else:
                    fullshape = (self.samples,)
                    if sample_shape is not None:
                        fullshape += sample_shape
                    if fullshape != self._internal[key].detector_shape:
                        msg = "Assignment value has wrong detector shape"
                        raise ValueError(msg)
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
                    )
                else:
                    fullshape = (self.samples,)
                    if sample_shape is not None:
                        fullshape += sample_shape
                    if fullshape != self._internal[key].detector_shape:
                        msg = "Assignment value has wrong detector shape"
                        raise ValueError(msg)
                self._internal[key][:] = value
            else:
                # Incompatible
                msg = "Assignment of detector data from an array only supports full size or single detector"
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
            val += "\n    {}: shape = {}, dtype = {}".format(
                k, self._internal[k].shape, self._internal[k].dtype
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
            # if self._internal[k] != other._internal[k]:
            if not np.allclose(self._internal[k], other._internal[k]):
                log.verbose(
                    f"  detector data {k} not equal:  {self._internal[k]} != {other._internal[k]}"
                )
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class SharedDataManager(MutableMapping):
    """Class used to manage shared data objects in an Observation.

    New objects can be created with the "create()" method:

        obs.shared.create(name, shape=None, dtype=None, comm=None)

    The communicator defaults to sharing the data across the observation comm, but
    other options would be to pass in the observation comm_row or comm_col communicators
    in order to share common detector information across the process grid row or to
    share telescope data across the process grid column.

    You can also create shared objects by assignment from an existing MPIShared object
    or an array on one process.  In the case of creating from an array assignment, an
    extra communication step is required to determine what process is sending the data
    (all processes except for one should pass 'None' as the data).  For example:

        timestamps = None
        if obs.comm_col_rank == 0:
            # Input data only exists on one process
            timestamps = np.arange(obs.n_local_samples, dtype=np.float32)

        # Explicitly create the shared data and assign:
        obs.shared.create(
            "times",
            shape=(obs.n_local_samples,),
            dtype=np.float32,
            comm=obs.comm_col
        )
        obs.shared["times"].set(timestamps, offset=(0,), fromrank=0)

        # Create from existing MPIShared object:
        sharedtime = MPIShared((obs.n_local_samples,), np.float32, obs.comm_col)
        sharedtime[:] = timestamps
        obs.shared["times"] = sharedtime

        # Create from array on one process, pre-communication needed:
        obs.shared["times"] = timestamps

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
        self._internal = dict()

    def create(self, name, shape, dtype=None, comm=None):
        """Create a shared memory buffer.

        This buffer will be replicated across all nodes used by the processes owning
        the observation.  This uses the MPIShared class, which falls back to a simple
        numpy array if MPI is not being used.

        Args:
            name (str): Name of the shared memory object (e.g. "boresight").
            shape (tuple): The shape of the new buffer.
            dtype (np.dtype): Use this dtype for each element.
            comm (MPI.Comm): The communicator to use for the shared data.  If None
                then the communicator for the observation is used.  Other options
                would be to specify the grid_comm_row (for shared detector objects) or
                grid_comm_col (for shared timestream objects).

        Returns:
            None

        """
        log = Logger.get()
        if name in self._internal:
            msg = "Observation data with name '{}' already exists.".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        shared_comm = comm
        shared_comm_node = None
        shared_comm_rank_node = None
        if shared_comm is None:
            # Use the observation group communicator.
            shared_comm = self.dist.comm.comm_group
            shared_comm_node = self.dist.comm.comm_group_node
            shared_comm_rank_node = self.dist.comm.comm_group_node_rank

        if comm_equal(self.dist.comm.comm_group, shared_comm):
            # This is shared over the whole observation communicator, so it
            # may have any shape.
            shared_comm_node = self.dist.comm.comm_group_node
            shared_comm_rank_node = self.dist.comm.comm_group_node_rank
        elif comm_equal(self.dist.comm_row, shared_comm):
            # This is shared over the row communicator, so the leading
            # dimension must be the number of local detectors.
            if shape[0] != self.n_detectors:
                msg = f"When creating shared data '{name}' on the row communicator, "
                msg += f"the leading dimension should be the number of local "
                msg += f"detectors ({self.n_detectors}).  Shape given = {shape}."
                log.error(msg)
                raise RuntimeError(msg)
            shared_comm_node = self.dist.comm_row_node
            shared_comm_rank_node = self.dist.comm_row_rank_node
        elif comm_equal(self.dist.comm_col, shared_comm):
            # This is shared over the column communicator, so the leading
            # dimension must be the number of local samples.
            if shape[0] != self.n_samples:
                msg = f"When creating shared data '{name}' on the column communicator, "
                msg += f"the leading dimension should be the number of local "
                msg += f"samples ({self.n_samples}).  Shape given = {shape}"
                log.error(msg)
                raise RuntimeError(msg)
            shared_comm_node = self.dist.comm_col_node
            shared_comm_rank_node = self.dist.comm_col_rank_node
        else:
            msg = f"When creating shared data '{name}', you must use either the "
            msg += "observation communicator or the row or column communicators."
            log.error(msg)
            raise RuntimeError(msg)

        shared_dtype = dtype

        # Use defaults for dtype if not set
        if shared_dtype is None:
            shared_dtype = np.float64

        # Create the data object
        self._internal[name] = MPIShared(
            shape,
            shared_dtype,
            shared_comm,
            comm_node=shared_comm_node,
            comm_node_rank=shared_comm_rank_node,
        )

        return

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        if key in self._internal:
            self._internal[key].close()
            del self._internal[key]

    def __setitem__(self, key, value):
        log = Logger.get()
        if isinstance(value, MPIShared):
            # This is an existing shared object.
            if key not in self._internal:
                self.create(
                    key,
                    shape=value.shape,
                    dtype=value.dtype,
                    comm=self.dist.comm.comm_group,
                )
            else:
                # Verify that communicators and dimensions match
                if value.shape != self._internal[key].shape:
                    msg = "Direct assignment of shared object must have the same shape."
                    log.error(msg)
                    raise RuntimeError(msg)
                if value.dtype != self._internal[key].dtype:
                    msg = "Direct assignment of shared object must have the same dtype."
                    log.error(msg)
                    raise RuntimeError(msg)
                if not comm_equivalent(value.comm, self._internal[key].comm):
                    msg = "Direct assignment object must have equivalent communicator."
                    log.error(msg)
                    raise RuntimeError(msg)
            # Assign from just one process.
            offset = None
            dval = None
            if value.comm is None or value.comm.rank == 0:
                offset = tuple([0 for x in self._internal[key].shape])
                dval = value.data
            self._internal[key].set(dval, offset=offset, fromrank=0)
        else:
            # This must be an array on one process.
            if key not in self._internal:
                # We need to create it.  In that case we use the default communicator
                # (the full observation comm).  We also need to get the array
                # properties to all processes in order to create the object.
                if self.dist.comm.comm_group is None:
                    # No MPI
                    self.create(key, shape=value.shape, dtype=value.dtype)
                    offset = tuple([0 for x in self._internal[key].shape])
                    self._internal[key].set(value, offset=offset, fromrank=0)
                else:
                    shp = None
                    dt = None
                    check_rank = np.zeros((self.dist.comm.group_size,), dtype=np.int32)
                    check_result = np.zeros(
                        (self.dist.comm.group_size,), dtype=np.int32
                    )
                    if value is not None:
                        shp = value.shape
                        dt = value.dtype
                        check_rank[self.dist.comm.group_rank] = 1
                    self.dist.comm.comm_group.Allreduce(
                        check_rank, check_result, op=MPI.SUM
                    )
                    tot = np.sum(check_result)
                    if tot > 1:
                        msg = "When creating shared data with [] notation, only "
                        msg += "one process may have a non-None value for the data"
                        log.error_rank(msg)
                        raise RuntimeError(msg)

                    from_rank = np.where(check_result == 1)[0][0]
                    shp = self.dist.comm.comm_group.bcast(shp, root=from_rank)
                    dt = self.dist.comm.comm_group.bcast(dt, root=from_rank)
                    self.create(key, shape=shp, dtype=dt)
                    offset = None
                    if self.dist.comm.group_rank == from_rank:
                        offset = tuple([0 for x in self._internal[key].shape])
                    self._internal[key].set(value, offset=offset, fromrank=from_rank)
            else:
                # Already exists, just do the assignment
                slc = None
                if value is not None:
                    if value.shape != self._internal[key].shape:
                        raise ValueError(
                            "When assigning directly to a shared object, the value must have the same dimensions"
                        )
                    slc = tuple([slice(0, x) for x in self._internal[key].shape])
                self._internal[key][slc] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        for k in self._internal.keys():
            self._internal[k].close()

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
            log.verbose(f"  keys {self._internal.keys()} != {other._internal.keys()}")
            return False
        for k in self._internal.keys():
            if self._internal[k].shape != other._internal[k].shape:
                log.verbose(
                    f"  key {k} shape {self._internal[k].shape} != {other._internal[k].shape}"
                )
                return False
            if self._internal[k].dtype != other._internal[k].dtype:
                log.verbose(
                    f"  key {k} dtype {self._internal[k].dtype} != {other._internal[k].dtype}"
                )
                return False
            if not comm_equivalent(self._internal[k].comm, other._internal[k].comm):
                log.verbose(f"  key {k} shape comms not equivalent")
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
            if self._internal[k].nodecomm is not None:
                node_rank = self._internal[k].nodecomm.rank
            if node_rank == 0:
                node_elems = 1
                for d in self._internal[k].shape:
                    node_elems *= d
                node_bytes += node_elems * self._internal[k].data.itemsize
            if self._internal[k].comm is None:
                shared_bytes = node_bytes
            else:
                shared_bytes = self._internal[k].comm.allreduce(node_bytes, op=MPI.SUM)
            bytes += shared_bytes
        return bytes

    def __repr__(self):
        val = "<SharedDataManager"
        for k in self._internal.keys():
            val += "\n    {}: shape = {}, dtype = {}".format(
                k, self._internal[k].shape, self._internal[k].dtype
            )
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

    def __init__(self, dist):
        self.comm = dist.comm
        self.comm_col = dist.comm_col
        self.comm_row = dist.comm_row
        self._internal = dict()
        self._del_callbacks = dict()

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
                    global_timespans = self.comm_row.bcast(
                        global_timespans, root=send_row_rank
                    )
        # Every process column creates their local intervals
        self.create_col(name, global_timespans, local_times, fromrank=send_col_rank)

    def register_delete_callback(self, key, fn):
        self._del_callbacks[key] = fn

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
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
        if set(self._internal.keys()) != set(other._internal.keys()):
            log.verbose(f"  keys {self._internal.keys()} != {other._internal.keys()}")
            return False
        for k in self._internal.keys():
            if self._internal[k] != other._internal[k]:
                log.verbose(f"  key {k} interval lists not equal")
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
