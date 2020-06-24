# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys

import numbers

from collections.abc import MutableMapping, Sequence

import numpy as np

from .mpi import MPI

from .instrument import Telescope, Focalplane

from .intervals import IntervalList

from .dist import distribute_samples

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
    name_UID,
)

from pshmem import MPIShared

from .cuda import use_pycuda


class DetectorData(object):
    """Class representing a logical collection of co-sampled detector data.

    This class works like an array of detector data where the first dimension is the
    number of detectors and the second dimension is the data for that detector.  The
    data for a particular detector may itself be multi-dimensional, with the first
    dimension the number of samples.

    The data in this container may be sliced by both detector indices and also by
    detector name.

    Example:
        Imagine we have 3 detectors and each has 10 samples.  We want to store a
        4-element value at each sample using 4-byte floats.  We would do::

            detdata = DetectorData(["d01", "d02", "d03"], (10, 4), np.float32)

        and then we can access the data for an individual detector either by index
        or by name with::

            detdata["d01"] = np.ones((10, 4), dtype=np.float32)
            firstdet = detdata[0]

        slicing by index and by a list of detectors is possible::

            view = detdata[0:-1]
            view = detdata[("d01", "d03")]

    Args:
        detectors (list):  A list of detector names in exactly the order you wish.
            This order is fixed for the life of the object.
        shape (tuple):  The shape of the data *for each detector*.  The first element
            of this shape should be the number of samples.
        dtype (numpy.dtype):  A numpy-compatible dtype for each element of the detector
            data.  The only supported types are 1, 2, 4, and 8 byte signed and unsigned
            integers, 4 and 8 byte floating point numbers, and 4 and 8 byte complex
            numbers.

    """

    def __init__(self, detectors, shape, dtype):
        log = Logger.get()

        self._detectors = detectors
        if len(self._detectors) == 0:
            msg = "You must specify a list of at least one detector name"
            log.error(msg)
            raise ValueError(msg)

        self._name2idx = {y: x for x, y in enumerate(self._detectors)}

        # construct a new dtype in case the parameter given is shortcut string
        ttype = np.dtype(dtype)

        self._storage_class = None
        if ttype.char == "b":
            self._storage_class = AlignedI8
        elif ttype.char == "B":
            self._storage_class = AlignedU8
        elif ttype.char == "h":
            self._storage_class = AlignedI16
        elif ttype.char == "H":
            self._storage_class = AlignedU16
        elif ttype.char == "i":
            self._storage_class = AlignedI32
        elif ttype.char == "I":
            self._storage_class = AlignedU32
        elif (ttype.char == "q") or (ttype.char == "l"):
            self._storage_class = AlignedI64
        elif (ttype.char == "Q") or (ttype.char == "L"):
            self._storage_class = AlignedU64
        elif ttype.char == "f":
            self._storage_class = AlignedF32
        elif ttype.char == "d":
            self._storage_class = AlignedF64
        elif ttype.char == "F":
            raise NotImplementedError("No support yet for complex numbers")
        elif ttype.char == "D":
            raise NotImplementedError("No support yet for complex numbers")
        else:
            msg = "Unsupported data typecode '{}'".format(ttype.char)
            log.error(msg)
            raise ValueError(msg)
        self._dtype = ttype

        # Verify that our shape contains only integral values
        self._flatshape = len(self._detectors)
        for d in shape:
            if not isinstance(d, (int, np.integer)):
                msg = "input shape contains non-integer values"
                log.error(msg)
                raise ValueError(msg)
            self._flatshape *= d

        shp = [len(self._detectors)]
        shp.extend(shape)
        self._shape = tuple(shp)
        self._raw = self._storage_class.zeros(self._flatshape)
        self._data = self._raw.array().reshape(self._shape)

    @property
    def detectors(self):
        return list(self._detectors)

    def keys(self):
        return list(self._detectors)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def detector_shape(self):
        return tuple(self._shape[1:])

    @property
    def data(self):
        return self._data

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        if hasattr(self, "_data"):
            del self._data
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
            except TypeError:
                log = Logger.get()
                msg = "Detector indexing supports slice, int, string or iterable, not '{}'".format(
                    key
                )
                log.error(msg)
                raise TypeError(msg)
        return view

    def _get_view(self, key):
        if isinstance(key, tuple):
            # We are slicing in both detector and sample dimensions
            if len(key) > 2:
                msg = "DetectorData has only 2 dimensions"
                log.error(msg)
                raise TypeError(msg)
            if len(key) == 1:
                # Only detector slice
                return self._det_axis_view(key[0])
            else:
                detview = self._det_axis_view(key[0])
                return detview, key[1]
        else:
            # Only detector slice
            return self._det_axis_view(key)

    def __getitem__(self, key):
        view = self._get_view(key)
        return np.array(self._data[view], dtype=self._dtype, copy=False)

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete individual elements")
        return

    def __setitem__(self, key, value):
        view = self._get_view(key)
        self._data[view] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._detectors)

    def __repr__(self):
        val = "<DetectorData {} detectors each with shape {} and type {}:".format(
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


class DetDataMgr(MutableMapping):
    """Class used to manage DetectorData objects in an Observation.

    New objects can be created with the "create()" method:

        ob.detdata.create(name, original=None, shape=None, dtype=None, detectors=None)

    Which supports copying data from an existing DetectorData object or a dictionary
    of individual detector timestreams.  The list of detectors to store can also be
    reduced from the full set in the observation by giving a list of detectors.

    After creation, you can access a given DetectorData object by name with standard
    dictionary syntax:

        ob.detdata[name]

    And delete it as well:

        del ob.detdata[name]

    There are shortcut methods to create standard data products:

        ob.detdata.create_signal()
        ob.detdata.create_flags()

    """

    def __init__(self, samples, detectors):
        self.samples = samples
        self.detectors = detectors
        self._internal = dict()

    def create(self, name, original=None, shape=None, dtype=None, detectors=None):
        """Create a local DetectorData buffer on this process.

        This method can be used to create arrays of detector data for storing signal,
        flags, or other timestream products on each process.

        Args:
            name (str): The name of the detector data (signal, flags, etc)
            original (DetectorData, dict): Copy an existing data object.  This can
                either be another DetectorData object or a dictionary of arrays.  This
                must have exactly the same detectors as the local detector list,
                although the ordering may be different.
            shape (tuple): If not constructing from an existing array, use this shape
                for the data of each detector.
            dtype (np.dtype): If not constructing from an existing array, use this
                dtype for each element.
            detectors (list):  Only construct a data object for this set of detectors.
                This is useful if creating temporary data within a pipeline working
                on a subset of detectors.

        Returns:
            None

        """
        if detectors is None:
            detectors = self.detectors

        log = Logger.get()
        if name in self._internal:
            msg = "Detector data with name '{}' already exists.".format(name)
            log.error(msg)
            raise RuntimeError(msg)

        data_shape = shape
        data_dtype = dtype
        if original is not None:
            # We are copying input data.  Ensure that the detector lists are the same
            # and that the shapes and types of every array are the same.
            data_dets = original.keys()
            if set(data_dets) != set(detectors):
                msg = "Input data to copy has a different detector list"
                log.error(msg)
                raise RuntimeError(msg)
            data_shape = None
            data_dtype = None
            for d in data_dets:
                if data_shape is None:
                    data_shape = original[d].shape
                    data_dtype = original[d].dtype
                if original[d].shape != data_shape:
                    msg = "All input detector arrays must have the same shape"
                    log.error(msg)
                    raise RuntimeError(msg)
                if original[d].dtype != data_dtype:
                    msg = "All input detector arrays must have the same dtype"
                    log.error(msg)
                    raise RuntimeError(msg)
        else:
            # If not specified, use defaults for shape and dtype.
            if data_shape is None:
                data_shape = (self.samples,)
            if data_dtype is None:
                data_dtype = np.float64

        if data_shape[0] != self.samples:
            msg = "Detector data first dimension size ({}) does not match number of local samples ({})".format(
                data_shape[0], self.samples
            )
            log.error(msg)
            raise RuntimeError(msg)

        # Create the data object
        self._internal[name] = DetectorData(detectors, data_shape, data_dtype)

        # Copy input data if given
        if original is not None:
            for d in self._internal[name].keys():
                self._internal[name][d] = original[d]

        return

    # Shortcuts for creating standard data objects

    def create_signal(self, name="signal"):
        self.create(name, shape=(self.samples,), dtype=np.float64)

    def create_flags(self, name="flags"):
        self.create(name, shape=(self.samples,), dtype=np.uint8)

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        self._internal[key].clear()
        del self._internal[key]

    def __setitem__(self, key, value):
        self._internal[key] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        for k in self._internal.keys():
            self._internal[k].clear()

    def __repr__(self):
        val = "<DetDataMgr {} local detectors, {} samples".format(
            len(self.detectors), self.samples
        )
        for k in self._internal.keys():
            val += "\n    {}: shape = {}, dtype = {}".format(
                k, self._internal[k].shape, self._internal[k].dtype
            )
        val += ">"
        return val


class SharedDataMgr(MutableMapping):
    """Class used to manage shared data objects in an Observation.

    New objects can be created with "create()" method:

        obs.shared.create(name, original=None, shape=None, dtype=None, comm=None)

    Which supports copying data from an existing MPIShared object.  The communicator
    defaults to sharing the data across the observation communicator, but other options
    would be to pass in the observation grid_comm_row or grid_comm_col communicators
    in order to share detector information across the process grid row or to share
    telescope data across the process grid column.

    There are shortcut methods to create standard data products:

        ob.detdata.create_signal()
        ob.detdata.create_flags()

    After creation, you can access a given object by name with standard dictionary
    syntax:

        obs.shared[name]

    And delete it as well:

        del obs.shared[name]

    NOTE:  These shared memory objects can be read from all processes asynchronously,
    but write access must be synchronized.  You must set the data collectively by
    using the "set" method and passing in data on one process.  Here is an example
    creating timestamps that are common to all processes in a column of the grid and
    setting those on the column rank zero process.

        n_time = obs.local_samples[1]
        col_rank = obs.grid_ranks[0]

        timestamps = None
        if col_rank == 0:
            # Input data only exists on one process
            timestamps = np.arange(n_time, dtype=np.float32)

        obs.shared.create(
            name="times",
            shape=(n_time,),
            dtype=np.float32,
            comm=obs.grid_comm_col
        )
        obs.shared["times"].set(timestamps, offset=(0,), fromrank=0)

    """

    def __init__(self, samples, detectors, comm, comm_row, comm_col):
        self.samples = samples
        self.detectors = detectors
        self.comm = comm
        self.comm_row = comm_row
        self.comm_col = comm_col
        self._internal = dict()

    def create(self, name, original=None, shape=None, dtype=None, comm=None):
        """Create a shared memory buffer.

        This buffer will be replicated across all nodes used by the processes owning
        the observation.  This uses the MPIShared class, which falls back to a simple
        numpy array if MPI is not being used.  After creating the buffer, you should
        set the elements by using the MPIShared.set() method.

        Args:
            name (str): Name of the shared memory object (e.g. "boresight").
            original (array): Construct the shared array copying this input local
                memory. This argument is only meaningful on the rank zero process of the
                specified communicator.
            shape (tuple): If not constructing from an existing array, use this shape
                for the new buffer.
            dtype (np.dtype): If not constructing from an existing array, use this
                dtype for each element.
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
        if shared_comm is None:
            # Use the observation communicator.
            shared_comm = self.comm

        copy_data = 0
        shared_shape = shape
        shared_dtype = dtype
        if original is not None:
            copy_data = 1
        if shared_comm is not None:
            copy_data = shared_comm.allreduce(copy_data, op=MPI.SUM)
        if copy_data > 1:
            msg = "If passing in an existing data buffer, it should exist on "
            "exactly one process (rank 0).  All other ranks should pass in None."
            log.error(msg)
            raise RuntimeError(msg)

        if shared_comm is None or shared_comm.rank == 0:
            if copy_data == 1:
                shared_shape = original.shape
                shared_dtype = original.dtype
            shared_dtype = np.dtype(shared_dtype)

        if shared_comm is not None:
            shared_shape = shared_comm.bcast(shared_shape, root=0)
            shared_dtype = shared_comm.bcast(shared_dtype, root=0)

        # Use defaults for shape and dtype if not set
        if shared_shape is None:
            shared_shape = (self.samples,)
        if shared_dtype is None:
            shared_dtype = np.float64

        # Create the data object
        self._internal[name] = MPIShared(shared_shape, shared_dtype, shared_comm)

        # Copy input data if given
        if copy_data == 1:
            self._internal[name].set(original, np.zeros_like(shared_shape), fromrank=0)
        return

    # Shortcuts for creating standard data objects

    def create_times(self, name="times"):
        self.create(name, shape=(self.samples,), dtype=np.float64, comm=self.comm_col)

    def create_flags(self, name="flags"):
        self.create(name, shape=(self.samples,), dtype=np.uint8, comm=self.comm_col)

    def create_velocity(self, name="velocity"):
        self.create(name, shape=(self.samples, 3), dtype=np.float64, comm=self.comm_col)

    def create_position(self, name="position"):
        self.create(name, shape=(self.samples, 3), dtype=np.float64, comm=self.comm_col)

    def create_hwp_angle(self, name="hwp_angle"):
        self.create(name, shape=(self.samples,), dtype=np.float64, comm=self.comm_col)

    def create_boresight_radec(self, name="boresight_radec"):
        self.create(name, shape=(self.samples, 4), dtype=np.float64, comm=self.comm_col)

    def create_boresight_azel(self, name="boresight_azel"):
        self.create(name, shape=(self.samples, 4), dtype=np.float64, comm=self.comm_col)

    def create_boresight_response(self, name="boresight_response"):
        self.create(
            name, shape=(self.samples, 16), dtype=np.float32, comm=self.comm_col
        )

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        self._internal[key].close()
        del self._internal[key]

    def __setitem__(self, key, value):
        self._internal[key] = value

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

    def __repr__(self):
        val = "<SharedDataMgr {} samples".format(self.samples)
        for k in self._internal.keys():
            val += "\n    {}: shape = {}, dtype = {}".format(
                k, self._internal[k].shape, self._internal[k].dtype
            )
        val += ">"
        return val


class IntervalMgr(MutableMapping):
    """Class for creating and storing interval lists in an observation.

    Named lists of intervals are accessed by dictionary style syntax ([] brackets).
    When making new interval lists, these can be added directly on each process, or
    some helper functions can be used to create the appropriate local interval lists
    given a global set of ranges.

    Args:
        comm (mpi4py.MPI.Comm):  The observation communicator.
        comm_row (mpi4py.MPI.Comm):  The process row communicator.
        comm_col (mpi4py.MPI.Comm):  The process column communicator.

    """

    def __init__(self, comm, comm_row, comm_col):
        self.comm = comm
        self.comm_col = comm_col
        self.comm_row = comm_row
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
        self._internal[name] = IntervalList(local_times, timespans=global_timespans)

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
        if self.comm is not None:
            # Find the process grid ranks of the incoming data
            if self.comm.rank == fromrank:
                send_col_rank = self.comm_col.rank
                send_row_rank = self.comm_row.rank
            send_col_rank = self.comm.bcast(send_col_rank, root=0)
            send_row_rank = self.comm.bcast(send_row_rank, root=0)
            # Broadcast data along the row
            if self.comm_col.rank == send_col_rank:
                global_times = self.comm_row.bcast(global_times, root=send_row_rank)

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
            except:
                pass
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
        val = "<IntervalMgr {} lists".format(len(self._internal))
        for k in self._internal.keys():
            val += "\n  {}: {} intervals".format(k, len(self._internal[k]))
        val += ">"
        return val


class DetDataView(MutableMapping):
    """Class that applies views to a DetDataMgr instance."""

    def __init__(self, obj, slices):
        self.obj = obj
        self.slices = slices

    # Mapping methods

    def __getitem__(self, key):
        vw = [
            np.array(
                self.obj.detdata[key][:, x],
                dtype=self.obj.detdata[key].dtype,
                copy=False,
            )
            for x in self.slices
        ]
        return vw

    def __delitem__(self, key):
        raise RuntimeError(
            "Cannot delete views of detdata, since they are created on demand"
        )

    def __setitem__(self, key, value):
        vw = [
            np.array(
                self.obj.detdata[key][:, x],
                dtype=self.obj.detdata[key].dtype,
                copy=False,
            )
            for x in self.slices
        ]
        if isinstance(value, numbers.Number) or len(value) == 1:
            # This is a numerical scalar or identical array for all slices
            for v in vw:
                v[:] = value
        else:
            # One element of value for each slice
            vw[:] = value

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def __repr__(self):
        val = "<DetDataView {} slices".format(len(self.slices))
        val += ">"
        return val


class SharedView(MutableMapping):
    """Class that applies views to a SharedDataMgr instance."""

    def __init__(self, obj, slices):
        self.obj = obj
        self.slices = slices

    # Mapping methods

    def __getitem__(self, key):
        vw = [
            np.array(
                self.obj.shared[key][x], dtype=self.obj.shared[key].dtype, copy=False
            )
            for x in self.slices
        ]
        return vw

    def __delitem__(self, key):
        raise RuntimeError(
            "Cannot delete views of shared data, since they are created on demand"
        )

    def __setitem__(self, key, value):
        raise RuntimeError(
            "Cannot set views of shared data- use the set() method on the original."
        )

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def __repr__(self):
        val = "<SharedView {} slices".format(len(self.slices))
        val += ">"
        return val


class View(Sequence):
    """Class representing a list of views into any of the local observation data."""

    def __init__(self, obj, key):
        self.obj = obj
        self.key = key
        # Compute a list of slices for these intervals
        self.slices = [slice(x.first, x.last + 1, 1) for x in self.obj.intervals[key]]
        self.detdata = DetDataView(obj, self.slices)
        self.shared = SharedView(obj, self.slices)

    def __getitem__(self, key):
        return self.slices[key]

    def __contains__(self, item):
        for sl in self.slices:
            if sl == item:
                return True
        return False

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def __repr__(self):
        s = "["
        if len(self.slices) > 1:
            for it in self.slices[0:-1]:
                s += str(it)
                s += ", "
        if len(self.slices) > 0:
            s += str(self.slices[-1])
        s += "]"
        return s


class ViewMgr(MutableMapping):
    """Class to manage views into observation data objects."""

    def __init__(self, obj):
        self.obj = obj
        if not hasattr(obj, "_views"):
            self.obj._views = dict()

    # Mapping methods

    def __getitem__(self, key):
        if key not in self.obj._views:
            # View does not yet exist, create it.
            if key not in self.obj.intervals:
                raise KeyError(
                    "Observation does not have interval list named '{}'".format(key)
                )
            self.obj._views[key] = View(self.obj, key)
            # Register deleter callback
            self.obj.intervals.register_delete_callback(key, self.__delitem__)
        return self.obj._views[key]

    def __delitem__(self, key):
        del self.obj._views[key]

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot set views directly- simply access them.")

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def clear(self):
        self.obj._views.clear()


class ViewInterface(object):
    """Descriptor class for accessing the views in an observation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        else:
            if not hasattr(obj, "_viewmgr"):
                obj._viewmgr = ViewMgr(obj)
                print("ViewInterface __get__ created ", obj._viewmgr)
            return obj._viewmgr

    def __set__(self, obj, value):
        raise AttributeError("Cannot reset the view interface")

    def __delete__(self, obj):
        raise AttributeError("Cannot delete the view interface")


class DistDetSamp(object):
    """Class used within an Observation to store the detector and sample distribution.

    This is just a simple container for various properties of the distribution.

    Args:

    """

    def __init__(
        self, samples, detectors, sample_sets, detector_sets, comm, process_rows
    ):
        log = Logger.get()

        self.detectors = detectors
        self.samples = samples
        self.sample_sets = sample_sets
        self.detector_sets = detector_sets
        self.process_rows = process_rows

        if self.samples is None or self.samples <= 0:
            msg = "You must specify the number of samples as a positive integer"
            log.error(msg)
            raise RuntimeError(msg)

        self.comm = comm
        self.comm_size = 1
        self.comm_rank = 0
        if self.comm is not None:
            self.comm_size = self.comm.size
            self.comm_rank = self.comm.rank

        if self.process_rows is None:
            if self.comm is None:
                # No MPI, default to 1
                self.process_rows = 1
            else:
                # We have MPI, default to the size of the communicator
                self.process_rows = self.comm.size

        self.process_cols = 1
        self.comm_row_size = 1
        self.comm_row_rank = 0
        self.comm_col_size = 1
        self.comm_col_rank = 0
        self.comm_row = None
        self.comm_col = None

        if self.comm is None:
            if self.process_rows != 1:
                msg = "MPI is disabled, so process_rows must equal 1"
                log.error(msg)
                raise RuntimeError(msg)
        else:
            if comm.size % self.process_rows != 0:
                msg = "The number of process_rows ({}) does not divide evenly into the communicator size ({})".format(
                    self.process_rows, comm.size
                )
                log.error(msg)
                raise RuntimeError(msg)
            self.process_cols = comm.size // self.process_rows
            self.comm_col_rank = comm.rank // self.process_cols
            self.comm_row_rank = comm.rank % self.process_cols

            # Split the main communicator into process row and column
            # communicators.

            if self.process_cols == 1:
                self.comm_row = MPI.COMM_SELF
            else:
                self.comm_row = self.comm.Split(self.comm_col_rank, self.comm_row_rank)
                self.comm_row_size = self.comm_row.size

            if self.process_rows == 1:
                self.comm_col = MPI.COMM_SELF
            else:
                self.comm_col = self.comm.Split(self.comm_row_rank, self.comm_col_rank)
                self.comm_col_size = self.comm_col.size

        # If detector_sets is specified, check consistency.

        if self.detector_sets is not None:
            test = 0
            for ds in self.detector_sets:
                test += len(ds)
                for d in ds:
                    if d not in self.detectors:
                        msg = (
                            "Detector {} in detector_sets but not in detectors".format(
                                d
                            )
                        )
                        log.error(msg)
                        raise RuntimeError(msg)
            if test != len(detectors):
                msg = "{} detectors given, but detector_sets has {}".format(
                    len(detectors), test
                )
                log.error(msg)
                raise RuntimeError(msg)

        # If sample_sets is specified, it must be consistent with
        # the total number of samples.

        if self.sample_sets is not None:
            test = 0
            for st in self.sample_sets:
                test += np.sum(st)
            if samples != test:
                msg = (
                    "Sum of sample_sizes ({}) does not equal total samples ({})".format(
                        test, samples
                    )
                )
                log.error(msg)
                raise RuntimeError(msg)

        (self.dets, self.det_sets, self.samps, self.samp_sets) = distribute_samples(
            self.comm,
            self.detectors,
            self.samples,
            detranks=self.process_rows,
            detsets=self.detector_sets,
            sampsets=self.sample_sets,
        )


class Observation(MutableMapping):
    """Class representing the data for one observation.

    An Observation stores information about data distribution across one or more MPI
    processes and is a container for three types of objects:

        * Local detector data (unique to each process).
        * Shared data that has one common copy for every node spanned by the
          observation.
        * Other arbitrary small metadata.

    Small metadata can be store directly in the Observation using normal square
    bracket "[]" access to elements (an Observation is a dictionary).  Groups of
    detector data (e.g. "signal", "flags", etc) can be accessed in the separate
    detector data dictionary (the "d" attribute).  Shared data can be similarly stored
    in the "shared" attribute.

    The detector data within an Observation is distributed among the processes in an
    MPI communicator.  The processes in the communicator are arranged in a rectangular
    grid, with each process storing some number of detectors for a piece of time
    covered by the observation.  The most common configuration (and the default) is to
    make this grid the size of the communicator in the "detector direction" and a size
    of one in the "sample direction":

        MPI           det1  sample(0), sample(1), sample(2), ...., sample(N-1)
        rank 0        det2  sample(0), sample(1), sample(2), ...., sample(N-1)
        --------------------------------------------------------------------------
        MPI           det3  sample(0), sample(1), sample(2), ...., sample(N-1)
        rank 1        det4  sample(0), sample(1), sample(2), ...., sample(N-1)

    So each process has a subset of detectors for the whole span of the observation
    time.  You can override this shape by setting the process_rows to something
    else.  For example, process_rows=1 would result in:

                  MPI rank 0              |          MPI rank 1
                                          |
        det1  sample(0), sample(1), ...,  |  ...., sample(N-1)
        det2  sample(0), sample(1), ...,  |  ...., sample(N-1)
        det3  sample(0), sample(1), ...,  |  ...., sample(N-1)
        det4  sample(0), sample(1), ...,  |  ...., sample(N-1)

    Args:
        telescope (Telescope):  An instance of a Telescope object.
        samples (int):  The total number of samples for this observation.
        name (str):  (Optional) The observation name.
        UID (int):  (Optional) The Unique ID for this observation.  If not specified,
            the UID will be computed from a hash of the name.
        comm (mpi4py.MPI.Comm):  (Optional) The MPI communicator to use.
        detector_sets (list):  (Optional) List of lists containing detector names.
            These discrete detector sets are used to distribute detectors- a detector
            set will always be within a single row of the process grid.  If None,
            every detector is a set of one.
        sample_sets (list):  (Optional) List of lists of chunk sizes (integer numbers of
            samples).  These discrete sample sets are used to distribute sample data.
            A sample set will always be within a single column of the process grid.  If
            None, any distribution break in the sample direction will happen at an
            arbitrary place.  The sum of all chunks must equal the total number of
            samples.
        process_rows (int):  (Optional) The size of the rectangular process grid
            in the detector direction.  This number must evenly divide into the size of
            mpicomm.  If not specified, defaults to the size of the communicator.

    """

    view = ViewInterface()

    def __init__(
        self,
        telescope,
        samples,
        name=None,
        UID=None,
        comm=None,
        detector_sets=None,
        sample_sets=None,
        process_rows=None,
    ):
        log = Logger.get()
        self._telescope = telescope
        self._samples = samples
        self._name = name
        self._UID = UID
        self._comm = comm
        self._detector_sets = detector_sets
        self._sample_sets = sample_sets

        if self._UID is None and self._name is not None:
            self._UID = name_UID(self._name)

        self.dist = DistDetSamp(
            self._samples,
            self._telescope.focalplane.detectors,
            self._sample_sets,
            self._detector_sets,
            self._comm,
            process_rows,
        )

        if self.dist.comm_rank == 0:
            # check that all processes have some data, otherwise print warning
            for d in range(self.dist.process_rows):
                if len(self.dist.dets[d]) == 0:
                    msg = "WARNING: process row rank {} has no detectors"
                    " assigned in observation.".format(d)
                    log.warning(msg)
            for r in range(self.dist.process_cols):
                if self.dist.samps[r][1] <= 0:
                    msg = "WARNING: process column rank {} has no data assigned "
                    "in observation.".format(r)
                    log.warning(msg)

        # The internal metadata dictionary
        self._internal = dict()

        # Set up the data managers
        self.detdata = DetDataMgr(self._samples, self.detectors)

        self.shared = SharedDataMgr(
            self._samples,
            self.detectors,
            self._comm,
            self.dist.comm_row,
            self.dist.comm_col,
        )

        self.intervals = IntervalMgr(self._comm, self.dist.comm_row, self.dist.comm_col)

    # General properties

    @property
    def telescope(self):
        """
        (Telescope):  The Telescope instance for this observation.
        """
        return self._telescope

    @property
    def name(self):
        """
        (str):  The name of the observation.
        """
        return self._name

    @property
    def UID(self):
        """
        (int):  The Unique ID for this observation.
        """
        return self._UID

    # The overall MPI communicator for this observation.

    @property
    def comm(self):
        """
        (mpi4py.MPI.Comm):  The group communicator for this observation (or None).
        """
        return self.dist.comm

    @property
    def comm_size(self):
        """
        (int): The number of processes in the observation communicator.
        """
        return self.dist.comm_size

    @property
    def comm_rank(self):
        """
        (int): The rank of this process in the observation communicator.
        """
        return self.dist.comm_rank

    # The MPI communicator along the current row of the process grid

    @property
    def comm_row(self):
        """
        (mpi4py.MPI.Comm):  The communicator for processes in the same row (or None).
        """
        return self.dist.comm_row

    @property
    def comm_row_size(self):
        """
        (int): The number of processes in the row communicator.
        """
        return self.dist.comm_row_size

    @property
    def comm_row_rank(self):
        """
        (int): The rank of this process in the row communicator.
        """
        return self.dist.comm_row_rank

    # The MPI communicator along the current column of the process grid

    @property
    def comm_col(self):
        """
        (mpi4py.MPI.Comm):  The communicator for processes in the same column (or None).
        """
        return self.dist.comm_col

    @property
    def comm_col_size(self):
        """
        (int): The number of processes in the column communicator.
        """
        return self.dist.comm_col_size

    @property
    def comm_col_rank(self):
        """
        (int): The rank of this process in the column communicator.
        """
        return self.dist.comm_col_rank

    # Detector distribution

    @property
    def detectors(self):
        """
        (list): All detectors.  Convenience wrapper for telescope.focalplane.detectors
        """
        return self._telescope.focalplane.detectors

    @property
    def local_detectors(self):
        """
        (list): The detectors assigned to this process.
        """
        return self.dist.dets[self.dist.comm_col_rank]

    def select_local_detectors(self, selection=None):
        """
        (list): The detectors assigned to this process, optionally pruned.
        """
        if selection is None:
            return self.local_detectors
        else:
            dets = list()
            for det in self.local_detectors:
                if det in selection:
                    dets.append(det)
            return dets

    # Detector set distribution

    @property
    def detector_sets(self):
        """
        (list):  The total list of detector sets for this observation.
        """
        return self._detector_sets

    @property
    def local_detector_sets(self):
        """
        (list):  The detector sets assigned to this process (or None).
        """
        if self._detector_sets is None:
            return None
        else:
            ds = list()
            for d in range(self.dist.det_sets[self.dist.comm_col_rank][1]):
                off = self.dist.det_sets[self.dist.comm_col_rank][0]
                ds.append(self._detector_sets[off + d])
            return ds

    # Sample distribution

    @property
    def n_sample(self):
        """(int): the total number of samples in this observation."""
        return self._samples

    @property
    def local_samples(self):
        """
        (tuple): The first element of the tuple is the first observation sample
            assigned to this process.  The second element of the tuple is the number of
            samples assigned to this process.
        """
        return self.dist.samps[self.dist.comm_row_rank]

    @property
    def offset(self):
        """
        The first sample on this process, relative to the observation start.
        """
        return self.local_samples[0]

    @property
    def n_local(self):
        """
        The number of local samples on this process.
        """
        return self.local_samples[1]

    # Sample set distribution

    @property
    def sample_sets(self):
        """
        (list):  The input full list of sample sets used in data distribution
        """
        return self._sample_sets

    @property
    def local_sample_sets(self):
        """
        (list):  The sample sets assigned to this process (or None).
        """
        if self._sample_sets is None:
            return None
        else:
            ss = list()
            for s in range(self.dist.samp_sets[self.dist.comm_row_rank][1]):
                off = self.dist.samp_sets[self.dist.comm_row_rank][0]
                ss.append(self._sample_sets[off + d])
            return ss

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        del self._internal[key]

    def __setitem__(self, key, value):
        self._internal[key] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def __del__(self):
        if hasattr(self, "d"):
            self.detdata.clear()
        if hasattr(self, "shared"):
            self.shared.clear()

    def __repr__(self):
        val = "<Observation"
        val += "\n  name = '{}'".format(self.name)
        val += "\n  UID = '{}'".format(self.UID)
        if self._comm is None:
            val += "  group has a single process (no MPI)"
        else:
            val += "  group has {} processes".format(self._comm.size)
        val += "\n  telescope = {}".format(self._telescope.__repr__())
        for k, v in self._internal.items():
            val += "\n  {} = {}".format(k, v)
        val += "\n  {} samples".format(self._samples)
        val += "\n  shared:  {}".format(self.shared)
        val += "\n  detdata:  {}".format(self.detdata)
        val += "\n>"
        return val

    # Redistribution

    def redistribute(self, process_rows):
        """Take the currently allocated observation and redistribute in place.

        This changes the data distribution within the observation.  After
        re-assigning all detectors and samples, the currently allocated shared data
        objects and detector data objects are redistributed using the observation
        communicator.

        Args:
            process_rows (int):  The size of the new process grid in the detector
                direction.  This number must evenly divide into the size of the
                observation communicator.

        Returns:
            None

        """
        if process_rows == self.dist.process_rows:
            # Nothing to do!
            return
        pass

    # Accelerator use

    # @property
    # def accelerator(self):
    #     """Return dictionary of objects mirrored to the accelerator.
    #     """
    #     return None
    #
    # def to_accelerator(self, keys, detectors=None):
    #     """Copy data objects to the accelerator.
    #
    #     Keys may be standard key names ("SIGNAL", "FLAGS", etc) or arbitrary keys.
    #     In the case of standard keys, any internal overrides specified at construction
    #     are applied.
    #
    #     Args:
    #         keys (iterable): the objects to stage to accelerator memory.  These must
    #             be scalars or arrays of C-compatible types.
    #         detectors (list): Copy only the selected detectors to the accelerator.
    #
    #     Returns:
    #         None
    #
    #     """
    #     log = Logger.get()
    #
    #     # Clear the dictionary of accelerator objects
    #
    #     if have_pycuda:
    #         # Using NVIDIA GPUs
    #         # Compute the set of data that needs to be copied to each GPU.
    #         pass
    #     else:
    #         msg = "No supported accelerator found"
    #         log.warning(msg)
    #     return
    #
    # def from_accelerator(self, keys, detectors=None):
    #     """Copy data objects from the accelerator.
    #
    #     Keys may be standard key names ("SIGNAL", "FLAGS", etc) or arbitrary keys.
    #     In the case of standard keys, any internal overrides specified at construction
    #     are applied.
    #
    #     Args:
    #         keys (iterable): the objects to copy from accelerator memory.  These must
    #             be scalars or arrays of C-compatible types.
    #         detectors (list): Copy only the selected detectors to the accelerator.
    #
    #     Returns:
    #         None
    #
    #     """
    #     log = Logger.get()
    #
    #     if have_pycuda:
    #         # Using NVIDIA GPUs
    #         # Find the superset of all data that needs to move from each GPU
    #         # Copy data
    #         # Free GPU memory
    #         pass
    #     else:
    #         msg = "No supported accelerator found"
    #         log.warning(msg)
    #         return
    #
    #     return
