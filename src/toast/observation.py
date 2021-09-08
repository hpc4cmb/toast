# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys

import types

import copy

import numbers

from collections.abc import MutableMapping, Sequence, Mapping

import numpy as np

from pshmem.utils import mpi_data_type

from .mpi import MPI, comm_equal

from .instrument import Telescope

from .dist import distribute_samples

from .intervals import IntervalList

from .utils import (
    Logger,
    name_UID,
)

from .timing import function_timer

from .cuda import use_pycuda

from .observation_data import (
    DetectorData,
    DetDataManager,
    SharedDataManager,
    IntervalsManager,
)

from .observation_view import DetDataView, SharedView, View, ViewManager, ViewInterface

from .observation_dist import (
    DistDetSamp,
    redistribute_data,
)


default_names = None


def set_default_names(names=None):
    """Update default names for common Observation objects.

    Args:
        names (dict):  The dictionary specifying any name overrides.

    Returns:
        None

    """
    global default_names

    default_values = {
        "times": "times",
        "shared_flags": "flags",
        "det_data": "signal",
        "det_flags": "flags",
        "hwp_angle": "hwp_angle",
        "azimuth": "azimuth",
        "elevation": "elevation",
        "boresight_azel": "boresight_azel",
        "boresight_radec": "boresight_radec",
        "position": "position",
        "velocity": "velocity",
        "pixels": "pixels",
        "weights": "weights",
        "quats": "quats",
        "quats_azel": "quats_azel",
    }

    defaults = dict()
    defaults.update(default_values)
    if names is not None:
        defaults.update(names)
    default_names = types.SimpleNamespace(**defaults)


if default_names is None:
    set_default_names()


class Observation(MutableMapping):
    """Class representing the data for one observation.

    An Observation stores information about data distribution across one or more MPI
    processes and is a container for four types of objects:

        * Local detector data (unique to each process).
        * Shared data that has one common copy for every node spanned by the
          observation.
        * Intervals defining spans of data with some common characteristic.
        * Other arbitrary small metadata.

    Small metadata can be stored directly in the Observation using normal square
    bracket "[]" access to elements (an Observation is a dictionary).  Groups of
    detector data (e.g. "signal", "flags", etc) can be accessed in the separate
    detector data dictionary (the "detdata" attribute).  Shared data can be similarly
    stored in the "shared" attribute.  Lists of intervals are accessed in the
    "intervals" attribute and data views can use any interval list to access subsets
    of detector and shared data.

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
        n_samples (int):  The total number of samples for this observation.
        name (str):  (Optional) The observation name.
        uid (int):  (Optional) The Unique ID for this observation.  If not specified,
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
            comm.  If not specified, defaults to the size of the communicator.

    """

    view = ViewInterface()

    def __init__(
        self,
        telescope,
        n_samples,
        name=None,
        uid=None,
        comm=None,
        detector_sets=None,
        sample_sets=None,
        process_rows=None,
    ):
        log = Logger.get()
        self._telescope = telescope
        self._name = name
        self._uid = uid

        if self._uid is None and self._name is not None:
            self._uid = name_UID(self._name)

        self.dist = DistDetSamp(
            n_samples,
            self._telescope.focalplane.detectors,
            sample_sets,
            detector_sets,
            comm,
            process_rows,
        )

        # The internal metadata dictionary
        self._internal = dict()

        # Set up the data managers
        self.detdata = DetDataManager(self.dist)
        self.shared = SharedDataManager(self.dist)
        self.intervals = IntervalsManager(self.dist)

    # Fully clear the observation

    def clear(self):
        self.view.clear()
        self.intervals.clear()
        self.detdata.clear()
        self.shared.clear()
        self._internal.clear()

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
    def uid(self):
        """
        (int):  The Unique ID for this observation.
        """
        return self._uid

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
    def all_detectors(self):
        """
        (list): All detectors stored in this observation.
        """
        return self.dist.detectors

    @property
    def local_detectors(self):
        """
        (list): The detectors assigned to this process.
        """
        return self.dist.dets[self.dist.comm_rank]

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
    def all_detector_sets(self):
        """
        (list):  The total list of detector sets for this observation.
        """
        return self.dist.detector_sets

    @property
    def local_detector_sets(self):
        """
        (list):  The detector sets assigned to this process (or None).
        """
        if self.dist.detector_sets is None:
            return None
        else:
            ds = list()
            for d in range(self.dist.det_sets[self.dist.comm_rank].n_elem):
                off = self.dist.det_sets[self.dist.comm_rank].offset
                ds.append(self.dist.detector_sets[off + d])
            return ds

    # Sample distribution

    @property
    def n_all_samples(self):
        """(int): the total number of samples in this observation."""
        return self.dist.samples

    @property
    def local_index_offset(self):
        """
        The first sample on this process, relative to the observation start.
        """
        return self.dist.samps[self.dist.comm_rank].offset

    @property
    def n_local_samples(self):
        """
        The number of local samples on this process.
        """
        return self.dist.samps[self.dist.comm_rank].n_elem

    # Sample set distribution

    @property
    def all_sample_sets(self):
        """
        (list):  The input full list of sample sets used in data distribution
        """
        return self.dist.sample_sets

    @property
    def local_sample_sets(self):
        """
        (list):  The sample sets assigned to this process (or None).
        """
        if self.dist.sample_sets is None:
            return None
        else:
            ss = list()
            for s in range(self.dist.samp_sets[self.dist.comm_rank].n_elem):
                off = self.dist.samp_sets[self.dist.comm_rank].offset
                ss.append(self.dist.sample_sets[off + s])
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
        if hasattr(self, "detdata"):
            self.detdata.clear()
        if hasattr(self, "shared"):
            self.shared.clear()

    def __repr__(self):
        val = "<Observation"
        val += f"\n  name = '{self.name}'"
        val += f"\n  uid = '{self.uid}'"
        if self.comm is None:
            val += "  group has a single process (no MPI)"
        else:
            val += f"  group has {self.comm.size} processes"
        val += f"\n  telescope = {self._telescope.__repr__()}"
        for k, v in self._internal.items():
            val += f"\n  {k} = {v}"
        val += f"\n  {self.n_all_samples} total samples ({self.n_local_samples} local)"
        val += f"\n  shared:  {self.shared}"
        val += f"\n  detdata:  {self.detdata}"
        val += f"\n  intervals:  {self.intervals}"
        val += "\n>"
        return val

    def __eq__(self, other):
        # Note that testing for equality is quite expensive, since it means testing all
        # metadata and also all detector, shared, and interval data.  This is mainly
        # used for unit tests.
        log = Logger.get()
        fail = 0
        if self.name != other.name:
            fail = 1
            log.verbose(f"Obs names {self.name} != {other.name}")
        if self.uid != other.uid:
            fail = 1
            log.verbose(f"Obs uid {self.uid} != {other.uid}")
        if self.telescope != other.telescope:
            fail = 1
            log.verbose("Obs telescopes not equal")
        if self.dist != other.dist:
            fail = 1
            log.verbose("Obs distributions not equal")
        if self._internal.keys() != other._internal.keys():
            fail = 1
            log.verbose("Obs metadata keys not equal")
        for k, v in self._internal.items():
            if v != other._internal[k]:
                feq = True
                try:
                    feq = np.allclose(v, other._internal[k])
                except Exception:
                    # Not floating point data
                    feq = False
                if not feq:
                    fail = 1
                    log.verbose(f"Obs metadata[{k}]:  {v} != {other[k]}")
                    break
        if self.shared != other.shared:
            fail = 1
            log.verbose("Obs shared data not equal")
        if self.detdata != other.detdata:
            fail = 1
            log.verbose("Obs detdata not equal")
        if self.intervals != other.intervals:
            fail = 1
            log.verbose("Obs intervals not equal")
        if self.comm is not None:
            fail = self.comm.allreduce(fail, op=MPI.SUM)
        return fail == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def duplicate(self, times=None):
        """Return a copy of the observation and all its data.

        The times field should be the name of the shared field containing timestamps.
        This is used when copying interval lists to the new observation so that these
        objects reference the timestamps within this observation (rather than the old
        one).  If this is not specified and some intervals exist, then an exception is
        raised.

        Args:
            times (str):  The name of the timestamps shared field.

        Returns:
            (Observation):  The new copy of the observation.

        """
        log = Logger.get()
        if times is None and len(self.intervals) > 0:
            msg = "You must specify the times field when duplicating observations "
            msg += "that have some intervals defined."
            log.error(msg)
            raise RuntimeError(msg)
        new_obs = Observation(
            self.telescope,
            self.n_all_samples,
            name=self.name,
            uid=self.uid,
            comm=self.dist.comm,
            detector_sets=self.all_detector_sets,
            sample_sets=self.all_sample_sets,
            process_rows=self.dist.process_rows,
        )
        for k, v in self._internal.items():
            new_obs[k] = copy.deepcopy(v)
        for name, data in self.detdata.items():
            new_obs.detdata[name] = data
        for name, data in self.shared.items():
            # Create the object on the corresponding communicator in the new obs
            new_comm = None
            if comm_equal(data.comm, self.dist.comm_row):
                # Row comm
                new_comm = new_obs.dist.comm_row
            elif comm_equal(data.comm, self.dist.comm_col):
                # Col comm
                new_comm = new_obs.dist.comm_col
            else:
                # Full obs comm
                new_comm = new_obs.dist.comm
            new_obs.shared.create(name, data.shape, dtype=data.dtype, comm=new_comm)
            offset = None
            dval = None
            if new_comm is None or new_comm.rank == 0:
                offset = tuple([0 for x in data.shape])
                dval = data.data
            new_obs.shared[name].set(dval, offset=offset, fromrank=0)
        for name, data in self.intervals.items():
            timespans = [(x.start, x.stop) for x in data]
            new_obs.intervals[name] = IntervalList(
                new_obs.shared[times], timespans=timespans
            )
        return new_obs

    def memory_use(self):
        """Estimate the memory used by shared and detector data.

        This sums the memory used by the shared and detdata attributes and returns the
        total on all processes.  This function is blocking on the observation
        communicator.

        Returns:
            (int):  The number of bytes of memory used by timestream data.

        """
        # Get local memory from detector data
        local_mem = self.detdata.memory_use()

        # If there are many intervals, this could take up non-trivial space.  Add them
        # to the local total
        for iname, it in self.intervals.items():
            if len(it) > 0:
                local_mem += len(it) * (
                    sys.getsizeof(it[0]._start)
                    + sys.getsizeof(it[0]._stop)
                    + sys.getsizeof(it[0]._first)
                    + sys.getsizeof(it[0]._last)
                )

        # Sum the aggregate local memory
        total = None
        if self.comm is None:
            total = local_mem
        else:
            total = self.comm.allreduce(local_mem, op=MPI.SUM)

        # The total shared memory use is already returned on every process by this
        # next function.
        total += self.shared.memory_use()
        return total

    # Redistribution

    @function_timer
    def redistribute(self, process_rows, times=None):
        """Take the currently allocated observation and redistribute in place.

        This changes the data distribution within the observation.  After
        re-assigning all detectors and samples, the currently allocated shared data
        objects and detector data objects are redistributed using the observation
        communicator.

        Args:
            process_rows (int):  The size of the new process grid in the detector
                direction.  This number must evenly divide into the size of the
                observation communicator.
            times (str):  The shared data field representing the timestamps.  This
                is used to recompute the intervals after redistribution.

        Returns:
            None

        """
        log = Logger.get()
        if process_rows == self.dist.process_rows:
            # Nothing to do!
            return

        # Create the new distribution
        new_dist = DistDetSamp(
            self.dist.samples,
            self._telescope.focalplane.detectors,
            self.dist.sample_sets,
            self.dist.detector_sets,
            self.dist.comm,
            process_rows,
        )

        # Do the actual redistribution
        new_shr_manager, new_det_manager, new_intervals_manager = redistribute_data(
            self.dist, new_dist, self.shared, self.detdata, self.intervals, times=times
        )

        # Replace our distribution and data managers with the new ones.
        self.dist.close()
        del self.dist
        self.dist = new_dist

        self.shared.clear()
        del self.shared
        self.shared = new_shr_manager

        self.detdata.clear()
        del self.detdata
        self.detdata = new_det_manager

        self.intervals.clear()
        del self.intervals
        self.intervals = new_intervals_manager

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
