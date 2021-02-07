# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys

import numbers

from collections.abc import MutableMapping, Sequence, Mapping

import numpy as np

from .mpi import MPI

from .instrument import Telescope

from .dist import distribute_samples

from .utils import (
    Logger,
    name_UID,
)

from .cuda import use_pycuda

from .observation_data import DetectorData, DetDataMgr, SharedDataMgr, IntervalMgr

from .observation_view import DetDataView, SharedView, View, ViewMgr, ViewInterface


class DistDetSamp(object):
    """Class used within an Observation to store the detector and sample distribution.

    This is just a simple container for various properties of the distribution.

    Args:
        samples (int):  The total number of samples.
        detectors (list):  The list of detector names.
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
        comm (mpi4py.MPI.Comm):  (Optional) The MPI communicator to use.
        process_rows (int):  (Optional) The size of the rectangular process grid
            in the detector direction.  This number must evenly divide into the size of
            comm.  If not specified, defaults to the size of the communicator.

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
        self._samples = n_samples
        self._name = name
        self._uid = uid
        self._comm = comm
        self._detector_sets = detector_sets
        self._sample_sets = sample_sets

        if self._uid is None and self._name is not None:
            self._uid = name_UID(self._name)

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
        self.detdata = DetDataMgr(self.local_detectors, self.n_local_samples)

        self.shared = SharedDataMgr(
            self._comm,
            self.dist.comm_row,
            self.dist.comm_col,
        )

        self.intervals = IntervalMgr(self._comm, self.dist.comm_row, self.dist.comm_col)

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
    def all_detector_sets(self):
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
    def n_all_samples(self):
        """(int): the total number of samples in this observation."""
        return self._samples

    @property
    def local_index_offset(self):
        """
        The first sample on this process, relative to the observation start.
        """
        return self.dist.samps[self.dist.comm_row_rank][0]

    @property
    def n_local_samples(self):
        """
        The number of local samples on this process.
        """
        return self.dist.samps[self.dist.comm_row_rank][1]

    # Sample set distribution

    @property
    def all_sample_sets(self):
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
        if hasattr(self, "detdata"):
            self.detdata.clear()
        if hasattr(self, "shared"):
            self.shared.clear()

    def __repr__(self):
        val = "<Observation"
        val += "\n  name = '{}'".format(self.name)
        val += "\n  uid = '{}'".format(self.uid)
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
        val += "\n  intervals:  {}".format(self.intervals)
        val += "\n>"
        return val

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

        # Construct the new distribution
        newdist = DistDetSamp(
            self._samples,
            self._telescope.focalplane.detectors,
            self._sample_sets,
            self._detector_sets,
            self._comm,
            process_rows,
        )

        det_order = {y: x for x, y in enumerate(self.all_detectors)}

        if newdist.comm_rank == 0:
            # check that all processes have some data, otherwise print warning
            for d in range(newdist.process_rows):
                if len(newdist.dets[d]) == 0:
                    msg = "WARNING: process row rank {} has no detectors"
                    " assigned in new distribution.".format(d)
                    log.warning(msg)
            for r in range(newdist.process_cols):
                if newdist.samps[r][1] <= 0:
                    msg = "WARNING: process column rank {} has no data assigned "
                    "in new distribution.".format(r)
                    log.warning(msg)

        # Clear all views, since they depend on the intervals we are about to
        # delete.  Views are re-created on demand.

        print("clearing views", flush=True)
        self.view.clear()

        # After an IntervalList is added, only the local intervals on each process are
        # kept.  We need to construct the original timespans that were used and save,
        # so that we can rebuild them after the timestamps have been redistributed.

        if times is None:
            if self.comm_rank == 0:
                msg = "Time stamps not specified when redistributing observation {}."
                msg += "  Intervals will not be redistributed."
                log.warning(msg)

        global_intervals = None
        if self.comm_rank == 0:
            global_intervals = dict()

        for iname in list(self.intervals.keys()):
            print("gathering interval {}".format(iname), flush=True)
            ilist = self.intervals[iname]
            all_ilist = None
            if self.comm_row is None:
                all_ilist = [(ilist, self.n_local_samples)]
            else:
                # Gather across the process row
                if self.comm_col_rank == 0:
                    all_ilist = self.comm_row.gather(
                        (ilist, self.n_local_samples), root=0
                    )
            del ilist
            if self.comm_rank == 0:
                # Only one process builds the list of start / stop times.
                glist = list()
                last_continue = False
                last_start = 0
                last_stop = 0
                for pdata, pn in all_ilist:
                    if len(pdata) == 0:
                        continue

                    for intvl in pdata:
                        if last_continue:
                            if invl.first == 0:
                                last_stop = intvl.stop
                            else:
                                glist.append((last_start, last_stop))
                                last_continue = False
                                last_start = intvl.start
                                last_stop = intvl.stop
                        else:
                            last_start = intvl.start
                            last_stop = intvl.stop

                        if intvl.last == pn - 1:
                            last_continue = True
                        else:
                            glist.append((last_start, last_stop))
                            last_continue = False
                if last_continue:
                    # add final range
                    glist.append((last_start, last_stop))
            global_intervals[iname] = glist
            print("purging interval {}".format(iname), flush=True)
            del self.intervals[iname]

        def compute_1d(old_off, old_n, pdist):
            """Helper function to compute slices along one dimension."""
            pnew = list()
            for ip, p in enumerate(pdist):
                poff = p[0]
                pend = p[0] + p[1]
                if poff >= old_off + old_n:
                    continue
                if pend <= old_off:
                    continue
                new_off = old_off
                if poff > old_off:
                    new_off = poff
                new_n = old_off + old_n - new_off
                if pend < old_off + old_n:
                    new_n = pend - new_off
                pnew.append((ip, new_off, new_n))
            return pnew

        # Redistribute detector data

        newdetdata = DetDataMgr(
            newdist.dets[newdist.comm_col_rank],
            newdist.samps[newdist.comm_row_rank][1],
        )

        newdets = list()
        for pdet in newdist.dets:
            dfirst = det_order[pdet[0]]
            dlast = det_order[pdet[-1]]
            newdets.append(dfirst, dlast - dfirst + 1)

        olddets = list()
        for pdet in self.dist.dets:
            dfirst = det_order[pdet[0]]
            dlast = det_order[pdet[-1]]
            olddets.append(dfirst, dlast - dfirst + 1)

        # Every process figures out its send and receive information

        send_row = compute_1d(
            self.local_index_offset, self.n_local_samples, newdist.samps
        )
        det_first = det_order[self.local_detectors[0]]
        det_last = det_order[self.local_detectors[-1]]
        send_col = compute_1d(det_first, det_last - det_first + 1, newdets)
        send_info = list()
        for sr in send_row:
            for sc in send_col:
                psend = sr[0] * newdist.comm_col_size + sc[0]
                send_info.append((psend, sr[1], sr[2], sc[1], sc[2]))

        msg = "Proc {} det send:  {}".format(self.comm_rank, send_info)
        print(msg, flush=True)

        recv_row = compute_1d(
            newdist.samps[newdist.comm_row_rank][0],
            newdist.samps[newdist.comm_row_rank][1],
            self.dist.samps,
        )
        det_first = det_order[newdist.dets[newdist.comm_col_rank][0]]
        det_last = det_order[newdist.dets[newdist.comm_col_rank][-1]]
        recv_col = compute_1d(det_first, det_last - det_first + 1, olddets)
        recv_info = list()
        for rr in recv_row:
            for rc in recv_col:
                precv = rr[0] * self.dist.comm_col_size + rc[0]
                recv_info.append((precv, rr[1], rr[2], rc[1], rc[2]))

        msg = "Proc {} det recv:  {}".format(self.comm_rank, recv_info)
        print(msg, flush=True)

        # for field in list(self.detdata.keys()):
        #
        #     self._dist.comm.Alltoallv(
        #         [self.raw, self._send_counts, self._send_displ, self.mpitype],
        #         [self.receive, self._recv_counts, self._recv_displ, self.mpitype],
        #     )
        #
        #     del self.datadata[field]
        #
        # # Redistribute shared data
        #
        # newshared = SharedDataMgr(
        #     self._comm,
        #     newdist.comm_row,
        #     newdist.comm_col,
        # )
        #
        # # Redistribute intervals
        #
        # newintervals = IntervalMgr(self._comm, newdist.comm_row, newdist.comm_col)

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
