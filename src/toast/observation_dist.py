# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys

import numbers

from collections.abc import MutableMapping, Sequence, Mapping

import numpy as np

from pshmem.utils import mpi_data_type

from .mpi import MPI

from .dist import distribute_samples

from .utils import (
    Logger,
    name_UID,
)

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


def compute_1d_offsets(off, n, new_dist):
    """Helper function to compute slices along one dimension.

    Args:
        off (int):  The local offset.
        n (int):  The local number of elements.
        new_dist (list):  A list of tuples, one per process, with the local offset and
            number of elements for that process.

    Returns:
        (list):  A list of tuples (one per process) containing information about which
            local samples map to each target process in new_dist.  Each tuple contains:
            (target process, local_offset, global_offset, num_elements).

    """
    pnew = list()
    for ip, p in enumerate(new_dist):
        poff = p[0]
        pend = p[0] + p[1]
        if poff >= off + n:
            continue
        if pend <= off:
            continue
        new_off = off
        if poff > off:
            new_off = poff
        new_n = off + n - new_off
        if pend < off + n:
            new_n = pend - new_off
        pnew.append((ip, new_off - off, new_off, new_n))
    return pnew


def compute_det_sample_offsets(all_dets, old_dist, new_dist):
    """ """
    det_order = {y: x for x, y in enumerate(all_dets)}

    old_dets = list()
    for pdet in old_dist.dets:
        dfirst = det_order[pdet[0]]
        dlast = det_order[pdet[-1]]
        old_dets.append((dfirst, dlast - dfirst + 1))

    new_dets = list()
    for pdet in new_dist.dets:
        dfirst = det_order[pdet[0]]
        dlast = det_order[pdet[-1]]
        new_dets.append((dfirst, dlast - dfirst + 1))

    # Every process figures out its send and receive information

    send_row = compute_1d(self.local_index_offset, self.n_local_samples, newdist.samps)
    det_first = det_order[self.local_detectors[0]]
    det_last = det_order[self.local_detectors[-1]]
    send_col = compute_1d(det_first, det_last - det_first + 1, newdets)
    send_info = list()
    for sc in send_col:
        for sr in send_row:
            psend = sc[0] * newdist.comm_row_size + sr[0]
            send_info.append((psend, sc[1], sc[2], sc[3], sr[1], sr[2], sr[3]))

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
    for rc in recv_col:
        for rr in recv_row:
            precv = rc[0] * self.dist.comm_row_size + rr[0]
            recv_info.append((precv, rc[1], rc[2], rc[3], rr[1], rr[2], rr[3]))

    msg = "Proc {} det recv:  {}".format(self.comm_rank, recv_info)
    print(msg, flush=True)


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
            msg += "  Intervals will be deleted and not redistributed."
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
                all_ilist = self.comm_row.gather((ilist, self.n_local_samples), root=0)
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

    # Create the new detdata manager

    newdetdata = DetDataMgr(
        newdist.dets[newdist.comm_col_rank],
        newdist.samps[newdist.comm_row_rank][1],
    )

    # Redistribute detector data.  For purposes of redistribution, we require
    # that all DetectorData objects span the full set of local detectors.  This
    # allows all detectors to determine (without further communication) the new
    # and old distribution of all the data.
    #
    # In practice, smaller DetectorData objects are only used for intermediate data
    # products.  Data redistribution is something that will happen infrequently at
    # fixed points in a larger workflow.  If this ever becomes too much of a
    # limitation, we could add more logic to handle the case of detdata fields with
    # subsets of local detectors.

    send_counts = np.zeros(self.comm_size, dtype=np.int32)
    recv_counts = np.zeros(self.comm_size, dtype=np.int32)
    send_displ = np.zeros(self.comm_size, dtype=np.int32)
    recv_displ = np.zeros(self.comm_size, dtype=np.int32)

    for field in list(self.detdata.keys()):
        field_dets = self.detdata[field].detectors
        if field_dets != self.local_detectors:
            msg = "Redistribution only supports detdata with all local detectors."
            msg += " Field {} has {} dets instead of {}".format(
                field, len(field_dets), len(self.local_detectors)
            )
            raise NotImplementedError(msg)

        # Get MPI data type.  Note that in the case of no MPI, this function would
        # have returned at the very start.  So we know that we have a real
        # communicator at this point.
        mpibytesize, mpitype = mpi_data_type(self.comm, self.detdata[field].dtype)

        # Allocate new data
        sample_shape = None
        if len(self.detdata[field].detector_shape) == 1:
            # One number per sample
            sample_shape = (1,)
        else:
            sample_shape = self.detdata[field].detector_shape[1:]

        newdetdata.create(
            field,
            sample_shape=sample_shape,
            dtype=self.detdata[field].dtype,
            detectors=None,
        )

        # Set up the send / receive counts and displacements

        send_counts[:] = 0
        send_displ[:] = 0
        for (
            p,
            loc_det_off,
            glob_det_off,
            n_det,
            loc_samp_off,
            glob_samp_off,
            n_samp,
        ) in send_info:
            pass

        # Communicate data

        self._dist.comm.Alltoallv(
            [
                self.detdata[field].flatdata,
                self._send_counts,
                self._send_displ,
                mpitype,
            ],
            [
                newdetdata[field].flatdata,
                self._recv_counts,
                self._recv_displ,
                mpitype,
            ],
        )

        del self.detdata[field]

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

    # Swap new data objects into place
