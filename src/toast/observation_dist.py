# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numbers
import sys
from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np
from pshmem.utils import mpi_data_type

from .dist import DistRange, distribute_samples
from .mpi import MPI, comm_equal, comm_equivalent
from .observation_data import (
    DetDataManager,
    DetectorData,
    IntervalsManager,
    SharedDataManager,
    SharedDataType,
)
from .observation_view import DetDataView, SharedView, View, ViewInterface, ViewManager
from .timing import function_timer
from .utils import AlignedI32, Logger, dtype_to_aligned, name_UID


class DistDetSamp(object):
    """Class used within an Observation to store the detector and sample distribution.

    This is just a simple container for various properties of the distribution.

    Args:
        samples (int):  The total number of samples.
        detectors (list):  The full list of detector names.  Note that detector_sets
            may be used to prune this full list to only include a subset of detectors.
        detector_sets (list or dict):  (Optional) List of lists containing detector
            names or dictionary of lists with detector names (the keys are irrelevant).
            These discrete detector sets are used to distribute detectors- a detector
            set will always be within a single row of the process grid.  If None,
            every detector is a set of one.
        sample_sets (list):  (Optional) List of lists of chunk sizes (integer numbers of
            samples).  These discrete sample sets are used to distribute sample data.
            A sample set will always be within a single column of the process grid.  If
            None, any distribution break in the sample direction will happen at an
            arbitrary place.  The sum of all chunks must equal the total number of
            samples.
        comm (toast.Comm):  The communicator to use.
        process_rows (int):  (Optional) The size of the rectangular process grid
            in the detector direction.  This number must evenly divide into the size of
            comm.  If not specified, defaults to the size of the communicator.

    """

    @function_timer
    def __init__(
        self,
        samples,
        detectors,
        sample_sets,
        detector_sets,
        comm,
        process_rows,
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

        if self.process_rows is None:
            if self.comm.comm_group is None:
                # No MPI, default to 1
                self.process_rows = 1
            else:
                # We have MPI, default to the size of the group
                self.process_rows = self.comm.group_size

        # Get the row / column communicators for this grid shape

        rowcolcomm = self.comm.comm_row_col(self.process_rows)
        self.comm_row = rowcolcomm["row"]
        self.comm_col = rowcolcomm["col"]
        self.comm_row_node = rowcolcomm["row_node"]
        self.comm_col_node = rowcolcomm["col_node"]
        self.comm_row_rank_node = rowcolcomm["row_rank_node"]
        self.comm_col_rank_node = rowcolcomm["col_rank_node"]

        self.comm_row_size = 1
        self.comm_row_rank = 0
        if self.comm_row is not None:
            self.comm_row_size = self.comm_row.size
            self.comm_row_rank = self.comm_row.rank
        self.comm_col_size = 1
        self.comm_col_rank = 0
        if self.comm_col is not None:
            self.comm_col_size = self.comm_col.size
            self.comm_col_rank = self.comm_col.rank

        # If detector_sets is specified, check consistency.

        if self.detector_sets is not None:
            # We have some detector sets.  Verify that every det set contains detectors
            # in the full list.  Make a new list of detectors including only those in
            # the det sets.
            new_dets = list()
            detsets = list()
            detectors_set = set(self.detectors)
            if isinstance(self.detector_sets, list):
                # We have a list of lists
                for ds in self.detector_sets:
                    for d in ds:
                        if d not in detectors_set:
                            raise RuntimeError(
                                f"detector {d} in a detset but not in detector list"
                            )
                        new_dets.append(d)
                    detsets.append(list(ds))
            elif isinstance(self.detector_sets, dict):
                # We have a detector group dictionary
                for ds in self.detector_sets.values():
                    for d in ds:
                        if d not in detectors_set:
                            raise RuntimeError(
                                f"detector {d} in a detset but not in detector list"
                            )
                        new_dets.append(d)
                    detsets.append(list(ds))
            else:
                raise RuntimeError("detector_sets should be a list or dict")

            # Replace original detector list with only those found in detsets.
            # Replace detector_sets with the structure needed by the low level
            # distribution code.
            self.detectors = new_dets
            self.detector_sets = detsets

        # Detector name to index
        self.det_index = {y: x for x, y in enumerate(self.detectors)}

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
            self.comm.comm_group,
            self.detectors,
            self.samples,
            detranks=self.process_rows,
            detsets=self.detector_sets,
            sampsets=self.sample_sets,
        )

        self.det_indices = list()
        for ds in self.dets:
            dfirst = self.det_index[ds[0]]
            dlast = self.det_index[ds[-1]]
            self.det_indices.append(DistRange(dfirst, dlast - dfirst + 1))

        if self.comm.group_rank == 0:
            # check that all processes have some data, otherwise print warning
            for i, ds in enumerate(self.dets):
                if len(ds) == 0:
                    msg = f"Process {i} has no detectors assigned."
                    log.warning(msg)
            for i, ss in enumerate(self.samps):
                if ss[1] == 0:
                    msg = f"Process {i} has no samples assigned."
                    log.warning(msg)

    def __eq__(self, other):
        log = Logger.get()
        fail = 0
        if self.process_rows != other.process_rows:
            fail = 1
            log.verbose(f"  process_rows {self.process_rows} != {other.process_rows}")
        if not comm_equivalent(self.comm.comm_group, other.comm.comm_group):
            fail = 1
            log.verbose(f"  obs group comm not equivalent")
        # If the group comms are equivalent, then we know that the row / col
        # comms are equivalent, since they come from the same cache for a given
        # value of process_rows.
        #
        # Test the resulting distribution quantities, rather than the
        # inputs passed to the constructor, in case those are slightly
        # different types.
        if self.dets != other.dets:
            fail = 1
            log.verbose(f"  dist dets {self.dets} != {other.dets}")
        if self.det_sets != other.det_sets:
            fail = 1
            log.verbose(f"  dist detsets {self.det_sets} != {other.det_sets}")
        if self.samps != other.samps:
            fail = 1
            log.verbose(f"  dist samps {self.samps} != {other.samps}")
        if self.samp_sets != other.samp_sets:
            fail = 1
            log.verbose(f"  dist samp_sets {self.samp_sets} != {other.samp_sets}")
        if self.comm.comm_group is not None:
            fail = self.comm.comm_group.allreduce(fail, op=MPI.SUM)
        return fail == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        s = f"<DistDetSamp "
        s += f"P_row {self.comm_row_rank}/{self.comm_row_size} "
        s += f"P_col {self.comm_col_rank}/{self.comm_col_size} "
        s += "dets=["
        for d in self.dets[self.comm.group_rank]:
            s += f"{d},"
        s += "]"
        s += " samples=["
        off = self.samps[self.comm.group_rank].offset
        nsamp = self.samps[self.comm.group_rank].n_elem
        s += f"{off} ... {off+nsamp}]>"
        return s


def compute_1d_offsets(off, n, target_dist):
    """Helper function to compute slices along one dimension.

    Args:
        off (int):  The offset of the local data.
        n (int):  The local number of elements.
        target_dist (list):  A list of tuples, one per process, with the offset
            and number of elements for that process in the new distribution.

    Returns:
        (list):  A list of slices (one per process) with the slice of local data
            that overlaps the data on the target process.  If there is no overlap
            a None value is returned.

    """
    out = list()
    for ip, p in enumerate(target_dist):
        toff = p[0]
        tend = p[0] + p[1]
        if toff >= off + n:
            out.append(None)
            continue
        if tend <= off:
            out.append(None)
            continue
        local_off = 0
        target_off = 0
        if toff >= off:
            local_off = toff - off
        else:
            target_off = off - toff
        tnum = n - local_off
        if tend < off + n:
            tnum = tend - local_off
        out.append(slice(local_off, local_off + tnum, 1))
    return out


def redistribute_buffer(
    comm,
    buffer_class,
    mpi_type,
    input,
    output,
    send_info,
    recv_info,
):
    """Low-level Alltoallv redistribution of multidimensional buffers.

    The send and receive slices for each process are used to construct flat-packed
    buffers and copy data into and out of them.

    Args:
        comm (mpi4py.Comm):  The communicator
        buffer_class (object):  The AlignedXX class used for the send and receive
            buffers.
        mpi_type (MPI.Type):  The MPI data type.
        input (array-like):  The multidimensional local piece of the input.
        output (array-like):  The multidimensional local piece of the output.
        send_info (list):  A list of tuples, one per process, with the send slices.
        recv_info (list):  A list of tuples, one per process, with the receive slices.

    Returns:
        None

    """
    # Per-process send and receive information in terms of the flat
    # packed buffers.
    send_counts = AlignedI32(comm.size)
    recv_counts = AlignedI32(comm.size)
    send_displ = AlignedI32(comm.size)
    recv_displ = AlignedI32(comm.size)

    off = 0
    send_shape = list()
    for iproc, send_slc in enumerate(send_info):
        send_displ[iproc] = off
        n_elem = 0
        if send_slc is not None:
            n_elem = 1
            ss = list()
            for slc in send_slc:
                slen = slc.stop - slc.start
                n_elem *= slen
                ss.append(slen)
            send_shape.append(tuple(ss))
        else:
            send_shape.append(None)
        send_counts[iproc] = n_elem
        off += n_elem
    send_n_elem = off

    off = 0
    recv_shape = list()
    for iproc, recv_slc in enumerate(recv_info):
        recv_displ[iproc] = off
        n_elem = 0
        if recv_slc is not None:
            n_elem = 1
            rs = list()
            for slc in recv_slc:
                rlen = slc.stop - slc.start
                n_elem *= rlen
                rs.append(rlen)
            recv_shape.append(tuple(rs))
        else:
            recv_shape.append(None)
        recv_counts[iproc] = n_elem
        off += n_elem
    recv_n_elem = off

    # Set up the send / receive buffers
    send_buf = buffer_class(send_n_elem)
    recv_buf = buffer_class(recv_n_elem)

    # Copy data into the send buffer
    for iproc, send_slc in enumerate(send_info):
        off = send_displ[iproc]
        n_elem = send_counts[iproc]
        if n_elem > 0:
            send_buf[off : off + n_elem] = input[send_slc].flatten()

    # Communicate data
    comm.Alltoallv(
        [
            send_buf,
            send_counts,
            send_displ,
            mpi_type,
        ],
        [
            recv_buf,
            recv_counts,
            recv_displ,
            mpi_type,
        ],
    )

    # Copy data to output
    for iproc, recv_slc in enumerate(recv_info):
        off = recv_displ[iproc]
        n_elem = recv_counts[iproc]
        if n_elem > 0:
            output[recv_slc] = (
                recv_buf[off : off + n_elem].array().reshape(recv_shape[iproc])
            )

    # Delete buffers
    send_buf.clear()
    del send_buf
    recv_buf.clear()
    del recv_buf
    send_counts.clear()
    del send_counts
    send_displ.clear()
    del send_displ
    recv_counts.clear()
    del recv_counts
    recv_displ.clear()
    del recv_displ


def global_interval_times(dist, intervals_manager, name, join=False):
    """Return the global list of interval timespans on the root process.

    After creation, the list of intervals is local to each process.  This function
    reconstructs the full global list of interval times on the rank zero process
    of the group.

    Args:
        dist (DistDetSamp):  The data distribution in the observation.
        intervals_manager (IntervalsManager):  The manager instance (e.g. obs.intervals)
        name (str):  The name of the object.
        join (bool):  If True, join together intervals that are broken by distribution
            boundaries in the sample direction.

    Returns:
        (list):  List of tuples on the root process, and None on other processes.

    """
    ilist = [(x.start, x.stop, x.first, x.last) for x in intervals_manager[name]]
    all_ilist = None
    if dist.comm_row is None:
        all_ilist = [(ilist, dist.samps[dist.comm.group_rank].n_elem)]
    else:
        # Gather across the process row
        if dist.comm_col_rank == 0:
            all_ilist = dist.comm_row.gather(
                (ilist, dist.samps[dist.comm.group_rank].n_elem), root=0
            )
    del ilist

    glist = None
    if dist.comm.group_rank == 0:
        # Only one process builds the list of start / stop times.  By definition,
        # the rank zero process of the observation is also the process with rank
        # zero along both the rows and columns.
        glist = list()
        last_continue = False
        last_start = 0
        last_stop = 0
        for pdata, pn in all_ilist:
            if len(pdata) == 0:
                continue
            for start, stop, first, last in pdata:
                if last_continue:
                    if first == 0:
                        last_stop = stop
                    else:
                        glist.append((last_start, last_stop))
                        last_continue = False
                        last_start = start
                        last_stop = stop
                else:
                    last_start = start
                    last_stop = stop

                if join:
                    if last == pn - 1:
                        last_continue = True
                    else:
                        glist.append((last_start, last_stop))
                        last_continue = False
                else:
                    glist.append((last_start, last_stop))
        if last_continue:
            # add final range
            glist.append((last_start, last_stop))
    return glist


def extract_global_intervals(old_dist, intervals_manager):
    """Helper function to reconstruct global interval timespans.

    After an IntervalList is added, only the local intervals on each process are
    kept.  We need to reconstruct the original timespans that were used and save,
    so that we can rebuild them after the timestamps have been redistributed.

    Args:
        old_dist (DistDetSamp):  The existing data distribution.
        intervals_manager (IntervalsManager):  The existing interval manager instance.

    Returns:
        (dict):  The dictionary of reconstructed global timespan intervals on rank
            zero of the observation, otherwise None.

    """
    log = Logger.get()

    global_intervals = None
    if old_dist.comm.group_rank == 0:
        global_intervals = dict()

    for iname in list(intervals_manager.keys()):
        if iname == intervals_manager.all_name:
            continue
        result = global_interval_times(old_dist, intervals_manager, iname, join=False)
        if old_dist.comm.group_rank == 0:
            global_intervals[iname] = result

    return global_intervals


def redistribute_detector_data(
    old_dist,
    new_dist,
    detdata_manager,
    old_local_dets,
    det_send_info,
    samp_send_info,
    det_recv_info,
    samp_recv_info,
):
    """Redistribute detector data.

    For purposes of redistribution, we require that all DetectorData objects span the
    full set of local detectors.  In practice, smaller DetectorData objects are only
    used for intermediate data products.  Data redistribution is something that will
    happen infrequently at fixed points in a larger workflow.  If this ever becomes too
    much of a limitation, we could add more logic to handle the case of detdata fields
    with subsets of local detectors.

    Args:
        old_dist (DistDetSamp):  The existing data distribution.
        new_dist (DistDetSamp):  The new data distribution.
        detdata_manager (DetectorDataManager):  The existing detector data manager
            instance.
        old_local_dets (list):  The local detectors in the old distribution.
        det_send_info (list):  The send slices along the detector axis.
        samp_send_info (list):  The send slices along the sample axis.
        det_recv_info (list):  The receive slices along the detector axis.
        samp_recv_info (list):  The receive slices along the sample axis.

    Returns:
        (tuple):  The new DetDataManager.

    """
    log = Logger.get()

    new_detdata_manager = DetDataManager(new_dist)

    # Process every detdata object

    for field in list(detdata_manager.keys()):
        field_dets = detdata_manager[field].detectors

        if not all([x == y for x, y in zip(field_dets, old_local_dets)]):
            msg = "Redistribution only supports detdata with all local detectors."
            msg += f" Field {field} has {len(field_dets)} dets instead of "
            msg += f"{len(old_local_dets)}.  Deleting."
            log.warning_rank(msg, comm=old_dist.comm.comm_group)
            del detdata_manager[field]
            continue

        # Buffer class to use
        buffer_class, _ = dtype_to_aligned(detdata_manager[field].dtype)

        # Get MPI data type.  Note that in the case of no MPI, this function would
        # have returned at the very start.  So we know that we have a real
        # communicator at this point.
        mpibytesize, mpitype = mpi_data_type(
            old_dist.comm.comm_group, detdata_manager[field].dtype
        )

        # Units
        units = detdata_manager[field].units

        # Allocate new data
        sample_shape = None
        n_per_sample = 1
        if len(detdata_manager[field].detector_shape) > 1:
            sample_shape = detdata_manager[field].detector_shape[1:]
            for dim in sample_shape:
                n_per_sample *= dim

        new_detdata_manager.create(
            field,
            sample_shape=sample_shape,
            dtype=detdata_manager[field].dtype,
            units=units,
        )

        # Redistribution send / recv slices
        samp_slices = None
        if sample_shape is not None:
            samp_slices = [slice(0, x, 1) for x in sample_shape]
        send_info = list()
        for dinfo, sinfo in zip(det_send_info, samp_send_info):
            proc_slices = None
            if dinfo is not None and sinfo is not None:
                proc_slices = [dinfo, sinfo]
                if samp_slices is not None:
                    proc_slices.extend(samp_slices)
                proc_slices = tuple(proc_slices)
            send_info.append(proc_slices)

        recv_info = list()
        for dinfo, sinfo in zip(det_recv_info, samp_recv_info):
            proc_slices = None
            if dinfo is not None and sinfo is not None:
                proc_slices = [dinfo, sinfo]
                if samp_slices is not None:
                    proc_slices.extend(samp_slices)
                proc_slices = tuple(proc_slices)
            recv_info.append(proc_slices)

        # Redistribute
        redistribute_buffer(
            old_dist.comm.comm_group,
            buffer_class,
            mpitype,
            detdata_manager[field].data,
            new_detdata_manager[field].data,
            send_info,
            recv_info,
        )
    return new_detdata_manager


def redistribute_shared_data(
    old_dist,
    new_dist,
    shared_manager,
    old_det_n,
    new_det_n,
    old_samp_n,
    new_samp_n,
    det_send_info,
    samp_send_info,
    det_recv_info,
    samp_recv_info,
):
    """Redistribute shared data.

    Shared data.  If the data is shared across the observation (group) communicator,
    then no action is needed.  If it is shared across a row or column communicator,
    then only the rank zero processes in those communicators need to do anything.

    In order to determine how to split up and recombine shared objects, we require
    that the leading dimension of the object corresponds to either the number of
    local detectors (for row-shared objects) or the number of local samples (for
    column-shared objects).

    Args:
        old_dist (DistDetSamp):  The existing data distribution.
        new_dist (DistDetSamp):  The new data distribution.
        shared_manager (SharedManager):  The existing detector data manager
            instance.
        old_det_n (int):  The old number of local detectors.
        new_det_n (int):  The new number of local detectors.
        old_samp_n (int):  The old number of local samples.
        new_samp_n (int):  The new number of local samples.
        det_send_info (list):  The send slices along the detector axis.
        samp_send_info (list):  The send slices along the sample axis.
        det_recv_info (list):  The receive slices along the detector axis.
        samp_recv_info (list):  The receive slices along the sample axis.

    Returns:
        (tuple):  The new DetDataManager.

    """
    log = Logger.get()

    # Create the new shared manager.
    new_shared_manager = SharedDataManager(new_dist)

    for field in shared_manager.keys():
        shobj = shared_manager[field]
        commtype = shared_manager.comm_type(field)
        if commtype == "group":
            # Using full group communicator, just copy to new data manager.
            new_shared_manager.assign_mpishared(field, shobj, commtype)
            continue

        # Full shape of the object
        shp = shobj.shape

        # Buffer type
        buffer_class, _ = dtype_to_aligned(shobj.dtype)
        mpibytesize, mpitype = mpi_data_type(shobj.comm, shobj.dtype)

        # Shape of the non-leading dimensions
        other_shape = None
        n_per_leading = 1
        if len(shp) > 1:
            other_shape = shp[1:]
            for dim in other_shape:
                n_per_leading *= dim

        # slices for non-leading dimensions
        dim_slices = None
        if other_shape is not None:
            dim_slices = [slice(0, x, 1) for x in other_shape]

        if commtype == "row":
            # Shared in the sample direction.
            if shp[0] != old_det_n:
                msg = f"Shared object {field} uses the row communicator, "
                msg += f"but has leading dimension {shp[0]} rather than the "
                msg += f"number of local detectors ({old_det_n}).  "
                msg += f"Cannot redistribute."
                raise RuntimeError(msg)

            # Redistribution send / recv slices
            send_info = [None for x in range(old_dist.comm_row_size)]
            recv_info = [None for x in range(old_dist.comm_row_size)]

            if old_dist.comm_row_rank == 0:
                # We are sending something
                send_info = list()
                for rproc, sinfo in enumerate(det_send_info):
                    proc_slices = None
                    if sinfo is not None and rproc % new_dist.comm_row_size == 0:
                        proc_slices = [sinfo]
                        if dim_slices is not None:
                            proc_slices.extend(dim_slices)
                        proc_slices = tuple(proc_slices)
                    send_info.append(proc_slices)
            if new_dist.comm_row_rank == 0:
                # We are receiving something
                recv_info = list()
                for sproc, rinfo in enumerate(det_recv_info):
                    proc_slices = None
                    if rinfo is not None and sproc % old_dist.comm_row_size == 0:
                        proc_slices = [rinfo]
                        if dim_slices is not None:
                            proc_slices.extend(dim_slices)
                        proc_slices = tuple(proc_slices)
                    recv_info.append(proc_slices)

            # Create the new object
            new_shp = [new_det_n]
            if other_shape is not None:
                new_shp.extend(other_shape)

            new_shared_manager.create_row(field, new_shp, dtype=shobj.dtype)

            # Redistribute
            redistribute_buffer(
                old_dist.comm.comm_group,
                buffer_class,
                mpitype,
                shobj.data,
                new_shared_manager[field].data,
                send_info,
                recv_info,
            )
        elif commtype == "column":
            # Shared in the detector direction
            shp = shobj.shape
            if shp[0] != old_samp_n:
                msg = f"Shared object {field} uses the column communicator, "
                msg += f"but has leading dimension {shp[0]} rather than the "
                msg += f"number of local samples {old_samp_n}.  "
                msg += f"Cannot redistribute."
                raise RuntimeError(msg)

            # Redistribution send / recv slices
            send_info = [None for x in range(old_dist.comm_col_size)]
            recv_info = [None for x in range(old_dist.comm_col_size)]
            if old_dist.comm_col_rank == 0:
                # We are sending something
                send_info = list()
                for rproc, sinfo in enumerate(samp_send_info):
                    proc_slices = None
                    if sinfo is not None and rproc % new_dist.comm_col_size == 0:
                        proc_slices = [sinfo]
                        if dim_slices is not None:
                            proc_slices.extend(dim_slices)
                        proc_slices = tuple(proc_slices)
                    send_info.append(proc_slices)
            if new_dist.comm_col_rank == 0:
                # We are receiving something
                recv_info = list()
                for sproc, rinfo in enumerate(samp_recv_info):
                    proc_slices = None
                    if rinfo is not None and sproc % old_dist.comm_col_size == 0:
                        proc_slices = [rinfo]
                        if dim_slices is not None:
                            proc_slices.extend(dim_slices)
                        proc_slices = tuple(proc_slices)
                    recv_info.append(proc_slices)

            # Create the new object
            new_shp = [new_samp_n]
            if other_shape is not None:
                new_shp.extend(other_shape)
            new_shared_manager.create_column(field, new_shp, dtype=shobj.dtype)

            # Redistribute
            redistribute_buffer(
                old_dist.comm.comm_group,
                buffer_class,
                mpitype,
                shobj.data,
                new_shared_manager[field].data,
                send_info,
                recv_info,
            )
        else:
            # Note- this code should never be executed, since the commtype value
            # comes from an existing object and is therefore guaranteed to be
            # valid.
            msg = "Only shared objects using the group, row, and column "
            msg += "communicators can be redistributed"
            log.error(msg)
            raise RuntimeError(msg)

    return new_shared_manager


def redistribute_data(
    old_dist,
    new_dist,
    shared_manager,
    detdata_manager,
    intervals_manager,
    times=None,
    dbg=None,
):
    """Helper function to redistribute data in an observation.

    Given the old and new data distributions, redistribute all objects in the
    shared, detdata, and intervals manager objects.  The input managers are cleared
    upon return.  If times is None and there exist some intervals, a warning
    will be given and the intervals will be deleted.

    Args:
        old_dist (DistDetSamp):  The existing data distribution.
        new_dist (DistDetSamp):  The new data distribution.
        shared_manager (SharedDataManager):  The existing shared manager instance.
        detdata_manager (DetectorDataManager):  The existing detector data manager
            instance.
        intervals_manager (IntervalsManager):  The existing interval manager instance.
        times (str):  The name of the shared object containing the timestamps.

    Returns:
        (tuple):  The new (SharedDataManager, DetDataManager, IntervalsManager)
            instances.

    """
    log = Logger.get()

    global_intervals = dict()
    if times is None and len(intervals_manager.keys()) > 0:
        if old_dist.comm.group_rank == 0:
            msg = "Time stamps not specified when redistributing observation."
            msg += "  Intervals will be deleted and not redistributed."
            log.warning(msg)
    else:
        global_intervals = extract_global_intervals(old_dist, intervals_manager)

    intervals_manager.clear()

    # Compute the detector and sample ranges we are sending and receiving.  These are
    # common for all detdata objects.

    old_det_off = old_dist.det_indices[old_dist.comm.group_rank].offset
    old_det_n = old_dist.det_indices[old_dist.comm.group_rank].n_elem
    old_local_dets = old_dist.detectors[old_det_off : old_det_off + old_det_n]
    det_send_info = compute_1d_offsets(old_det_off, old_det_n, new_dist.det_indices)

    old_samp_off = old_dist.samps[old_dist.comm.group_rank].offset
    old_samp_n = old_dist.samps[old_dist.comm.group_rank].n_elem
    samp_send_info = compute_1d_offsets(old_samp_off, old_samp_n, new_dist.samps)

    new_det_off = new_dist.det_indices[new_dist.comm.group_rank].offset
    new_det_n = new_dist.det_indices[new_dist.comm.group_rank].n_elem
    det_recv_info = compute_1d_offsets(new_det_off, new_det_n, old_dist.det_indices)

    new_samp_off = new_dist.samps[new_dist.comm.group_rank].offset
    new_samp_n = new_dist.samps[new_dist.comm.group_rank].n_elem
    samp_recv_info = compute_1d_offsets(new_samp_off, new_samp_n, old_dist.samps)

    # Redistribute detector data.

    new_detdata_manager = redistribute_detector_data(
        old_dist,
        new_dist,
        detdata_manager,
        old_local_dets,
        det_send_info,
        samp_send_info,
        det_recv_info,
        samp_recv_info,
    )

    # Redistribute shared data

    new_shared_manager = redistribute_shared_data(
        old_dist,
        new_dist,
        shared_manager,
        old_det_n,
        new_det_n,
        old_samp_n,
        new_samp_n,
        det_send_info,
        samp_send_info,
        det_recv_info,
        samp_recv_info,
    )

    # Re-create the intervals in the new data distribution.
    new_intervals_manager = IntervalsManager(
        new_dist, new_dist.samps[new_dist.comm.group_rank].n_elem
    )

    # Communicate the field list
    gkeys = None
    if old_dist.comm.group_rank == 0:
        gkeys = list(global_intervals.keys())
    ivl_fields = old_dist.comm.comm_group.bcast(gkeys, root=0)

    for field in ivl_fields:
        glb = None
        if old_dist.comm.group_rank == 0:
            glb = global_intervals[field]
        new_intervals_manager.create(field, glb, new_shared_manager[times], fromrank=0)

    return new_shared_manager, new_detdata_manager, new_intervals_manager
