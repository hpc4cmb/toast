# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import json
import os

import h5py
import numpy as np
from astropy import units as u
from astropy.table import Table

from ..instrument import GroundSite
from ..mpi import MPI
from ..observation import default_values as defaults
from ..observation_data import DetectorData
from ..observation_dist import global_interval_times
from ..timing import GlobalTimers, Timer, function_timer
from ..utils import (
    Environment,
    Logger,
    dtype_to_aligned,
    hdf5_use_serial,
    object_fullname,
)
from .compression import compress_detdata, decompress_detdata
from .hdf_utils import check_dataset_buffer_size, hdf5_open


@function_timer
def save_hdf5_shared(obs, hgrp, fields, log_prefix):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # Get references to the distribution of detectors and samples
    proc_rows = obs.dist.process_rows
    proc_cols = obs.dist.comm.group_size // proc_rows
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    # Are we doing serial I/O?
    use_serial = hdf5_use_serial(hgrp, obs.comm.comm_group)

    for ifield, field in enumerate(fields):
        tag_offset = (obs.comm.group * 1000 + ifield) * obs.comm.group_size

        if field not in obs.shared:
            msg = f"Shared data '{field}' does not exist in observation "
            msg += f"{obs.name}.  Skipping."
            log.warning_rank(msg, comm=obs.comm.comm_group)
            continue

        # Compute properties of the full set of data across the observation

        scomm = obs.shared.comm_type(field)
        sdata = obs.shared[field]
        sdtype = sdata.dtype

        if scomm == "group":
            sshape = sdata.shape
        elif scomm == "column":
            sshape = (obs.n_all_samples,) + sdata.shape[1:]
        else:
            sshape = (len(obs.all_detectors),) + sdata.shape[1:]

        # The buffer class to use for allocating receive buffers
        bufclass, _ = dtype_to_aligned(sdtype)

        hdata = None
        if hgrp is not None:
            # This process is participating
            hdata = hgrp.create_dataset(field, sshape, dtype=sdtype)
            hdata.attrs["comm_type"] = scomm

        if use_serial:
            # Send data to rank zero of the group for writing.
            if scomm == "group":
                # Easy...
                if obs.comm.group_rank == 0:
                    hdata[:] = sdata.data
            elif scomm == "column":
                # Send data to root process
                for proc in range(proc_cols):
                    # Process grid is indexed row-major, so the rank-zero process
                    # of each column is just the first row of the grid.
                    send_rank = proc

                    # Leading data range for this process
                    off = dist_samps[send_rank].offset
                    nelem = dist_samps[send_rank].n_elem
                    nflat = nelem * np.prod(sshape[1:])
                    shp = (nelem,) + sshape[1:]
                    if send_rank == 0:
                        # Root process writes local data
                        if obs.comm.group_rank == 0:
                            hdata[off : off + nelem] = sdata.data
                    elif send_rank == obs.comm.group_rank:
                        # We are sending
                        obs.comm.comm_group.Send(
                            sdata.data.flatten(), dest=0, tag=tag_offset + send_rank
                        )
                    elif obs.comm.group_rank == 0:
                        # We are receiving and writing
                        recv = bufclass(nflat)
                        obs.comm.comm_group.Recv(
                            recv, source=send_rank, tag=tag_offset + send_rank
                        )
                        hdata[off : off + nelem] = recv.array().reshape(shp)
                        recv.clear()
                        del recv
            else:
                # Send data to root process
                for proc in range(proc_rows):
                    # Process grid is indexed row-major, so the rank-zero process
                    # of each row is strided by the number of columns.
                    send_rank = proc * proc_cols
                    # Leading data range for this process
                    off = dist_dets[send_rank].offset
                    nelem = dist_dets[send_rank].n_elem
                    nflat = nelem * np.prod(sshape[1:])
                    shp = (nelem,) + sshape[1:]
                    if send_rank == 0:
                        # Root process writes local data
                        if obs.comm.group_rank == 0:
                            hdata[off : off + nelem] = sdata.data
                    elif send_rank == obs.comm.group_rank:
                        # We are sending
                        obs.comm.comm_group.Send(
                            sdata.data.flatten(), dest=0, tag=tag_offset + send_rank
                        )
                    elif obs.comm.group_rank == 0:
                        # We are receiving and writing
                        recv = bufclass(nflat)
                        obs.comm.comm_group.Recv(
                            recv, source=send_rank, tag=tag_offset + send_rank
                        )
                        hdata[off : off + nelem] = recv.array().reshape(shp)
                        recv.clear()
                        del recv
        else:
            # If we have parallel support, the rank zero of each comm can write
            # independently.
            if scomm == "group":
                # Easy...
                if obs.comm.group_rank == 0:
                    msg = f"Shared field {field} ({scomm})"
                    slices = tuple([slice(0, x) for x in sshape])
                    check_dataset_buffer_size(msg, slices, sdtype, True)
                    hdata.write_direct(sdata.data, slices, slices)
            elif scomm == "column":
                # Rank zero of each column writes
                if sdata.comm is None or sdata.comm.rank == 0:
                    sh_slices = tuple([slice(0, x) for x in sshape])
                    offset = dist_samps[obs.comm.group_rank].offset
                    nelem = dist_samps[obs.comm.group_rank].n_elem
                    hf_slices = [
                        slice(offset, offset + nelem),
                    ]
                    hf_slices.extend([slice(0, x) for x in sdata.shape[1:]])
                    hf_slices = tuple(hf_slices)
                    msg = f"Shared field {field} ({scomm})"
                    check_dataset_buffer_size(msg, hf_slices, sdtype, True)
                    hdata.write_direct(sdata.data, sh_slices, hf_slices)
            else:
                # Rank zero of each row writes
                if sdata.comm is None or sdata.comm.rank == 0:
                    sh_slices = tuple([slice(0, x) for x in sshape])
                    offset = dist_dets[obs.comm.group_rank].offset
                    nelem = dist_dets[obs.comm.group_rank].n_elem
                    hf_slices = [
                        slice(offset, offset + nelem),
                    ]
                    hf_slices.extend([slice(0, x) for x in sdata.shape[1:]])
                    hf_slices = tuple(hf_slices)
                    msg = f"Shared field {field} ({scomm})"
                    check_dataset_buffer_size(msg, hf_slices, sdtype, True)
                    hdata.write_direct(sdata.data, sh_slices, hf_slices)

        log.verbose_rank(
            f"{log_prefix}  Shared finished {field} write in",
            comm=obs.comm.comm_group,
            timer=timer,
        )
        del hdata


@function_timer
def save_hdf5_detdata(obs, hgrp, fields, log_prefix, use_float32=False):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # Get references to the distribution of detectors and samples
    proc_rows = obs.dist.process_rows
    proc_cols = obs.dist.comm.group_size // proc_rows
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    # We are using the group communicator
    comm = obs.comm.comm_group
    nproc = obs.comm.group_size
    rank = obs.comm.group_rank

    # Are we doing serial I/O?
    use_serial = hdf5_use_serial(hgrp, comm)

    for ifield, (field, fieldcomp) in enumerate(fields):
        tag_offset = (obs.comm.group * 1000 + ifield) * obs.comm.group_size
        if field not in obs.detdata:
            msg = f"Detector data '{field}' does not exist in observation "
            msg += f"{obs.name}.  Skipping."
            log.warning_rank(msg, comm=comm)
            continue

        local_data = obs.detdata[field]
        if local_data.detectors != obs.local_detectors:
            msg = f"Detector data '{field}' does not contain all local detectors."
            log.error(msg)
            raise RuntimeError(msg)

        # If we are using compression, we require the data to be distributed by
        # by detector, since we will compress each detector independently.
        if fieldcomp is not None and proc_cols != 1:
            msg = f"Detector data '{field}' compression requested, but data for "
            msg += f"individual channels is split between processes."
            raise RuntimeError(msg)

        # Compute properties of the full set of data across the observation
        ddtype = local_data.dtype
        dshape = (len(obs.all_detectors), obs.n_all_samples)
        dvalshape = None
        if len(local_data.detector_shape) > 1:
            dvalshape = local_data.detector_shape[1:]
            dshape += dvalshape

        fdtype = ddtype
        if ddtype.char == "d" and use_float32:
            # We are truncating to single precision
            fdtype = np.dtype(np.float32)

        # If we are using our own internal compression, each process compresses their
        # local data and sends it to one process for insertion into the overall blob
        # of bytes.

        if fieldcomp is None or "type_hdf5" in fieldcomp:
            # The buffer class to use for allocating receive buffers
            bufclass, _ = dtype_to_aligned(ddtype)

            # Group handle
            hdata = None
            if hgrp is not None:
                # This process is participating.
                #
                # Future NOTE:  Here is where we could extract the "type_hdf5" parameter
                # from the dictionary and create the dataset with appropriate compression
                # and chunking settings.  Detector data would then be written to the
                # dataset in the usual way, with compression "under the hood" done by
                # HDF5.
                #
                hdata = hgrp.create_dataset(field, dshape, dtype=fdtype)
                hdata.attrs["units"] = local_data.units.to_string()

            if use_serial:
                # Send data to rank zero of the group for writing.
                for proc in range(nproc):
                    # Data ranges for this process
                    samp_off = dist_samps[proc].offset
                    samp_nelem = dist_samps[proc].n_elem
                    det_off = dist_dets[proc].offset
                    det_nelem = dist_dets[proc].n_elem
                    nflat = det_nelem * samp_nelem
                    shp = (det_nelem, samp_nelem)
                    detdata_slice = [slice(0, det_nelem, 1), slice(0, samp_nelem, 1)]
                    hf_slice = [
                        slice(det_off, det_off + det_nelem, 1),
                        slice(samp_off, samp_off + samp_nelem, 1),
                    ]
                    if dvalshape is not None:
                        nflat *= np.prod(dvalshape)
                        shp += dvalshape
                        detdata_slice.extend([slice(0, x) for x in dvalshape])
                        hf_slice.extend([slice(0, x) for x in dvalshape])
                    detdata_slice = tuple(detdata_slice)
                    hf_slice = tuple(hf_slice)
                    if proc == 0:
                        # Root process writes local data
                        if rank == 0:
                            hdata.write_direct(
                                local_data.data.astype(fdtype), detdata_slice, hf_slice
                            )
                    elif proc == rank:
                        # We are sending
                        comm.Send(local_data.flatdata, dest=0, tag=tag_offset + proc)
                    elif rank == 0:
                        # We are receiving and writing
                        recv = bufclass(nflat)
                        comm.Recv(recv, source=proc, tag=tag_offset + proc)
                        hdata.write_direct(
                            recv.array().astype(fdtype).reshape(shp),
                            detdata_slice,
                            hf_slice,
                        )
                        recv.clear()
                        del recv
            else:
                # If we have parallel support, every process can write independently.
                samp_off = dist_samps[obs.comm.group_rank].offset
                samp_nelem = dist_samps[obs.comm.group_rank].n_elem
                det_off = dist_dets[obs.comm.group_rank].offset
                det_nelem = dist_dets[obs.comm.group_rank].n_elem

                detdata_slice = [slice(0, det_nelem, 1), slice(0, samp_nelem, 1)]
                hf_slice = [
                    slice(det_off, det_off + det_nelem, 1),
                    slice(samp_off, samp_off + samp_nelem, 1),
                ]
                if dvalshape is not None:
                    detdata_slice.extend([slice(0, x) for x in dvalshape])
                    hf_slice.extend([slice(0, x) for x in dvalshape])
                detdata_slice = tuple(detdata_slice)
                hf_slice = tuple(hf_slice)
                msg = f"Detector data field {field} (group rank {obs.comm.group_rank})"
                check_dataset_buffer_size(msg, hf_slice, ddtype, True)

                with hdata.collective:
                    hdata.write_direct(
                        local_data.data.astype(fdtype), detdata_slice, hf_slice
                    )
            del hdata
            log.verbose_rank(
                f"{log_prefix}  Detdata finished {field} serial write in",
                comm=comm,
                timer=timer,
            )
        else:
            # Compress our local detector data.  The starting dictionary of properties
            # is passed in and additional metadata is appended.
            if ddtype.char == "d" and use_float32:
                temp_detdata = DetectorData(
                    obs.detdata[field].detectors,
                    obs.detdata[field].detector_shape,
                    np.float32,
                    units=obs.detdata[field].units,
                )
                temp_detdata.data[:] = obs.detdata[field].data.astype(np.float32)
                comp_bytes, comp_ranges, comp_props = compress_detdata(
                    temp_detdata, fieldcomp
                )
                del temp_detdata
            else:
                comp_bytes, comp_ranges, comp_props = compress_detdata(
                    obs.detdata[field], fieldcomp
                )

            # Extract per-detector quantities for communicating / writing later
            comp_data_offsets = None
            if "data_offsets" in comp_props:
                comp_data_offsets = comp_props["data_offsets"]
            comp_data_gains = None
            if "data_gains" in comp_props:
                comp_data_gains = comp_props["data_gains"]

            # Get the total number of bytes
            n_local_bytes = len(comp_bytes)
            if comm is None:
                n_all_bytes = n_local_bytes
            else:
                n_all_bytes = comm.allreduce(n_local_bytes, op=MPI.SUM)

            # Create the datasets
            hdata_bytes = None
            hdata_ranges = None
            hdata_offsets = None
            hdata_gains = None
            cgrp = None
            if hgrp is not None:
                # This process is participating.
                cgrp = hgrp.create_group(field)
                hdata_bytes = cgrp.create_dataset(
                    "compressed", n_all_bytes, dtype=np.uint8
                )
                hdata_bytes.attrs["units"] = local_data.units.to_string()
                # Write common properties of many compression schemes
                hdata_bytes.attrs["dtype"] = str(comp_props["dtype"])
                hdata_bytes.attrs["det_shape"] = str(comp_props["det_shape"])
                hdata_bytes.attrs["comp_type"] = comp_props["type"]
                if "level" in comp_props:
                    hdata_bytes.attrs["comp_level"] = comp_props["level"]
                hdata_ranges = cgrp.create_dataset(
                    "ranges",
                    (len(obs.all_detectors), 2),
                    dtype=np.int64,
                )
                if comp_data_offsets is not None:
                    hdata_offsets = cgrp.create_dataset(
                        "offsets",
                        (len(obs.all_detectors),),
                        dtype=np.float64,
                    )
                if comp_data_gains is not None:
                    hdata_gains = cgrp.create_dataset(
                        "gains",
                        (len(obs.all_detectors),),
                        dtype=np.float64,
                    )

            # Send data to rank zero of the group for writing.
            hf_det = 0
            hf_bytes = 0
            for proc in range(nproc):
                if rank == 0:
                    if proc == 0:
                        # Root process writes local data
                        det_ranges = np.array(
                            [(x[0] + hf_bytes, x[1] + hf_bytes) for x in comp_ranges],
                            dtype=np.int64,
                        ).reshape((-1, 2))
                        dslc = (
                            slice(0, len(det_ranges), 1),
                            slice(0, 2, 1),
                        )
                        hslc = (
                            slice(hf_det, hf_det + len(det_ranges), 1),
                            slice(0, 2, 1),
                        )
                        hdata_ranges.write_direct(det_ranges, dslc, hslc)

                        dslc = (slice(0, n_local_bytes, 1),)
                        hslc = (slice(hf_bytes, hf_bytes + n_local_bytes, 1),)
                        hdata_bytes.write_direct(comp_bytes, dslc, hslc)

                        dslc = (slice(0, len(comp_ranges), 1),)
                        hslc = (slice(hf_det, hf_det + len(comp_ranges), 1),)
                        if comp_data_offsets is not None:
                            hdata_offsets.write_direct(comp_data_offsets, dslc, hslc)
                        if comp_data_gains is not None:
                            hdata_gains.write_direct(comp_data_gains, dslc, hslc)

                        hf_bytes += n_local_bytes
                        hf_det += len(comp_ranges)
                    else:
                        # Receive data and write
                        n_recv_bytes = comm.recv(
                            source=proc, tag=tag_offset + 10 * proc
                        )
                        n_recv_dets = comm.recv(
                            source=proc, tag=tag_offset + 10 * proc + 1
                        )

                        recv_bytes = np.zeros(n_recv_bytes, dtype=np.uint8)
                        comm.Recv(
                            recv_bytes, source=proc, tag=tag_offset + 10 * proc + 2
                        )
                        dslc = (slice(0, n_recv_bytes, 1),)
                        hslc = (slice(hf_bytes, hf_bytes + n_recv_bytes, 1),)
                        hdata_bytes.write_direct(recv_bytes, dslc, hslc)
                        del recv_bytes

                        recv_ranges = np.zeros(n_recv_dets * 2, dtype=np.int64)
                        comm.Recv(
                            recv_ranges,
                            source=proc,
                            tag=tag_offset + 10 * proc + 3,
                        )
                        recv_ranges[:] += hf_bytes
                        dslc = (
                            slice(0, n_recv_dets, 1),
                            slice(0, 2, 1),
                        )
                        hslc = (
                            slice(hf_det, hf_det + n_recv_dets, 1),
                            slice(0, 2, 1),
                        )
                        hdata_ranges.write_direct(
                            recv_ranges.reshape((n_recv_dets, 2)), dslc, hslc
                        )
                        del recv_ranges

                        recv_buf = np.zeros(n_recv_dets, dtype=np.float64)
                        dslc = (slice(0, n_recv_dets, 1),)
                        hslc = (slice(hf_det, hf_det + n_recv_dets, 1),)
                        if comp_data_offsets is not None:
                            comm.Recv(
                                recv_buf, source=proc, tag=tag_offset + 10 * proc + 4
                            )
                            hdata_offsets.write_direct(recv_buf, dslc, hslc)
                        if comp_data_gains is not None:
                            comm.Recv(
                                recv_buf, source=proc, tag=tag_offset + 10 * proc + 5
                            )
                            hdata_gains.write_direct(recv_buf, dslc, hslc)
                        del recv_buf

                        hf_bytes += n_recv_bytes
                        hf_det += n_recv_dets

                elif proc == rank:
                    # We are sending.  First send the number of bytes and detectors
                    det_ranges = np.zeros(
                        (len(comp_ranges), 2),
                        dtype=np.int64,
                    )
                    for d in range(len(comp_ranges)):
                        det_ranges[d, :] = comp_ranges[d]
                    comm.send(n_local_bytes, dest=0, tag=tag_offset + 10 * proc)
                    comm.send(len(det_ranges), dest=0, tag=tag_offset + 10 * proc + 1)

                    comm.Send(comp_bytes, dest=0, tag=tag_offset + 10 * proc + 2)
                    comm.Send(
                        det_ranges.flatten(), dest=0, tag=tag_offset + 10 * proc + 3
                    )
                    if comp_data_offsets is not None:
                        comm.Send(
                            comp_data_offsets, dest=0, tag=tag_offset + 10 * proc + 4
                        )
                    if comp_data_gains is not None:
                        comm.Send(
                            comp_data_gains, dest=0, tag=tag_offset + 10 * proc + 5
                        )


@function_timer
def save_hdf5_intervals(obs, hgrp, fields, log_prefix):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # We are using the group communicator
    comm = obs.comm.comm_group
    nproc = obs.comm.group_size
    rank = obs.comm.group_rank

    for field in fields:
        if field not in obs.intervals:
            msg = f"Intervals '{field}' does not exist in observation "
            msg += f"{obs.name}.  Skipping."
            log.warning_rank(msg, comm=comm)
            continue

        if field == obs.intervals.all_name:
            # This is the internal fake interval for all samples.  We don't
            # save this because it is re-created on demand.
            continue

        # Get the list of start / stop tuples on the rank zero process
        ilist = global_interval_times(obs.dist, obs.intervals, field, join=False)

        n_list = None
        if rank == 0:
            n_list = len(ilist)
        if comm is not None:
            n_list = comm.bcast(n_list, root=0)

        # Participating processes create the dataset
        hdata = None
        if hgrp is not None:
            hdata = hgrp.create_dataset(field, (2, n_list), dtype=np.float64)
            # Only the root process writes
            if rank == 0:
                hdata[:, :] = np.transpose(np.array(ilist))
        del hdata

        log.verbose_rank(
            f"{log_prefix}  Intervals finished {field} write in",
            comm=comm,
            timer=timer,
        )


@function_timer
def save_hdf5(
    obs,
    dir,
    meta=None,
    detdata=None,
    shared=None,
    intervals=None,
    config=None,
    times=defaults.times,
    force_serial=False,
    detdata_float32=False,
):
    """Save an observation to HDF5.

    This function writes an observation to a new file in the specified directory.  The
    name is built from the observation name and the observation UID.

    The telescope information is written to a sub-dataset.

    By default, all shared, intervals, and noise models are dumped as individual
    datasets.  A subset of objects may be specified with a list of names passed to
    the corresponding function arguments.

    For detector data, by default, all objects will be dumped uncompressed into
    individual datasets.  If you wish to specify a subset you can provide a list of
    names and only these will be dumped uncompressed.  To enable compression, provide
    a list of tuples, (detector data name, compression properties), where compression
    properties is a dictionary accepted by the `compress_detdata()` function.

    When dumping arbitrary metadata, scalars are stored as attributes of the observation
    "meta" group.  Any objects in the metadata which have a `save_hdf5()` method are
    passed a group and the name of the new dataset to create.  Other objects are
    attempted to be dumped by h5py and a warning is printed if it fails.  The list of
    metadata objects to dump can be given explicitly.

    Args:
        obs (Observation):  The observation to write.
        dir (str):  The parent directory containing the file.
        meta (list):  Only save this list of metadata objects.
        detdata (list):  Only save this list of detdata objects, optionally with
            compression.
        shared (list):  Only save this list of shared objects.
        intervals (list):  Only save this list of intervals objects.
        config (dict):  The job config dictionary to save.
        times (str):  The name of the shared timestamp field.
        force_serial (bool):  If True, do not use HDF5 parallel support,
            even if it is available.
        detdata_float32 (bool):  If True, cast any float64 detector fields
            to float32 on write.  Integer detdata is not affected.

    Returns:
        (str):  The full path of the file that was written.

    """
    log = Logger.get()
    env = Environment.get()
    if obs.comm.group_size == 1:
        # Force serial usage in this case, to avoid any MPI overhead
        force_serial = True

    if obs.name is None:
        raise RuntimeError("Cannot save observations that have no name")

    timer = Timer()
    timer.start()
    log_prefix = f"HDF5 save {obs.name}: "

    comm = obs.comm.comm_group
    rank = obs.comm.group_rank

    namestr = f"{obs.name}_{obs.uid}"
    hfpath = os.path.join(dir, f"{namestr}.h5")
    hfpath_temp = f"{hfpath}.tmp"

    # Create the file and get the root group
    hf = None
    hgroup = None
    vtimer = Timer()
    vtimer.start()

    hf = hdf5_open(hfpath_temp, "w", comm=comm, force_serial=force_serial)
    hgroup = hf

    shared_group = None
    detdata_group = None
    intervals_group = None
    if hgroup is not None:
        # This process is participating
        # Record the software versions and config
        hgroup.attrs["toast_version"] = env.version()
        if config is not None:
            hgroup.attrs["job_config"] = json.dumps(config)
        hgroup.attrs["toast_format_version"] = 1

        # Observation properties
        hgroup.attrs["observation_name"] = obs.name
        hgroup.attrs["observation_uid"] = obs.uid

        obs_all_dets = json.dumps(obs.all_detectors)
        obs_all_det_sets = "NONE"
        if obs.all_detector_sets is not None:
            obs_all_det_sets = json.dumps(obs.all_detector_sets)
        obs_all_samp_sets = "NONE"
        if obs.all_sample_sets is not None:
            obs_all_samp_sets = json.dumps(
                [[str(x) for x in y] for y in obs.all_sample_sets]
            )
        hgroup.attrs["observation_detectors"] = obs_all_dets
        hgroup.attrs["observation_detector_sets"] = obs_all_det_sets
        hgroup.attrs["observation_samples"] = obs.n_all_samples
        hgroup.attrs["observation_sample_sets"] = obs_all_samp_sets

    log.verbose_rank(
        f"{log_prefix}  Wrote observation attributes in",
        comm=comm,
        timer=vtimer,
    )

    inst_group = None
    if hgroup is not None:
        # Instrument properties
        inst_group = hgroup.create_group("instrument")
        inst_group.attrs["telescope_name"] = obs.telescope.name
        inst_group.attrs["telescope_class"] = object_fullname(obs.telescope.__class__)
        inst_group.attrs["telescope_uid"] = obs.telescope.uid
        site = obs.telescope.site
        inst_group.attrs["site_name"] = site.name
        inst_group.attrs["site_class"] = object_fullname(site.__class__)
        inst_group.attrs["site_uid"] = site.uid
        if isinstance(site, GroundSite):
            inst_group.attrs["site_lat_deg"] = site.earthloc.lat.to_value(u.degree)
            inst_group.attrs["site_lon_deg"] = site.earthloc.lon.to_value(u.degree)
            inst_group.attrs["site_alt_m"] = site.earthloc.height.to_value(u.meter)
            if site.weather is not None:
                if hasattr(site.weather, "name"):
                    # This is a simulated weather object, dump it.
                    inst_group.attrs["site_weather_name"] = str(site.weather.name)
                    inst_group.attrs[
                        "site_weather_realization"
                    ] = site.weather.realization
                    if site.weather.max_pwv is None:
                        inst_group.attrs["site_weather_max_pwv"] = "NONE"
                    else:
                        inst_group.attrs[
                            "site_weather_max_pwv"
                        ] = site.weather.max_pwv.to_value(u.mm)
                    inst_group.attrs[
                        "site_weather_time"
                    ] = site.weather.time.timestamp()
                    inst_group.attrs[
                        "site_weather_median"
                    ] = site.weather.median_weather
                else:
                    msg = "HDF5 saving currently only supports SimWeather instances"
                    raise NotImplementedError(msg)
        session = obs.session
        if session is not None:
            inst_group.attrs["session_name"] = session.name
            inst_group.attrs["session_class"] = object_fullname(session.__class__)
            inst_group.attrs["session_uid"] = session.uid
            if session.start is None:
                inst_group.attrs["session_start"] = "NONE"
            else:
                inst_group.attrs["session_start"] = session.start.timestamp()
            if session.end is None:
                inst_group.attrs["session_end"] = "NONE"
            else:
                inst_group.attrs["session_end"] = session.end.timestamp()
    log.verbose_rank(
        f"{log_prefix}  Wrote instrument attributes in",
        comm=comm,
        timer=vtimer,
    )

    obs.telescope.focalplane.save_hdf5(inst_group, comm=comm)
    del inst_group

    log.verbose_rank(
        f"{log_prefix}  Wrote focalplane in",
        comm=comm,
        timer=vtimer,
    )

    log.debug_rank(
        f"{log_prefix} Finished instrument model",
        comm=comm,
        timer=timer,
    )

    meta_group = None
    if hgroup is not None:
        meta_group = hgroup.create_group("metadata")

    for k, v in obs.items():
        if meta is not None and k not in meta:
            continue
        if hasattr(v, "save_hdf5"):
            kgroup = None
            if meta_group is not None:
                kgroup = meta_group.create_group(k)
                kgroup.attrs["class"] = object_fullname(v.__class__)
            v.save_hdf5(kgroup, obs)
            del kgroup
        elif isinstance(v, u.Quantity):
            if isinstance(v.value, np.ndarray):
                # Array quantity
                if meta_group is not None:
                    qdata = meta_group.create_dataset(k, data=v.value)
                    qdata.attrs["units"] = v.unit.to_string()
                    del qdata
            else:
                # Must be a scalar
                if meta_group is not None:
                    meta_group.attrs[f"{k}"] = v.value
                    meta_group.attrs[f"{k}_units"] = v.unit.to_string()
        elif isinstance(v, np.ndarray):
            if meta_group is not None:
                marr = meta_group.create_dataset(k, data=v)
                del marr
        elif meta_group is not None:
            try:
                if isinstance(v, u.Quantity):
                    meta_group.attrs[k] = v.value
                else:
                    meta_group.attrs[k] = v
            except ValueError as e:
                msg = f"Failed to store obs key '{k}' = '{v}' as an attribute ({e})."
                msg += f" Try casting it to a supported type when storing in the "
                msg += f"observation dictionary or implement save_hdf5() and "
                msg += f"load_hdf5() methods."
                log.verbose(msg)
    del meta_group

    log.verbose_rank(
        f"{log_prefix}  Wrote other metadata in",
        comm=comm,
        timer=vtimer,
    )

    log.debug_rank(
        f"{log_prefix} Finished metadata",
        comm=comm,
        timer=timer,
    )

    # Dump data

    if shared is None:
        fields = list(obs.shared.keys())
    else:
        fields = list(shared)

    dump_intervals = True
    if times not in obs.shared:
        msg = f"Timestamp field '{times}' does not exist.  Not saving intervals."
        log.warning_rank(msg, comm=comm)
        dump_intervals = False
    else:
        if times not in fields:
            fields.append(times)

    shared_group = None
    if hgroup is not None:
        shared_group = hgroup.create_group("shared")
    save_hdf5_shared(obs, shared_group, fields, log_prefix)
    del shared_group

    log.debug_rank(
        f"{log_prefix} Finished shared data",
        comm=comm,
        timer=timer,
    )

    if detdata is None:
        fields = [(x, None) for x in obs.detdata.keys()]
    else:
        fields = list()
        for df in detdata:
            if isinstance(df, (tuple, list)):
                fields.append(df)
            else:
                fields.append((df, None))
    detdata_group = None
    if hgroup is not None:
        detdata_group = hgroup.create_group("detdata")
    save_hdf5_detdata(
        obs, detdata_group, fields, log_prefix, use_float32=detdata_float32
    )
    del detdata_group
    log.debug_rank(
        f"{log_prefix} Finished detector data",
        comm=comm,
        timer=timer,
    )

    if intervals is None:
        fields = list(obs.intervals.keys())
    else:
        fields = list(intervals)
    if dump_intervals:
        intervals_group = None
        if hgroup is not None:
            intervals_group = hgroup.create_group("intervals")
            intervals_group.attrs["times"] = times
        save_hdf5_intervals(obs, intervals_group, fields, log_prefix)
        del intervals_group
    log.debug_rank(
        f"{log_prefix} Finished intervals data",
        comm=comm,
        timer=timer,
    )

    # Close file if we opened it
    del hgroup
    if hf is not None:
        hf.close()
    del hf

    if comm is not None:
        comm.barrier()

    # Move file into place
    if rank == 0:
        os.rename(hfpath_temp, hfpath)

    return hfpath
