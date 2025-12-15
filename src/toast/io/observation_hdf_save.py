# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import json
import os
from datetime import timezone

import h5py
import numpy as np
from astropy import units as u

import flacarray

from ..instrument import GroundSite
from ..observation import default_values as defaults
from ..observation_dist import global_interval_times
from ..timing import Timer, function_timer
from ..utils import (
    Environment,
    Logger,
    dtype_to_aligned,
    hdf5_use_serial,
    object_fullname,
)
from .hdf_utils import check_dataset_buffer_size, hdf5_open, save_meta_object


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
def save_hdf5_detdata(obs, hgrp, fields, log_prefix, in_place=False):
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

    # Valid flacarray dtypes
    flac_dtypes = [
        np.dtype(np.float64),
        np.dtype(np.float32),
        np.dtype(np.int64),
        np.dtype(np.int32),
    ]

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
            msg += "individual channels is split between processes."
            raise RuntimeError(msg)

        # Compute properties of the full set of data across the observation
        ddtype = local_data.dtype
        if ddtype in flac_dtypes:
            hdtype = ddtype
        else:
            # Cast flag bytes to int for compression
            hdtype = np.int32
        dshape = (len(obs.all_detectors), obs.n_all_samples)
        dvalshape = None
        if len(local_data.detector_shape) > 1:
            dvalshape = local_data.detector_shape[1:]
            dshape += dvalshape
        local_n_det = local_data.shape[0]

        if fieldcomp is None:
            # We are not using compression.

            # The buffer class to use for allocating receive buffers
            bufclass, _ = dtype_to_aligned(ddtype)

            # Group handle
            hdata = None
            if hgrp is not None:
                # This process is participating.
                hdata = hgrp.create_dataset(field, dshape, dtype=ddtype)
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
                                local_data.data.astype(ddtype), detdata_slice, hf_slice
                            )
                    elif proc == rank:
                        # We are sending
                        comm.Send(local_data.flatdata, dest=0, tag=tag_offset + proc)
                    elif rank == 0:
                        # We are receiving and writing
                        recv = bufclass(nflat)
                        comm.Recv(recv, source=proc, tag=tag_offset + proc)
                        hdata.write_direct(
                            recv.array().astype(ddtype).reshape(shp),
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
                        local_data.data.astype(ddtype), detdata_slice, hf_slice
                    )
            del hdata
            log.verbose_rank(
                f"{log_prefix}  Detdata finished {field} serial write in",
                comm=comm,
                timer=timer,
            )
        else:
            fgrp = None
            if hgrp is not None:
                # This process is participating.  Create a subgroup for this
                # field.
                fgrp = hgrp.create_group(field)
                # Add attributes for the original data properties
                fgrp.attrs["units"] = local_data.units.to_string()
                fgrp.attrs["dtype"] = ddtype.char
                fgrp.attrs["detector_shape"] = str(
                    [int(x) for x in local_data.detector_shape]
                )
            if "level" in fieldcomp:
                level = int(fieldcomp["level"])
            else:
                level = 5
            quanta = None
            precision = None

            if ddtype.char == "d" or ddtype.char == "f":
                # Floating point type
                if "quanta" in fieldcomp:
                    quanta = float(fieldcomp["quanta"])
                elif "precision" in fieldcomp:
                    precision = float(fieldcomp["precision"])
                if quanta is None and precision is None:
                    msg = "When compressing floating point data, you"
                    msg += " must specify the quanta or precision."
                    raise RuntimeError("You must specify the quanta")

            # We flatten all the per-sample data when compressing
            flacarray.hdf5.write_array(
                local_data.data.astype(hdtype).reshape((local_n_det, -1)),
                fgrp,
                level=level,
                quanta=quanta,
                precision=precision,
                mpi_comm=comm,
                use_threads=False,
            )
            if in_place:
                # Decompress data back into original location, to capture
                # any truncation effects.
                det_off = dist_dets[obs.comm.group_rank].offset
                det_nelem = dist_dets[obs.comm.group_rank].n_elem
                mpi_dist = [(x.offset, x.offset + x.n_elem) for x in dist_dets]

                flcdata = (
                    flacarray.hdf5.read_array(
                        fgrp,
                        keep=None,
                        stream_slice=None,
                        keep_indices=False,
                        mpi_comm=comm,
                        mpi_dist=mpi_dist,
                        use_threads=False,
                    )
                    .astype(ddtype)
                    .reshape(local_data.shape)
                )
                for idet, det in enumerate(local_data.detectors):
                    local_data.data[idet] = flcdata[idet]
                del flcdata


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


def save_instrument(parent_group, telescope, comm=None, session=None):
    """Save instrument information to an HDF5 group.

    Given the parent group (which might exist on multiple processes in the case of
    MPI use), create an instrument sub group and write telescope and optionally
    session information to that group.

    """
    inst_group = None
    if parent_group is not None:
        # Instrument properties
        inst_group = parent_group.create_group("instrument")
        inst_group.attrs["toast_format_version"] = 2

    telescope.save_hdf5(inst_group, comm=comm)
    if session is not None:
        session.save_hdf5(inst_group, comm=comm)
    del inst_group


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
    detdata_in_place=False,
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
        detdata_in_place (bool):  If True, input detdata will be replaced
            with a compressed and decompressed version that includes the
            digitization error.


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
    hfpath = os.path.join(dir, f"obs_{namestr}.h5")
    hfpath_temp = f"{hfpath}.tmp"

    # Create the file and get the root group
    hf = None
    hgroup = None
    vtimer = Timer()
    vtimer.start()

    hf = hdf5_open(hfpath_temp, "w", comm=comm, force_serial=force_serial)
    hgroup = hf

    # Gather the local detector flags to all writing processes
    if comm is None:
        all_det_flags = obs.local_detector_flags
    else:
        proc_det_flags = comm.gather(obs.local_detector_flags, root=0)
        all_det_flags = None
        if rank == 0:
            all_det_flags = dict()
            for pflags in proc_det_flags:
                all_det_flags.update(pflags)
        all_det_flags = comm.bcast(all_det_flags, root=0)

    shared_group = None
    detdata_group = None
    intervals_group = None
    if hgroup is not None:
        # This process is participating
        # Record the software versions and config
        hgroup.attrs["toast_version"] = env.version()
        if config is not None:
            hgroup.attrs["job_config"] = json.dumps(config)
        hgroup.attrs["toast_format_version"] = 2

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

        # Per detector flags.
        hgroup.attrs["observation_detector_flags"] = json.dumps(all_det_flags)

    log.verbose_rank(
        f"{log_prefix}  Wrote observation attributes in",
        comm=comm,
        timer=vtimer,
    )

    save_instrument(hgroup, obs.telescope, comm=comm, session=obs.session)

    log.verbose_rank(
        f"{log_prefix}  Wrote instrument in",
        comm=comm,
        timer=vtimer,
    )

    log.debug_rank(
        f"{log_prefix} Finished instrument model",
        comm=comm,
        timer=timer,
    )

    meta_group = None
    meta_other = None
    if hgroup is not None:
        meta_group = hgroup.create_group("metadata")
        meta_other = meta_group.create_group("other")

    # Process all metadata in the observation dictionary
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
        else:
            # Process this object recursively
            save_meta_object(meta_other, k, v)
    del meta_other
    del meta_group

    # Now pass through observation attributes and look for things to save
    attr_group = None
    if hgroup is not None:
        attr_group = hgroup.create_group("attr")
    for k, v in vars(obs).items():
        if k.startswith("_"):
            continue
        if hasattr(v, "save_hdf5"):
            kgroup = None
            if attr_group is not None:
                kgroup = attr_group.create_group(k)
                kgroup.attrs["class"] = object_fullname(v.__class__)
            v.save_hdf5(kgroup, obs)
            del kgroup

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
        # We are writing all detector data without, compression
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
        obs,
        detdata_group,
        fields,
        log_prefix,
        in_place=detdata_in_place,
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


def save_instrument_file(path, telescope, session):
    """Save instrument data to an HDF5 group.

    This function loads the telescope and session serially on one process.
    It supports including a relative internal path inside the HDF5 file by separating
    the filesystem path from the internal path with a colon.  For example:

    path="/path/to/file.h5:/obs1

    The internal path should be to the *parent* group of the "instrument" group.

    """
    parts = path.split(":")
    if len(parts) == 1:
        file = parts[0]
        internal = "/"
    else:
        file = parts[0]
        internal = parts[1]
    grouptree = internal.split(os.path.sep)
    with h5py.File(file, "w") as hf:
        parent = hf
        for grp in grouptree:
            if grp == "":
                continue
            parent = parent.create_group(grp)
        save_instrument(parent, telescope, session=session)
