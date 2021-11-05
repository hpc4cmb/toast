# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np

from astropy import units as u
from astropy.table import Table

import h5py

import json

from ..utils import (
    Environment,
    Logger,
    have_hdf5_parallel,
    object_fullname,
    dtype_to_aligned,
)

from ..mpi import MPI

from ..timing import Timer, function_timer, GlobalTimers

from ..instrument import GroundSite

from ..observation import default_values as defaults

from ..observation_dist import global_interval_times


@function_timer
def save_hdf5_shared_parallel(obs, hgrp, fields, log_prefix):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # Get references to the distribution of detectors and samples
    proc_rows = obs.dist.process_rows
    proc_cols = obs.dist.comm.group_size // proc_rows
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    for field in fields:
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

        # All processes create the dataset
        hdata = hgrp.create_dataset(field, sshape, dtype=sdtype)
        hdata.attrs["comm_type"] = scomm

        # If we have parallel support, the rank zero of each comm can write
        # independently.

        if scomm == "group":
            # Easy...
            if obs.comm.group_rank == 0:
                hdata[:] = sdata.data
        elif scomm == "column":
            # Rank zero of each column writes
            if sdata.comm is None or sdata.comm.rank == 0:
                hdata[
                    obs.local_index_offset : obs.local_index_offset
                    + obs.n_local_samples
                ] = sdata.data
        else:
            # Rank zero of each row writes
            if sdata.comm is None or sdata.comm.rank == 0:
                off = dist_dets[obs.comm.group_rank].offset
                nelem = dist_dets[obs.comm.group_rank].n_elem
                hdata[off : off + nelem] = sdata.data

        log.verbose_rank(
            f"{log_prefix}  Shared finished {field} parallel write in",
            comm=obs.comm.comm_group,
            timer=timer,
        )
        del hdata


@function_timer
def save_hdf5_shared_serial(obs, hgrp, fields, log_prefix):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # Get references to the distribution of detectors and samples
    proc_rows = obs.dist.process_rows
    proc_cols = obs.dist.comm.group_size // proc_rows
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    n_fields = len(fields)

    for ifield, field in enumerate(fields):
        tag_offset = ifield * obs.comm.group_size

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

        # Rank zero of the group creates the dataset
        hdata = None
        if obs.comm.group_rank == 0:
            hdata = hgrp.create_dataset(field, sshape, dtype=sdtype)
            hdata.attrs["comm_type"] = scomm

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

        log.verbose_rank(
            f"{log_prefix}  Shared finished {field} serial write in",
            comm=obs.comm.comm_group,
            timer=timer,
        )
        del hdata


@function_timer
def save_hdf5_detdata_parallel(obs, hgrp, fields, log_prefix):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # Get references to the distribution of detectors and samples
    proc_rows = obs.dist.process_rows
    proc_cols = obs.dist.comm.group_size // proc_rows
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    for field in fields:
        if field not in obs.detdata:
            msg = f"Detdata data '{field}' does not exist in observation "
            msg += f"{obs.name}.  Skipping."
            log.warning_rank(msg, comm=obs.comm.comm_group)
            continue

        local_data = obs.detdata[field]
        if local_data.detectors != obs.local_detectors:
            msg = f"Data data '{field}' does not contain all local detectors."
            log.error(msg)
            raise RuntimeError(msg)

        # Compute properties of the full set of data across the observation

        ddtype = local_data.dtype
        dshape = (len(obs.all_detectors), obs.n_all_samples)
        dvalshape = None
        if len(local_data.detector_shape) > 1:
            dvalshape = local_data.detector_shape[1:]
            dshape += dvalshape

        # The buffer class to use for allocating receive buffers
        bufclass, _ = dtype_to_aligned(ddtype)

        # All processes create the dataset
        hdata = hgrp.create_dataset(field, dshape, dtype=ddtype)
        hdata.attrs["units"] = local_data.units.to_string()

        # If we have parallel support, every process can write independently.
        samp_off = dist_samps[obs.comm.group_rank].offset
        samp_nelem = dist_samps[obs.comm.group_rank].n_elem
        det_off = dist_dets[obs.comm.group_rank].offset
        det_nelem = dist_dets[obs.comm.group_rank].n_elem
        with hdata.collective:
            hdata[
                det_off : det_off + det_nelem, samp_off : samp_off + samp_nelem
            ] = local_data

        del hdata
        log.verbose_rank(
            f"{log_prefix}  Detdata finished {field} parallel write in",
            comm=obs.comm.comm_group,
            timer=timer,
        )


@function_timer
def save_hdf5_detdata_serial(obs, hgrp, fields, log_prefix):
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

    n_fields = len(fields)

    for ifield, field in enumerate(fields):
        tag_offset = ifield * obs.comm.group_size
        if field not in obs.detdata:
            msg = f"Detdata data '{field}' does not exist in observation "
            msg += f"{obs.name}.  Skipping."
            log.warning_rank(msg, comm=comm)
            continue

        local_data = obs.detdata[field]
        if local_data.detectors != obs.local_detectors:
            msg = f"Data data '{field}' does not contain all local detectors."
            log.error(msg)
            raise RuntimeError(msg)

        # Compute properties of the full set of data across the observation

        ddtype = local_data.dtype
        dshape = (len(obs.all_detectors), obs.n_all_samples)
        dvalshape = None
        if len(local_data.detector_shape) > 1:
            dvalshape = local_data.detector_shape[1:]
            dshape += dvalshape

        # The buffer class to use for allocating receive buffers
        bufclass, _ = dtype_to_aligned(ddtype)

        # Only the root process creates the dataset
        hdata = None
        if rank == 0:
            hdata = hgrp.create_dataset(field, dshape, dtype=ddtype)
            hdata.attrs["units"] = local_data.units.to_string()

        # Send data to rank zero of the group for writing.

        for proc in range(nproc):
            # Data ranges for this process
            samp_off = dist_samps[proc].offset
            samp_nelem = dist_samps[proc].n_elem
            det_off = dist_dets[proc].offset
            det_nelem = dist_dets[proc].n_elem
            nflat = det_nelem * samp_nelem
            shp = (det_nelem, samp_nelem)
            if dvalshape is not None:
                nflat *= dvalshape
                shp += dvalshape
            if proc == 0:
                # Root process writes local data
                if rank == 0:
                    hdata[
                        det_off : det_off + det_nelem,
                        samp_off : samp_off + samp_nelem,
                    ] = local_data
            elif proc == rank:
                # We are sending
                comm.Send(local_data.flatdata, dest=0, tag=tag_offset + proc)
            elif rank == 0:
                # We are receiving and writing
                recv = bufclass(nflat)
                comm.Recv(recv, source=proc, tag=tag_offset + proc)
                hdata[
                    det_off : det_off + det_nelem, samp_off : samp_off + samp_nelem
                ] = recv.array().reshape(shp)
                recv.clear()
                del recv
        log.verbose_rank(
            f"{log_prefix}  Detdata finished {field} serial write in",
            comm=comm,
            timer=timer,
        )
        del hdata


@function_timer
def save_hdf5_intervals_parallel(obs, hgrp, fields, log_prefix):
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

        # Get the list of start / stop tuples on the rank zero process
        ilist = global_interval_times(obs.dist, obs.intervals, field, join=False)

        n_list = None
        if rank == 0:
            n_list = len(ilist)
        if comm is not None:
            n_list = comm.bcast(n_list, root=0)

        # All processes create the dataset
        hdata = hgrp.create_dataset(field, (2, n_list), dtype=np.float64)

        # Only the root process writes
        if rank == 0:
            hdata[:, :] = np.transpose(np.array(ilist))

        # Free object
        del hdata

        log.verbose_rank(
            f"{log_prefix}  Intervals finished {field} parallel write in",
            comm=comm,
            timer=timer,
        )


@function_timer
def save_hdf5_intervals_serial(obs, hgrp, fields, log_prefix):
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

        # Get the list of start / stop tuples on the rank zero process
        ilist = global_interval_times(obs.dist, obs.intervals, field, join=False)

        # Root process creates the dataset and writes
        if rank == 0:
            hdata = hgrp.create_dataset(field, (2, len(ilist)), dtype=np.float64)
            hdata[:, :] = np.transpose(np.array(ilist))
            del hdata

        log.verbose_rank(
            f"{log_prefix}  Intervals finished {field} serial write in",
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
):
    """Save an observation to HDF5.

    This function writes an observation to a new file in the specified directory.  The
    name is built from the observation name and the observation UID.

    The telescope information is written to a sub-dataset.

    By default, all detdata, shared, intervals, and noise models are dumped as
    individual datasets.  A subset of objects may be specified with a list of names
    passed to the corresponding function arguments.

    When dumping arbitrary metadata, scalars are stored as attributes of the observation
    "meta" group.  Any objects in the metadata which have a `save_hdf5()` method are
    passed a group and the name of the new dataset to create.  Other objects are
    attempted to be dumped by h5py and a warning is printed if it fails.  The list of
    metadata objects to dump can be given explicitly.

    Args:
        obs (Observation):  The observation to write.
        dir (str):  The parent directory containing the file.
        meta (list):  Only save this list of metadata objects.
        detdata (list):  Only save this list of detdata objects.
        shared (list):  Only save this list of shared objects.
        intervals (list):  Only save this list of intervals objects.
        config (dict):  The job config dictionary to save.
        times (str):  The name of the shared timestamp field.
        force_serial (bool):  If True, do not use HDF5 parallel support,
            even if it is available.

    Returns:
        (str):  The full path of the file that was written.

    """
    log = Logger.get()
    env = Environment.get()
    parallel = have_hdf5_parallel()
    if force_serial:
        parallel = False

    if obs.name is None:
        raise RuntimeError("Cannot save observations that have no name")

    timer = Timer()
    timer.start()
    log_prefix = f"HDF5 save {obs.name}: "

    namestr = f"{obs.name}_{obs.uid}"
    hfpath = os.path.join(dir, f"{namestr}.h5")
    hfpath_temp = f"{hfpath}.tmp"

    # Create the file and get the root group
    hf = None
    hgroup = None
    vtimer = Timer()
    vtimer.start()
    if parallel:
        hf = h5py.File(hfpath_temp, "w", driver="mpio", comm=obs.comm.comm_group)
        hgroup = hf
        log.verbose_rank(
            f"{log_prefix}  Opened temp file {hfpath_temp} in parallel in",
            comm=obs.comm.comm_group,
            timer=vtimer,
        )
    elif obs.comm.group_rank == 0:
        hf = h5py.File(hfpath_temp, "w")
        hgroup = hf
        log.verbose_rank(
            f"{log_prefix}  Opened temp file {hfpath_temp} on rank zero in",
            comm=None,
            timer=vtimer,
        )

    save_comm = None
    if parallel:
        save_comm = obs.comm.comm_group

    shared_group = None
    detdata_group = None
    intervals_group = None
    if parallel or obs.comm.group_rank == 0:
        # Record the software versions and config
        hgroup.attrs["toast_version"] = env.version()
        if config is not None:
            hgroup.attrs["job_config"] = json.dumps(config)
        hgroup.attrs["toast_format_version"] = 0

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
            f"{log_prefix}  Wrote observation attributes (parallel={parallel}) in",
            comm=save_comm,
            timer=vtimer,
        )

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
                    inst_group.attrs["site_weather_name"] = site.weather.name
                    inst_group.attrs[
                        "site_weather_realization"
                    ] = site.weather.realization
                    if site.weather.max_pwv is None:
                        inst_group.attrs["site_weather_max_pwv"] = "NONE"
                    else:
                        inst_group.attrs["site_weather_max_pwv"] = site.weather.max_pwv
                    inst_group.attrs[
                        "site_weather_time"
                    ] = site.weather.time.timestamp()
        log.verbose_rank(
            f"{log_prefix}  Wrote instrument attributes (parallel={parallel}) in",
            comm=save_comm,
            timer=vtimer,
        )
        obs.telescope.focalplane.save_hdf5(
            inst_group, comm=obs.comm.comm_group, force_serial=force_serial
        )
        log.verbose_rank(
            f"{log_prefix}  Wrote focalplane (parallel={parallel}) in",
            comm=save_comm,
            timer=vtimer,
        )
        del inst_group

        log.debug_rank(
            f"{log_prefix} finished instrument model",
            comm=save_comm,
            timer=timer,
        )

        meta_group = hgroup.create_group("metadata")

        for k, v in obs.items():
            if meta is not None and k not in meta:
                continue
            if hasattr(v, "save_hdf5"):
                kgroup = meta_group.create_group(k)
                kgroup.attrs["class"] = object_fullname(v.__class__)
                v.save_hdf5(kgroup, comm=save_comm, force_serial=force_serial)
                del kgroup
            elif isinstance(v, u.Quantity):
                if isinstance(v.value, np.ndarray):
                    # Array quantity
                    qdata = meta_group.create_dataset(k, data=v.value)
                    qdata.attrs["units"] = v.unit.to_string()
                    del qdata
                else:
                    # Must be a scalar
                    meta_group.attrs[f"{k}"] = v.value
                    meta_group.attrs[f"{k}_units"] = v.unit.to_string()
            elif isinstance(v, np.ndarray):
                marr = meta_group.create_dataset(k, data=v)
                del marr
            else:
                try:
                    if isinstance(v, u.Quantity):
                        meta_group.attrs[k] = v.value
                    else:
                        meta_group.attrs[k] = v
                except ValueError as e:
                    msg = f"Failed to store obs key '{k}' = '{v}' as an attribute ({e})"
                    log.verbose_rank(msg, comm=save_comm)
            log.verbose_rank(
                f"{log_prefix}  Meta finished {k} (parallel={parallel}) in",
                comm=save_comm,
                timer=vtimer,
            )

        del meta_group

        log.debug_rank(
            f"{log_prefix} Finished metadata",
            comm=save_comm,
            timer=timer,
        )

        shared_group = hgroup.create_group("shared")
        detdata_group = hgroup.create_group("detdata")
        intervals_group = hgroup.create_group("intervals")
        intervals_group.attrs["times"] = times

    # Dump data

    if shared is None:
        fields = list(obs.shared.keys())
    else:
        fields = list(shared)

    dump_intervals = True
    if times not in obs.shared:
        msg = f"Timestamp field '{times}' does not exist.  Not saving intervals."
        log.warning_rank(msg, comm=obs.comm.comm_group)
        dump_intervals = False
    else:
        if times not in fields:
            fields.append(times)
    if parallel:
        save_hdf5_shared_parallel(obs, shared_group, fields, log_prefix)
    else:
        save_hdf5_shared_serial(obs, shared_group, fields, log_prefix)

    log.debug_rank(
        f"{log_prefix} finished shared data",
        comm=obs.comm.comm_group,
        timer=timer,
    )

    if detdata is None:
        fields = list(obs.detdata.keys())
    else:
        fields = list(detdata)
    if parallel:
        save_hdf5_detdata_parallel(obs, detdata_group, fields, log_prefix)
    else:
        save_hdf5_detdata_serial(obs, detdata_group, fields, log_prefix)

    log.debug_rank(
        f"{log_prefix} finished detector data",
        comm=obs.comm.comm_group,
        timer=timer,
    )

    if intervals is None:
        fields = list(obs.intervals.keys())
    else:
        fields = list(intervals)
    if dump_intervals:
        if parallel:
            save_hdf5_intervals_parallel(obs, intervals_group, fields, log_prefix)
        else:
            save_hdf5_intervals_serial(obs, intervals_group, fields, log_prefix)

    log.debug_rank(
        f"{log_prefix} finished intervals data",
        comm=obs.comm.comm_group,
        timer=timer,
    )

    # Close file if we opened it

    del shared_group
    del detdata_group
    del intervals_group
    del hgroup

    if hf is not None:
        hf.flush()
        hf.close()
    del hf

    if obs.comm.comm_group is not None:
        obs.comm.comm_group.barrier()

    # Move file into place
    if obs.comm.group_rank == 0:
        if hfpath is not None:
            os.rename(hfpath_temp, hfpath)

    return hfpath
