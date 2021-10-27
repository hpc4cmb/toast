# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np

from astropy import units as u

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


@function_timer
def save_hdf5_shared(obs, hgrp, fields):
    log = Logger.get()
    parallel = have_hdf5_parallel()

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
        sbclass, _ = dtype_to_aligned(sdtype)

        # Participating processes create the dataset

        hdata = None
        if parallel or obs.comm.group_rank == 0:
            hdata = hgrp.create_dataset(field, sshape, dtype=sdtype)
            hdata.attrs["comm_type"] = scomm

        # If we have parallel support, the rank zero of each comm can write
        # independently.  Otherwise, send data to rank zero of the group
        # for writing.

        if scomm == "group":
            # Easy...
            if obs.comm.group_rank == 0:
                hdata[:] = sdata.data
        elif scomm == "column":
            if parallel:
                # Rank zero of each column writes
                if sdata.comm is None or sdata.comm.rank == 0:
                    hdata[
                        obs.local_index_offset : obs.local_index_offset
                        + obs.n_local_samples
                    ] = sdata.data
            else:
                # Send data to root process
                for proc in range(proc_cols):
                    # Process grid is indexed row-major, so the rank-zero process
                    # of each column is just the first row of the grid.
                    group_rank = proc
                    # Leading data range for this process
                    off = dist_samps[group_rank].offset
                    nelem = dist_samps[group_rank].n_elem
                    nflat = nelem * np.prod(sshape[1:])
                    shp = (nelem,) + sshape[1:]
                    if group_rank == 0:
                        # Root process writes local data
                        if obs.comm.group_rank == 0:
                            hdata[off : off + nelem] = sdata.data
                    elif group_rank == obs.comm.group_rank:
                        # We are sending
                        obs.comm.comm_group.Send(
                            sdata.data.flatten(), dest=0, tag=group_rank
                        )
                    elif obs.comm.group_rank == 0:
                        # We are receiving and writing
                        recv = sbclass(nflat)
                        obs.comm.comm_group.Recv(
                            recv, source=group_rank, tag=group_rank
                        )
                        hdata[off : off + nelem] = recv.array().reshape(shp)
                        recv.clear()
                        del recv
                    if obs.comm.comm_group is not None:
                        obs.comm.comm_group.barrier()
        else:
            if parallel:
                # Rank zero of each row writes
                if sdata.comm is None or sdata.comm.rank == 0:
                    off = dist_dets[obs.comm.group_rank].offset
                    nelem = dist_dets[obs.comm.group_rank].n_elem
                    hdata[off : off + nelem] = sdata.data
            else:
                # Send data to root process
                for proc in range(proc_rows):
                    # Process grid is indexed row-major, so the rank-zero process
                    # of each row is strided by the number of columns.
                    group_rank = proc * proc_cols
                    # Leading data range for this process
                    off = dist_dets[group_rank].offset
                    nelem = dist_dets[group_rank].n_elem
                    nflat = nelem * np.prod(sshape[1:])
                    shp = (nelem,) + sshape[1:]
                    if group_rank == 0:
                        # Root process writes local data
                        if obs.comm.group_rank == 0:
                            hdata[off : off + nelem] = sdata.data
                    elif group_rank == obs.comm.group_rank:
                        # We are sending
                        obs.comm.comm_group.Send(
                            sdata.data.flatten(), dest=0, tag=group_rank
                        )
                    elif obs.comm.group_rank == 0:
                        # We are receiving and writing
                        recv = sbclass(nflat)
                        obs.comm.comm_group.Recv(
                            recv, source=group_rank, tag=group_rank
                        )
                        hdata[off : off + nelem] = recv.array().reshape(shp)
                        recv.clear()
                        del recv
                    if obs.comm.comm_group is not None:
                        obs.comm.comm_group.barrier()


@function_timer
def save_hdf5_detdata(obs, hgrp, fields):
    pass


@function_timer
def save_hdf5_intervals(obs, hgrp, fields):
    pass


@function_timer
def save_hdf5(
    obs,
    dir,
    meta=None,
    detdata=None,
    shared=None,
    intervals=None,
    config=None,
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

    Returns:
        None

    """
    log = Logger.get()
    env = Environment.get()
    parallel = have_hdf5_parallel()

    if obs.name is None:
        raise RuntimeError("Cannot save observations that have no name")

    namestr = f"{obs.name}_{obs.uid}"
    hfpath = os.path.join(dir, f"{namestr}.h5")
    hfpath_temp = f"{hfpath}.tmp"

    # Create the file and get the root group
    hf = None
    hfgroup = None
    if parallel:
        hf = h5py.File(hfpath_temp, "w", driver="mpio", comm=obs.comm.comm_group)
        hgroup = hf
    elif obs.comm.group_rank == 0:
        hf = h5py.File(hfpath_temp, "w")
        hgroup = hf

    shared_group = None
    detdata_group = None
    intervals_group = None
    if parallel or obs.comm.group_rank == 0:
        # Record the software versions and config
        hgroup.attrs["toast_version"] = env.version()
        if config is not None:
            hgroup.attrs["job_config"] = json.dumps(config)

        # Observation properties
        hgroup.attrs["observation_name"] = obs.name
        hgroup.attrs["observation_uid"] = obs.uid
        # hgroup.attrs["observation_detector_sets"] = obs.all_detector_sets

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
        obs.telescope.focalplane.save_hdf5(inst_group, comm=obs.comm.comm_group)

        meta_group = hgroup.create_group("metadata")
        for k, v in obs.items():
            if hasattr(v, "save_hdf5"):
                kgroup = meta_group.create_group(k)
                v.save_hdf5(kgroup, comm=obs.comm.comm_group)
            else:
                try:
                    meta_group.attrs[k] = v
                except ValueError as e:
                    msg = f"Failed to store obs key '{k}' = '{v}' as an attribute ({e})"
                    log.verbose_rank(msg, comm=obs.comm.comm_group)

        shared_group = hgroup.create_group("shared")
        detdata_group = hgroup.create_group("detdata")
        intervals_group = hgroup.create_group("intervals")

    # Dump data

    fields = shared
    if fields is None:
        fields = obs.shared.keys()
    save_hdf5_shared(obs, shared_group, fields)

    fields = detdata
    if fields is None:
        fields = obs.detdata.keys()
    save_hdf5_detdata(obs, detdata_group, fields)

    fields = intervals
    if fields is None:
        fields = obs.intervals.keys()
    save_hdf5_intervals(obs, intervals_group, fields)

    # Close file if we opened it

    if hf is not None:
        hf.flush()
        hf.close()

    if obs.comm.comm_group is not None:
        obs.comm.comm_group.barrier()

    # Move file into place
    if obs.comm.group_rank == 0:
        if hfpath is not None:
            os.rename(hfpath_temp, hfpath)

    return
