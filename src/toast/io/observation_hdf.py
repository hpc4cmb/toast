# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np

from astropy import units as u

import h5py

import json

from ..utils import Environment, Logger, have_hdf5_parallel, object_fullname

from ..mpi import MPI

from ..timing import Timer, function_timer, GlobalTimers

from ..instrument import GroundSite


@function_timer
def save_hdf5(
    obs,
    meta=None,
    detdata=None,
    shared=None,
    intervals=None,
    dir=None,
    root=None,
    config=None,
):
    """Save an observation to HDF5.

    This function writes an observation either to a new file in the specified directory,
    or as a new HDF5 child group under the specified root.  In either case, the name is
    built from the observation name and the observation UID.

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

    if dir is None and root is None:
        raise ValueError(
            "You must specify either the parent directory or parent HDF5 group"
        )

    if dir is not None and root is not None:
        raise ValueError(
            "Cannot specify both the parent directory and the parent HDF5 group"
        )

    if obs.name is None:
        raise RuntimeError("Cannot save observations that have no name")

    namestr = f"{obs.name}_{obs.uid}"

    if dir is not None:
        # Create the file and get the root group
        hfpath = os.path.join(dir, f"{namestr}.h5")
        hfpath_temp = f"{hfpath}.tmp"
        if parallel:
            hf = h5py.File(hfpath_temp, "w", driver="mpio", comm=obs.comm.comm_group)
            hgroup = hf
        elif obs.comm.group_rank == 0:
            hf = h5py.File(hfpath_temp, "w")
            hgroup = hf
    else:
        # We already have an open handle.  Create a subgroup with our name.
        if parallel or obs.comm.group_rank == 0:
            hgroup = root.create_group(namestr)

    # Participating processes now go and create all sub-groups and datasets
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

    # Dump shared data

    # FIXME:  When dumping data from a serial job, we have no mechanism for detecting
    # if a shared object is shared across the row, column, or group communicators.

    for skey, sdata in obs.shared.items():
        sdims = None
        sdtype = None
        commstr = None
        if obs.comm.group_rank == 0:
            sdtype = sdata.dtype
            if (
                sdata.comm is None
                or MPI.Comm.Compare(sdata.comm, obs.comm.comm_group) == MPI.IDENT
            ):
                # Using the group comm
                sdims = sdata.shape
                commstr = "group"
            elif MPI.Comm.Compare(sdata.comm, obs.comm_row) == MPI.IDENT:
                # Using the row comm.  Compute total shape.
                sdims = (len(obs.all_detectors),) + sdata.shape[1:]
                commstr = "row"
            elif MPI.Comm.Compare(sdata.comm, obs.comm_col) == MPI.IDENT:
                # Using the col comm.  Compute total shape.
                sdims = (obs.n_all_samples,) + sdata.shape[1:]
                commstr = "col"
            else:
                msg = f"Shared object '{skey}' cannot be written- only the "
                msg += "group, row, and column communicators are supported"
                log.error(msg)
                raise ValueError(msg)
        if obs.comm.comm_group is not None:
            sdims = obs.comm.comm_group.bcast(sdims, root=0)
            sdtype = obs.comm.comm_group.bcast(sdtype, root=0)
            commstr = obs.comm.comm_group.bcast(commstr, root=0)
        if parallel or obs.comm.group_rank == 0:
            sds = shared_group.create_dataset(skey, sdims, dtype=sdtype)
            sds.attrs["comm"] = commstr
        # Rank zero of each comm writes the data
        if sdata.comm is None or (commstr == "group" and obs.comm.group_rank == 0):
            sds[:] = sdata.data
        elif commstr == "row":
            if parallel:
                # The first process in each row of the grid has direct access to the
                # dataset.
                if obs.comm_row_rank == 0:
                sds[
                    obs.local_index_offset : obs.local_index_offset + obs.n_local_samples
                ] = sdata.data
            else:
                # Only the root process of the group has access to the file.
                # Communicate the data for writing, by looping over all process
                # rows and sending data from the first process in each row.
                for prow in range(obs.comm_col_size):
                    if obs.comm_col_rank == prow and obs.comm_row_rank == 0:
                        if prow == 0:
                            # This is the root process of the whole group- just write
                            # our piece.
                            sds[obs.dist.]
                        else:
                            # My turn to send
                    elif obs.comm.group_rank == 0:
                        # Root process of group receives and writes


        elif commstr == "col" and obs.comm_col_rank == 0:
            sds[
                obs.local_index_offset : obs.local_index_offset + obs.n_local_samples
            ] = sdata.data

    if parallel or obs.comm.group_rank == 0:
        hf.flush()
        hf.close()

    if obs.comm.comm_group is not None:
        obs.comm.comm_group.barrier()

    # Move file into place
    if obs.comm.group_rank == 0:
        os.rename(hfpath_temp, hfpath)

    return
