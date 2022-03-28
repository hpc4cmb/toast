# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..mpi import MPI, use_mpi

from ..utils import Logger

from .io import have_hdf5_parallel


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
                        inst_group.attrs["site_weather_max_pwv"] = site.weather.max_pwv
                    inst_group.attrs[
                        "site_weather_time"
                    ] = site.weather.time.timestamp()
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
            v.save_hdf5(kgroup, comm=comm)
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
        fields = list(obs.detdata.keys())
    else:
        fields = list(detdata)
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
