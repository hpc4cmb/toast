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
    import_from_name,
    dtype_to_aligned,
)

from ..mpi import MPI

from ..timing import Timer, function_timer, GlobalTimers

from ..instrument import GroundSite, SpaceSite

from ..weather import SimWeather


@function_timer
def load_hdf5_shared(obs, hgrp, fields):
    log = Logger.get()
    parallel = have_hdf5_parallel()
    return


@function_timer
def load_hdf5_detdata(obs, hgrp, fields):
    pass


@function_timer
def load_hdf5_intervals(obs, hgrp, fields):
    pass


# FIXME:  Add options here to prune detectors on load.


@function_timer
def load_hdf5(
    path,
    comm,
    process_rows=None,
    meta=None,
    detdata=None,
    shared=None,
    intervals=None,
):
    """Load an HDF5 observation.

    By default, all detdata, shared, intervals, and noise models are loaded into
    memory.  A subset of objects may be specified with a list of names
    passed to the corresponding function arguments.

    Args:
        path (str):  The path to the file on disk.
        comm (toast.Comm):  The toast communicator to use.
        process_rows (int):  (Optional) The size of the rectangular process grid
            in the detector direction.  This number must evenly divide into the size of
            comm.  If not specified, defaults to the size of the communicator.
        meta (list):  Only save this list of metadata objects.
        detdata (list):  Only save this list of detdata objects.
        shared (list):  Only save this list of shared objects.
        intervals (list):  Only save this list of intervals objects.

    Returns:
        (Observation):  The constructed observation.

    """
    log = Logger.get()
    env = Environment.get()
    parallel = have_hdf5_parallel()

    # Open the file and get the root group.  In both serial and parallel HDF5,
    # multiple readers are supported.  We open the file with all processes to
    # enable reading detector and shared data in parallel.
    hf = None
    hfgroup = None
    if parallel:
        hf = h5py.File(path, "r", driver="mpio", comm=comm.comm_group)
        hgroup = hf
    else:
        hf = h5py.File(path, "r")
        hgroup = hf

    # Observation properties
    obs_name = hgroup.attrs["observation_name"]
    obs_uid = hgroup.attrs["observation_uid"]
    # det_sets = hgroup.attrs["observation_detector_sets"]

    # Instrument properties

    # FIXME:  We should add save / load methods to these classes to
    # generalize this and allow use of other classes.

    inst_group = hgroup["instrument"]
    telescope_name = inst_group.attrs["telescope_name"]
    telescope_class_name = inst_group.attrs["telescope_class"]
    telescope_uid = inst_group.attrs["telescope_uid"] = obs.telescope.uid

    site_name = inst_group.attrs["site_name"]
    site_class_name = inst_group.attrs["site_class"]
    site_uid = inst_group.attrs["site_uid"]

    site = None
    if "site_alt_m" in inst_group.attrs:
        # This is a ground based site
        site_alt_m = inst_group.attrs["site_alt_m"]
        site_lat_deg = inst_group.attrs["site_lat_deg"]
        site_lon_deg = inst_group.attrs["site_lon_deg"]

        weather = None
        if "site_weather_name" in inst_group.attrs:
            weather_name = inst_group.attrs["site_weather_name"]
            weather_realization = inst_group.attrs["site_weather_realization"]
            weather_max_pwv = None
            if inst_group.attrs["site_weather_max_pwv"] != "NONE":
                weather_max_pwv = inst_group.attrs["site_weather_max_pwv"]
            weather_time = inst_group.attrs["site_weather_time"]
            weather = SimWeather(
                time=weather_time,
                name=weather_name,
                site_uid=site_uid,
                realization=weather_realization,
                max_pwv=weather_max_pwv,
                median_weather=False,
            )
        site = GroundSite(
            site_name,
            site_lat_deg * u.degree,
            site_lon_deg * u.degree,
            site_alt_m * u.meter,
            uid=site_uid,
            weather=weather,
        )
    else:
        site = SpaceSite(site_name, uid=site_uid)

    focalplane = Focalplane()
    focalplane.load_hdf5(inst_group["focalplane"], comm=comm.comm_group)

    telescope = Telescope(
        telescope_name, uid=telescope_uid, focalplane=focalplane, site=site
    )

    # Create the observation

    # Load other metadata

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

    # Load shared data

    # Load intervals

    # Load detector data

    shared_group = hgroup.create_group("shared")
    detdata_group = hgroup.create_group("detdata")
    intervals_group = hgroup.create_group("intervals")
    intervals_group.attrs["times"] = times

    return
