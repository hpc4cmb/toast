# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np

import datetime

from astropy import units as u

import h5py

import json

from ..utils import (
    Environment,
    Logger,
    import_from_name,
    dtype_to_aligned,
    have_hdf5_parallel,
)

from ..mpi import MPI

from ..timing import Timer, function_timer, GlobalTimers

from ..instrument import GroundSite, SpaceSite, Focalplane, Telescope

from ..weather import SimWeather

from ..observation import Observation


@function_timer
def load_hdf5_shared(obs, hgrp, fields):
    log = Logger.get()

    # Get references to the distribution of detectors and samples
    proc_rows = obs.dist.process_rows
    proc_cols = obs.dist.comm.group_size // proc_rows
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    for field in list(hgrp.keys()):
        comm_type = hgrp[field].attrs["comm_type"]
        full_shape = hgrp[field].shape
        dtype = hgrp[field].dtype

        slc = list()
        shape = list()
        if comm_type == "row":
            off = dist_dets[obs.comm.group_rank].offset
            nelem = dist_dets[obs.comm.group_rank].n_elem
            slc.append(slice(off, off + nelem))
            shape.append(nelem)
        elif comm_type == "column":
            off = dist_samps[obs.comm.group_rank].offset
            nelem = dist_samps[obs.comm.group_rank].n_elem
            slc.append(slice(off, off + nelem))
            shape.append(nelem)
        else:
            slc.append(slice(0, full_shape[0]))
            shape.append(full_shape[0])
        if len(full_shape) > 1:
            for dim in full_shape[1:]:
                slc.append(slice(0, dim))
                shape.append(dim)
        slc = tuple(slc)
        shape = tuple(shape)

        print(f"create {field} ({comm_type}) with shape {shape}, dtype {dtype}")
        obs.shared.create_type(comm_type, field, shape, dtype)
        shcomm = obs.shared[field].comm
        print(shcomm)

        # Load the data on one process of the communicator
        data = None
        if shcomm is None or shcomm.rank == 0:
            data = np.array(hgrp[field][slc], copy=False).astype(
                obs.shared[field].dtype
            )
            print(f"shared data '{field}' ({data.dtype}) = {data}")

        obs.shared[field].set(data, fromrank=0)

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
    obs_dets = json.loads(hgroup.attrs["observation_detectors"])
    obs_det_sets = json.loads(hgroup.attrs["observation_detector_sets"])
    obs_samples = hgroup.attrs["observation_samples"]
    obs_sample_sets = [
        [int(x) for x in y] for y in json.loads(hgroup.attrs["observation_sample_sets"])
    ]

    # Instrument properties

    # FIXME:  We should add save / load methods to these classes to
    # generalize this and allow use of other classes.

    inst_group = hgroup["instrument"]
    telescope_name = inst_group.attrs["telescope_name"]
    telescope_class_name = inst_group.attrs["telescope_class"]
    telescope_uid = inst_group.attrs["telescope_uid"]

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
            weather_time = datetime.datetime.fromtimestamp(
                inst_group.attrs["site_weather_time"]
            )
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

    focalplane = Focalplane(file=inst_group, comm=comm.comm_group)

    telescope = Telescope(
        telescope_name, uid=telescope_uid, focalplane=focalplane, site=site
    )

    # Create the observation

    obs = Observation(
        comm,
        telescope,
        obs_samples,
        name=obs_name,
        uid=obs_uid,
        detector_sets=obs_det_sets,
        sample_sets=obs_sample_sets,
        process_rows=process_rows,
    )

    # Load other metadata

    meta_group = hgroup["metadata"]

    # Load shared data

    shared_group = hgroup["shared"]
    load_hdf5_shared(obs, shared_group, shared)

    # Load intervals

    intervals_group = hgroup["intervals"]
    intervals_times = intervals_group.attrs["times"]
    load_hdf5_intervals(obs, intervals_group, intervals)

    # Load detector data

    detdata_group = hgroup["detdata"]
    load_hdf5_detdata(obs, detdata_group, detdata)

    return obs
