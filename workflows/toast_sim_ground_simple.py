#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple ground simulation and makes a map.

This script is an example of fully specifying all operators and options within
the script (rather than from config files and command line options).  This is a
useful starting point for interactively hacking on a specific use case or test.

"""

import argparse
import os
import sys
import traceback

import numpy as np
from astropy import units as u

import toast
from toast.mpi import MPI

import toast.io
import toast.schedule
import toast.ops


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    # Get just our focalplane and schedule from the command line
    parser = argparse.ArgumentParser(description="Simple Ground Simulation Example.")

    parser.add_argument(
        "--focalplane", required=True, default=None, help="Input fake focalplane"
    )

    parser.add_argument(
        "--schedule", required=True, default=None, help="Input observing schedule"
    )

    parser.add_argument(
        "--thinfp",
        required=False,
        default=1,
        help="Focalplane thinning factor",
        type=int,
    )

    parser.add_argument(
        "--weather",
        required=False,
        default=None,
        help="Weather information. Infer from observing site if not provided.",
    )

    args = parser.parse_args()

    # Create our output directory
    out_dir = "toast_sim_ground_simple"
    if comm is None or comm.rank == 0:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    # Load a generic focalplane file.
    focalplane = toast.instrument.Focalplane(thinfp=args.thinfp)
    with toast.io.H5File(args.focalplane, "r", comm=comm, force_serial=True) as f:
        focalplane.load_hdf5(f.handle, comm=comm)

    # Load the schedule file
    schedule = toast.schedule.GroundSchedule()
    schedule.read(args.schedule, comm=comm)

    # Create a telescope for the simulation.  Again, for a specific experiment we
    # would use custom classes for the site.
    site = toast.instrument.GroundSite(
        schedule.site_name,
        schedule.site_lat,
        schedule.site_lon,
        schedule.site_alt,
    )
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )

    # Create the toast communicator.  Use the default of one group.
    toast_comm = toast.Comm(world=comm)

    # Create the (initially empty) data
    data = toast.Data(comm=toast_comm)

    # Simulate data
    # ---------------------------------------------------------------

    # Try using site name for weather if not user-provided
    weather = None
    if args.weather is None:
        weather = schedule.site_name.lower()
    else:
        weather = args.weather

    # Simulate the telescope pointing
    sim_ground = toast.ops.SimGround(
        telescope=telescope,
        schedule=schedule,
        detset_key="pixel",
        weather=weather,
    )
    sim_ground.apply(data)

    # Construct a "perfect" noise model just from the focalplane parameters
    default_model = toast.ops.DefaultNoiseModel()
    default_model.apply(data)

    # Set up detector pointing.  This just uses the focalplane offsets.
    det_pointing_azel = toast.ops.PointingDetectorSimple(
        boresight=sim_ground.boresight_azel, quats="quats_azel"
    )
    det_pointing_radec = toast.ops.PointingDetectorSimple(
        boresight=sim_ground.boresight_radec, quats="quats_radec"
    )

    # Elevation-modulated noise model.
    elevation_model = toast.ops.ElevationNoise(
        noise_model=default_model.noise_model,
        out_model="el_weighted_model",
        detector_pointing=det_pointing_azel,
        view=det_pointing_azel.view,
    )
    elevation_model.apply(data)

    # Set up the pointing matrix.  We will use the same pointing matrix for the
    # template solve and the final binning.
    pixel_pointing = toast.ops.PixelsHealpix(
        nside=256, mode="IQU", detector_pointing=det_pointing_radec
    )
    weights = toast.ops.StokesWeights(
        nside=256, mode="IQU", detector_pointing=det_pointing_radec
    )

    # Simulate sky signal from a map and accumulate.
    # scan_map = toast.ops.ScanHealpixMap(
    #     pixel_pointing=pixel_pointing, file="input.fits"
    # )
    # scan_map.apply(data)

    # Simulate detector noise and accumulate.
    sim_noise = toast.ops.SimNoise(noise_model=elevation_model.out_model)
    sim_noise.apply(data)

    # Simulate atmosphere signal
    sim_atm = toast.ops.SimAtmosphere(detector_pointing=det_pointing_azel)
    sim_atm.apply(data)

    # Reduce data
    # ---------------------------------------------------------------

    # Set up the binning operator.  We will use the same binning for the template solve
    # and the final map.
    binner = toast.ops.BinMap(
        pixel_pointing=pixel_pointing,
        stokes_weights=weights,
        noise_model=elevation_model.out_model,
    )

    # FIXME:  Apply filtering here, and optionally pass an empty template
    # list to disable the template solve and just make a binned map.

    # Set up the template matrix for the solve
    template_matrix = toast.ops.TemplateMatrix(
        templates=[
            toast.templates.Offset(),
        ]
    )

    # Map making
    mapmaker = toast.ops.MapMaker(
        det_data=sim_noise.det_data,
        binning=binner,
        template_matrix=template_matrix,
        output_dir=out_dir,
    )
    mapmaker.apply(data)


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
