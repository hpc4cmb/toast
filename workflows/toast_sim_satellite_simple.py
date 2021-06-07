#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple satellite simulation and makes a map.

This script is an example of fully specifying all operators and options within
the script (rather than from config files and command line options).  This is a
useful starting point for interactively hacking on a specific use case or test.

"""

import os
import sys
import traceback
import argparse

import numpy as np

from astropy import units as u

import toast

from toast.mpi import MPI


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    # Get just our focalplane and schedule from the command line
    parser = argparse.ArgumentParser(description="Simple Satellite Simulation Example.")

    parser.add_argument(
        "--focalplane", required=True, default=None, help="Input fake focalplane"
    )

    parser.add_argument(
        "--schedule", required=True, default=None, help="Input observing schedule"
    )

    args = parser.parse_args()

    # Create our output directory
    out_dir = "toast_sim_satellite_simple"
    if comm is None or comm.rank == 0:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    # Load a generic focalplane file.
    focalplane = toast.instrument.Focalplane(file=args.focalplane, comm=comm)

    # Load the schedule file
    schedule = toast.schedule.SatelliteSchedule()
    schedule.read(args.schedule, comm=comm)

    # Create a telescope for the simulation.  Again, for a specific experiment we
    # would use custom classes for the site.
    site = toast.instrument.SpaceSite(schedule.site_name)
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )

    # Create the toast communicator.  Use the default of one group.
    toast_comm = toast.Comm(world=comm)

    # Create the (initially empty) data
    data = toast.Data(comm=toast_comm)

    # Set up some operators that we are going to use later in both
    # Simulation and Reduction.
    #---------------------------------------------------------------

    # Construct a "perfect" noise model just from the focalplane parameters
    default_model = toast.ops.DefaultNoiseModel()

    # Set up detector pointing.  This just uses the focalplane offsets.
    det_pointing = toast.ops.PointingDetectorSimple()

    # Set up the pointing matrix.  We will use the same pointing matrix for the
    # template solve and the final binning.
    pointing = toast.ops.PointingHealpix(
        nside=512, 
        mode="IQU",
        detector_pointing=det_pointing
    )

    # Simulate data
    #---------------------------------------------------------------

    # Simulate the telescope pointing
    sim_satellite = toast.ops.SimSatellite(
        telescope=telescope,
        schedule=schedule
    )
    sim_satellite.apply(data)

    # Create a default noise model from focalplane parameters
    default_model.apply(data)

    # Simulate sky signal from a map and accumulate.
    # scan_map = toast.ops.ScanHealpix(
    #     pointing=pointing,
    #     file="input.fits"
    # )
    # scan_map.apply(data)

    # Simulate detector noise and accumulate.
    sim_noise = toast.ops.SimNoise()
    sim_noise.apply(data)

    # Reduce data
    #---------------------------------------------------------------

    # Set up the binning operator.  We will use the same binning for the template solve
    # and the final map.
    binner = toast.ops.BinMap(
        pointing=pointing,
        noise_model=default_model.noise_model
    )

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
    try:
        main()
    except Exception:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = toast.get_world()
        if procs == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None:
            mpiworld.Abort()
