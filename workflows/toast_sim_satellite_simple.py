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
from toast.ops import noise_model


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
    focalplane = toast.instrument.Focalplane()
    hf = toast.io.hdf5_open(args.focalplane, "r", comm=comm, force_serial=True)
    focalplane.load_hdf5(hf, comm=comm)
    if hf is not None:
        hf.close()
    del hf

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

    # Simulate data
    # ---------------------------------------------------------------

    # Simulate the telescope pointing
    sim_satellite = toast.ops.SimSatellite(
        telescope=telescope, schedule=schedule, detset_key="pixel"
    )
    sim_satellite.apply(data)

    # Construct a "perfect" noise model just from the focalplane parameters
    default_model = toast.ops.DefaultNoiseModel()
    default_model.apply(data)

    # Set up detector pointing.  This just uses the focalplane offsets.
    det_pointing = toast.ops.PointingDetectorSimple(boresight=sim_satellite.boresight)

    # Set up the pointing matrix.  We will use the same pointing matrix for the
    # template solve and the final binning.
    pointing = toast.ops.PointingHealpix(
        nside=512, mode="IQU", detector_pointing=det_pointing
    )

    # Simulate sky signal from a map and accumulate.
    # scan_map = toast.ops.ScanHealpix(
    #     pointing=pointing,
    #     file="input.fits"
    # )
    # scan_map.apply(data)

    # Simulate detector noise and accumulate.
    sim_noise = toast.ops.SimNoise(noise_model=default_model.noise_model)
    sim_noise.apply(data)

    # Reduce data
    # ---------------------------------------------------------------

    # Set up the binning operator.  We will use the same binning for the template solve
    # and the final map.
    binner = toast.ops.BinMap(pointing=pointing, noise_model=default_model.noise_model)

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
