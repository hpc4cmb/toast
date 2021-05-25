#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple satellite simulation and makes a map.

NOTE:  This script is an example.  If you are doing a simulation for a specific
experiment, you should use a custom Focalplane class rather that the simple base class
used here.

You can see the automatically generated command line options with:

    toast_sim_satellite.py --help

Or you can dump a config file with all the default values with:

    toast_sim_satellite.py --default_toml config.toml

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

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
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_satellite_sim (total)")

    # Get optional MPI parameters
    world_comm, procs, rank = toast.get_world()

    # Argument parsing
    parser = argparse.ArgumentParser(description="Satellite Simulation Example.")

    # Arguments specific to this script

    parser.add_argument(
        "--focalplane", required=True, default=None, help="Input fake focalplane"
    )

    parser.add_argument(
        "--schedule", required=True, default=None, help="Input observing schedule"
    )

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="toast_sim_satellite_out",
        help="The output directory",
    )

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    #
    # We can also set some default values here for the traits.

    madam_available = toast.ops.madam.available()

    operators = [
        toast.ops.SimSatellite(name="sim_satellite"),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.ScanHealpix(name="scan_map"),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.PointingDetectorSimple(name="det_pointing"),
        toast.ops.PointingHealpix(name="pointing", mode="IQU"),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(name="mapmaker"),
        toast.ops.PointingHealpix(name="pointing_final", enabled=False, mode="IQU"),
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        ),
    ]
    if madam_available:
        operators.append(toast.ops.Madam(name="madam", enabled=False))

    # Templates we want to configure from the command line or a parameter file.

    templates = [toast.templates.Offset(name="baselines")]

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
        templates=templates,
    )

    # Create our output directory
    if rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(args.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=world_comm)

    # Load the focalplane file.  NOTE:  again, this is just using the
    # built-in Focalplane class.  In a workflow for a specific experiment we would
    # have a custom class.

    focalplane = toast.instrument.Focalplane(file=args.focalplane, comm=world_comm)

    # Load the schedule file

    schedule = toast.schedule.SatelliteSchedule()
    schedule.read(args.schedule, comm=world_comm)

    # Create a telescope for the simulation.  Again, for a specific experiment we
    # would use custom classes for the site.

    site = toast.instrument.SpaceSite(schedule.site_name)
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )

    # Instantiate our objects that were configured from the command line / files

    job = toast.create_from_config(config)
    ops = job.operators
    tmpls = job.templates

    # Are we using full pointing?  We determine this from whether the binning operator
    # used in the solve has full pointing enabled and also whether madam (which
    # requires full pointing) is enabled.

    full_pointing = False
    if madam_available and ops.madam.enabled:
        full_pointing = True
    if ops.binner.full_pointing:
        full_pointing = True

    # Find the group size for this job, either from command-line overrides or
    # by estimating the data volume.

    group_size = toast.job_group_size(
        world_comm,
        jobargs,
        schedule=schedule,
        focalplane=focalplane,
        full_pointing=full_pointing,
    )

    # Create the toast communicator

    comm = toast.Comm(world=world_comm, groupsize=group_size)

    # Create the (initially empty) data

    data = toast.Data(comm=comm)

    # Simulate the telescope pointing

    ops.sim_satellite.telescope = telescope
    ops.sim_satellite.schedule = schedule
    ops.sim_satellite.apply(data)

    # Construct a "perfect" noise model just from the focalplane parameters

    ops.default_model.apply(data)

    # Set up the pointing.  Each pointing matrix operator requires a detector pointing
    # operator, and each binning operator requires a pointing matrix operator.

    ops.pointing.detector_pointing = ops.det_pointing
    ops.pointing_final.detector_pointing = ops.det_pointing

    ops.binner.pointing = ops.pointing
    ops.binner_final.pointing = ops.pointing_final

    # Are we using different pointing and binning operators for our final binning?
    # If not, we set those to be the same operators used for the solve.

    if not (ops.binner_final.enabled and ops.pointing_final.enabled):
        ops.binner_final = ops.binner
        ops.pointing_final = ops.pointing

    # Simulate sky signal from a map.  We scan the sky with the "final" pointing model
    # in case that is different from the solver pointing model.

    ops.scan_map.pixel_dist = ops.binner_final.pixel_dist
    ops.scan_map.pointing = ops.pointing_final
    ops.scan_map.save_pointing = full_pointing
    ops.scan_map.apply(data)

    # Simulate detector noise

    ops.sim_noise.apply(data)

    # The map maker requires the the binning operators used for the solve and final,
    # the templates, and the noise model.

    ops.binner.noise_model = ops.default_model.noise_model
    ops.binner_final.noise_model = ops.default_model.noise_model

    ops.mapmaker.binning = ops.binner
    ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[tmpls.baselines])
    ops.mapmaker.map_binning = ops.binner_final
    ops.mapmaker.det_data = ops.sim_noise.det_data
    ops.mapmaker.output_dir = args.out_dir

    ops.mapmaker.apply(data)

    # Run Madam
    if madam_available:
        ops.madam.apply(data)

    alltimers = toast.timing.gather_timers(comm=comm.comm_world)
    if comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        toast.timing.dump(alltimers, out)


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
