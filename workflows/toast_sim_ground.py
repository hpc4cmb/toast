#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple ground simulation and makes a map.

NOTE:  This script is an example.  If you are doing a simulation for a specific
experiment, you should use a custom Focalplane class rather that the simple base class
used here.

You can see the automatically generated command line options with:

    toast_sim_ground.py --help

Or you can dump a config file with all the default values with:

    toast_sim_ground.py --default_toml config.toml

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


def parse_config(operators, templates, comm):
    """Parse command line arguments and load any config files.

    Return the final config, remaining args, and job size args.

    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Ground Simulation Example.")

    # Arguments specific to this script

    parser.add_argument(
        "--focalplane", required=True, default=None, help="Input fake focalplane"
    )

    parser.add_argument(
        "--schedule", required=True, default=None, help="Input observing schedule"
    )

    # FIXME: Add weather name / path here

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="toast_sim_ground_out",
        help="The output directory",
    )

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
        templates=templates,
    )

    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(args.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    return config, args, jobargs


def load_instrument_and_schedule(args, comm):
    # Load a generic focalplane file.  NOTE:  again, this is just using the
    # built-in Focalplane class.  In a workflow for a specific experiment we would
    # have a custom class.
    focalplane = toast.instrument.Focalplane(file=args.focalplane, comm=comm)

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
        weather=None,
    )
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )
    return telescope, schedule


def use_full_pointing(job):
    # Are we using full pointing?  We determine this from whether the binning operator
    # used in the solve has full pointing enabled and also whether madam (which
    # requires full pointing) is enabled.
    full_pointing = False
    if toast.ops.madam.available() and job.operators.madam.enabled:
        full_pointing = True
    if job.operators.binner.full_pointing:
        full_pointing = True
    return full_pointing


def job_create(config, jobargs, telescope, schedule, comm):
    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)

    # Find the group size for this job, either from command-line overrides or
    # by estimating the data volume.
    full_pointing = use_full_pointing(job)
    group_size = toast.job_group_size(
        comm,
        jobargs,
        schedule=schedule,
        focalplane=telescope.focalplane,
        full_pointing=full_pointing,
    )
    return job, group_size, full_pointing


def simulate_data(job, toast_comm, telescope, schedule):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates
    world_comm = toast_comm.comm_world

    # Create the (initially empty) data

    data = toast.Data(comm=toast_comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Simulate the telescope pointing

    ops.sim_ground.telescope = telescope
    ops.sim_ground.schedule = schedule
    if ops.sim_ground.weather is None:
        ops.sim_ground.weather = telescope.site.name
    ops.sim_ground.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=world_comm, timer=timer)

    # Construct a "perfect" noise model just from the focalplane parameters

    ops.default_model.apply(data)
    log.info_rank("Created default noise model in", comm=world_comm, timer=timer)

    # Set up detector pointing in both Az/El and RA/DEC

    ops.det_pointing_azel.boresight = ops.sim_ground.boresight_azel
    ops.det_pointing_radec.boresight = ops.sim_ground.boresight_radec

    # Create the Elevation modulated noise model

    ops.elevation_model.noise_model = ops.default_model.noise_model
    ops.elevation_model.detector_pointing = ops.det_pointing_azel
    ops.elevation_model.view = ops.det_pointing_azel.view
    ops.elevation_model.apply(data)
    log.info_rank("Created elevation noise model in", comm=world_comm, timer=timer)

    # Set up the pointing.  Each pointing matrix operator requires a detector pointing
    # operator, and each binning operator requires a pointing matrix operator.
    ops.pointing.detector_pointing = ops.det_pointing_radec
    ops.pointing_final.detector_pointing = ops.det_pointing_radec

    ops.binner.pointing = ops.pointing

    # If we are not using a different pointing matrix for our final binning, then
    # use the same one as the solve.
    if not ops.pointing_final.enabled:
        ops.pointing_final = ops.pointing

    ops.binner_final.pointing = ops.pointing_final

    # If we are not using a different binner for our final binning, use the same one
    # as the solve.
    if not ops.binner_final.enabled:
        ops.binner_final = ops.binner

    # Simulate sky signal from a map.  We scan the sky with the "final" pointing model
    # in case that is different from the solver pointing model.

    ops.scan_map.pixel_dist = ops.binner_final.pixel_dist
    ops.scan_map.pointing = ops.pointing_final
    ops.scan_map.save_pointing = use_full_pointing(job)
    ops.scan_map.apply(data)
    log.info_rank("Simulated sky signal in", comm=world_comm, timer=timer)

    # Simulate detector noise

    ops.sim_noise.noise_model = ops.elevation_model.out_model
    ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    # Simulate atmosphere

    ops.sim_atmosphere.detector_pointing = ops.det_pointing_azel
    ops.sim_atmosphere.apply(data)
    log.info_rank("Simulated and observed atmosphere in", comm=world_comm, timer=timer)

    return data


def reduce_data(job, args, data):
    log = toast.utils.Logger.get()
    ops = job.operators
    tmpls = job.templates

    world_comm = data.comm.comm_world

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Collect signal statistics

    ops.statistics.output_dir = args.out_dir
    ops.statistics.apply(data)

    # Apply the filter stack

    ops.groundfilter.apply(data)
    log.info_rank("Finished ground-filtering in", comm=world_comm, timer=timer)
    ops.polyfilter1D.apply(data)
    log.info_rank("Finished 1D-poly-filtering in", comm=world_comm, timer=timer)
    ops.polyfilter2D.apply(data)
    log.info_rank("Finished 2D-poly-filtering in", comm=world_comm, timer=timer)
    ops.common_mode_filter.apply(data)
    log.info_rank("Finished common-mode-filtering in", comm=world_comm, timer=timer)

    # The map maker requires the the binning operators used for the solve and final,
    # the templates, and the noise model.

    ops.binner.noise_model = ops.elevation_model.out_model
    ops.binner_final.noise_model = ops.elevation_model.out_model

    ops.mapmaker.binning = ops.binner
    ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[tmpls.baselines])
    ops.mapmaker.map_binning = ops.binner_final
    ops.mapmaker.det_data = ops.sim_noise.det_data
    ops.mapmaker.output_dir = args.out_dir

    ops.mapmaker.apply(data)
    log.info_rank("Finished map-making in", comm=world_comm, timer=timer)

    # Optionally run Madam
    if toast.ops.madam.available():
        ops.madam.apply(data)
        log.info_rank("Finished Madam in", comm=world_comm, timer=timer)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_ground_sim (total)")

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    #
    # We can also set some default values here for the traits, including whether an
    # operator is disabled by default.

    # FIXME:  This example workflow will eventually include other operators for
    # atmosphere simulation, filtering, other types of map-making, etc.

    operators = [
        toast.ops.SimGround(name="sim_ground"),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.ElevationNoise(
            name="elevation_model",
            out_model="el_noise_model",
        ),
        toast.ops.PointingDetectorSimple(name="det_pointing_azel", quats="quats_azel"),
        toast.ops.PointingDetectorSimple(
            name="det_pointing_radec", quats="quats_radec"
        ),
        toast.ops.ScanHealpix(name="scan_map"),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.SimAtmosphere(name="sim_atmosphere"),
        toast.ops.PointingHealpix(name="pointing", mode="IQU"),
        toast.ops.Statistics(name="statistics"),
        toast.ops.GroundFilter(name="groundfilter"),
        toast.ops.PolyFilter(name="polyfilter1D"),
        toast.ops.PolyFilter2D(name="polyfilter2D"),
        toast.ops.CommonModeFilter(name="common_mode_filter"),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(name="mapmaker"),
        toast.ops.PointingHealpix(name="pointing_final", enabled=False, mode="IQU"),
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        ),
    ]
    if toast.ops.madam.available():
        operators.append(toast.ops.Madam(name="madam", enabled=False))

    # Templates we want to configure from the command line or a parameter file.
    templates = [toast.templates.Offset(name="baselines")]

    # Parse options
    config, args, jobargs = parse_config(operators, templates, comm)

    # Load our instrument model and observing schedule
    telescope, schedule = load_instrument_and_schedule(args, comm)

    # Instantiate our operators and get the size of the process groups
    job, group_size, full_pointing = job_create(
        config, jobargs, telescope, schedule, comm
    )

    # Create the toast communicator
    toast_comm = toast.Comm(world=comm, groupsize=group_size)

    # Create simulated data
    data = simulate_data(job, toast_comm, telescope, schedule)

    # Reduce the data
    reduce_data(job, args, data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=toast_comm.comm_world)
    if toast_comm.world_rank == 0:
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
