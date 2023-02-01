#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script sets up the input files needed by the toast_benchmark_ground workflow.
"""

import argparse
import os
import shutil
import sys
import warnings

import numpy as np
from astropy import units as u
from erfa import ErfaWarning

import toast
import toast.ops
from toast.instrument_sim import fake_hexagon_focalplane
from toast.schedule_sim_ground import run_scheduler
from toast.scripts.benchmarking_utilities import (
    create_input_maps,
    default_sim_atmosphere,
    get_mpi_settings,
    get_standard_ground_args,
    make_focalplane,
    python_startup_time,
)
from toast.timing import dump, function_timer, gather_timers

warnings.simplefilter("ignore", ErfaWarning)


def parse_arguments():
    """
    Defines and parses the arguments for the script.
    """
    # defines the parameters of the script
    parser = argparse.ArgumentParser(
        description="Setup inputs for the TOAST ground benchmark"
    )
    parser.add_argument(
        "--work_dir",
        required=False,
        default="toast_benchmark_ground_inputs",
        type=str,
        help="The working directory to use for benchmark inputs",
    )
    parser.add_argument(
        "--node_mem_gb",
        required=False,
        default=None,
        type=float,
        help="Use this much memory per node in GB",
    )
    parser.add_argument(
        "--overwrite",
        required=False,
        default=False,
        action="store_true",
        help="Overwrite files if they already exist",
    )

    parser.add_argument(
        "--atmosphere",
        required=False,
        default=False,
        action="store_true",
        help="Pre-simulate the atmosphere as well",
    )

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    operators = [
        toast.ops.SimGround(
            name="sim_ground",
            detset_key="pixel",
            median_weather=True,
        ),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.ElevationNoise(name="elevation_model", out_model="el_noise_model"),
        toast.ops.PointingDetectorSimple(name="det_pointing_azel", quats="quats_azel"),
        toast.ops.PointingDetectorSimple(
            name="det_pointing_radec", quats="quats_radec"
        ),
        default_sim_atmosphere(),
        toast.ops.PixelsHealpix(name="pixels"),
        toast.ops.StokesWeights(name="weights", mode="IQU"),
        toast.ops.MemoryCounter(name="mem_count", enabled=False),
    ]

    # generates config
    config, args, jobargs = toast.parse_config(parser, operators=operators)

    # hardcoded arguments
    args.dry_run = None
    args.max_n_det = None

    get_standard_ground_args(args)

    args.n_detector = args.max_detector

    return config, args, jobargs


def make_schedule(args, world_comm, site):
    """
    Creates a ground schedule
    """
    schedule = None
    if (world_comm is None) or (world_comm.rank == 0):
        schedule_file = os.path.join(args.work_dir, "ground_bench_schedule.txt")
        if not os.path.exists(schedule_file) or args.overwrite:
            # Building the schedule is serial and takes a while, only generate it
            # if needed.
            sch_opts = [
                "--site-name",
                args.site_name,
                "--telescope",
                args.telescope_name,
                "--site-lon",
                f"{site.earthloc.lon.to_value(u.degree)}",
                "--site-lat",
                f"{site.earthloc.lat.to_value(u.degree)}",
                "--site-alt",
                f"{site.earthloc.height.to_value(u.meter)}",
                "--patch-coord",
                args.patch_coord,
                "--el-min",
                f"{args.el_min}",
                "--el-max",
                f"{args.el_max}",
                "--sun-el-max",
                f"{args.sun_el_max}",
                "--sun-avoidance-angle",
                f"{args.sun_avoidance_angle}",
                "--moon-avoidance-angle",
                f"{args.moon_avoidance_angle}",
                "--gap-s",
                f"{args.gap_s}",
                "--gap-small-s",
                f"{args.gap_small_s}",
                "--ces-max-time",
                f"{args.ces_max_time}",
                "--boresight-angle-step",
                f"{args.boresight_angle_step}",
                "--boresight-angle-time",
                f"{args.boresight_angle_time}",
                "--start",
                args.schedule_start,
                "--stop",
                args.schedule_stop,
                "--out",
                schedule_file,
            ]
            for patch in args.schedule_patches:
                sch_opts.extend(["--patch", patch])
            run_scheduler(opts=sch_opts)
        schedule = toast.schedule.GroundSchedule()
        schedule.read(schedule_file)
    if world_comm is not None:
        schedule = world_comm.bcast(schedule, root=0)
    args.schedule = schedule


@function_timer
def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    env.enable_function_timers()
    global_timer = toast.timing.GlobalTimers.get()
    global_timer.start("toast_benchmark_ground_setup (total)")

    # defines and gets the arguments for the script
    config, args, jobargs = parse_arguments()

    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)
    job_ops = job.operators

    # gets the MPI parameters
    world_comm, n_procs, rank, n_nodes, avail_node_bytes = get_mpi_settings(
        args, log, env
    )

    # Log the config
    if rank == 0:
        if not os.path.isdir(args.work_dir):
            os.makedirs(args.work_dir)
        outlog = os.path.join(args.work_dir, "config_log.toml")
        toast.config.dump_toml(outlog, config)

    # Observing site
    site = toast.instrument.GroundSite(
        args.site_name, args.site_lat, args.site_lon, args.site_alt
    )

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Create the schedule, so that we know the total observing time.
    make_schedule(args, world_comm, site)
    log.info_rank("Construct full observing schedule in", comm=world_comm, timer=timer)

    # Create the input sky map
    map_file = os.path.join(args.work_dir, args.input_map)
    if rank == 0:
        if os.path.isfile(map_file) and args.overwrite:
            os.remove(map_file)
    create_input_maps(
        map_file,
        args.nside,
        rank,
        log,
        should_print_input_map_png=True,
    )
    log.info_rank("Create input sky maps in", comm=world_comm, timer=timer)

    # Creates the focalplane file.
    focalplane = make_focalplane(args, world_comm, log)

    # Create a telescope for the simulation.
    telescope = toast.instrument.Telescope(
        args.telescope_name, focalplane=focalplane, site=site
    )

    # Create the toast communicator.  We use a group size of one process so that each
    # process works on independent atmosphere sims.
    comm = toast.Comm(world=world_comm, groupsize=1)

    # Create the (initially empty) data
    data = toast.Data(comm=comm)

    job_ops.mem_count.prefix = "Before Simulation"
    job_ops.mem_count.apply(data)

    if args.atmosphere:
        # Loop over observations in the schedule and simulate one observation
        # at a time to reduce the memory overhead.

        n_scan = len(args.schedule.scans)

        scan_offset = 0
        while scan_offset < n_scan:
            n_scan_batch = comm.ngroups
            if scan_offset + 2 * n_scan_batch > n_scan:
                # Finish off the remaining observations, so that we are not
                # left with fewer scans than the number of groups.
                n_scan_batch = n_scan - scan_offset
            batch = [
                args.schedule.scans[x]
                for x in range(scan_offset, scan_offset + n_scan_batch)
            ]
            batch_str = f"{scan_offset} - {scan_offset + n_scan_batch - 1}"
            log.info_rank(
                f"Caching simulated atmosphere for observation batch {batch_str}",
                comm=world_comm,
            )
            batch_schedule = toast.schedule.GroundSchedule(
                scans=batch,
                site_name=args.schedule.site_name,
                telescope_name=args.schedule.telescope_name,
                site_lat=args.schedule.site_lat,
                site_lon=args.schedule.site_lon,
                site_alt=args.schedule.site_alt,
            )

            # Simulate the telescope pointing
            job_ops.sim_ground.telescope = telescope
            job_ops.sim_ground.schedule = batch_schedule
            job_ops.sim_ground.weather = telescope.site.name
            job_ops.sim_ground.apply(data)
            log.info_rank(
                "  Simulated telescope pointing in", comm=world_comm, timer=timer
            )

            # Set up detector pointing in both Az/El and RA/DEC
            job_ops.det_pointing_azel.boresight = job_ops.sim_ground.boresight_azel
            job_ops.det_pointing_radec.boresight = job_ops.sim_ground.boresight_radec

            # Construct a "perfect" noise model just from the focalplane parameters
            job_ops.default_model.apply(data)
            log.info_rank(
                "  Created default noise model in", comm=world_comm, timer=timer
            )

            # Create the Elevation modulated noise model
            job_ops.elevation_model.noise_model = job_ops.default_model.noise_model
            job_ops.elevation_model.detector_pointing = job_ops.det_pointing_azel
            job_ops.elevation_model.view = job_ops.det_pointing_azel.view
            job_ops.elevation_model.apply(data)
            log.info_rank(
                "Created elevation noise model in", comm=world_comm, timer=timer
            )

            # Set the pointing matrix operators to use the detector pointing
            job_ops.pixels.nside = args.nside
            job_ops.pixels.detector_pointing = job_ops.det_pointing_radec
            job_ops.weights.detector_pointing = job_ops.det_pointing_radec

            # Simulate atmosphere signal
            cache_dir = os.path.join(args.work_dir, "atmosphere")
            if rank == 0:
                if os.path.isdir(cache_dir) and args.overwrite:
                    os.shutil.rmtree(cache_dir)
            if world_comm is not None:
                world_comm.barrier()
            job_ops.sim_atmosphere.detector_pointing = job_ops.det_pointing_azel
            job_ops.sim_atmosphere.cache_dir = cache_dir
            job_ops.sim_atmosphere.cache_only = True
            job_ops.sim_atmosphere.apply(data)
            log.info_rank("  Simulated atmosphere in", comm=world_comm, timer=timer)

            job_ops.mem_count.prefix = f"After observation batch {batch_str}"
            job_ops.mem_count.apply(data)

            # Clear data for this observation
            data.clear()

            scan_offset += n_scan_batch

    # dumps all the timing information
    timer.stop()
    timer.clear()
    timer.start()
    alltimers = gather_timers(comm=world_comm)
    if comm.world_rank == 0:
        out = os.path.join(args.work_dir, "timing")
        dump(alltimers, out)
        timer.stop()
        timer.report("toast_benchmark_ground_setup (gathering and dumping timing info)")
    else:
        timer.stop()


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    python_startup_time(rank)
    with toast.mpi.exception_guard(comm=world):
        main()
