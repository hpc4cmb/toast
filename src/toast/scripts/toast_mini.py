#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple satellite simulation and makes a map.
The workflow is tailored to the size of the communicator.

total_sample = num_obs * obs_minutes * sample_rate * n_detector
"""

import argparse
import os
import sys
from datetime import datetime

from astropy import units as u

import toast
import toast.ops
from toast.accelerator.data_localization import display_datamovement
from toast.schedule_sim_satellite import create_satellite_schedule
from toast.scripts.benchmarking_utilities import (
    default_sim_atmosphere,
    estimate_memory_overhead,
    get_mpi_settings,
    make_focalplane,
    python_startup_time,
    select_case,
)
from toast.timing import dump, function_timer, gather_timers


def parse_arguments():
    """
    Defines and parses the arguments for the script.
    """
    # defines the parameters of the script
    parser = argparse.ArgumentParser(
        description="Run a TOAST mini-app for hackathon purposes"
    )
    parser.add_argument(
        "--out_dir",
        required=False,
        default="out_toast_mini",
        type=str,
        help="The output directory",
    )
    parser.add_argument(
        "--node_mem_gb",
        required=False,
        default=None,
        type=float,
        help="Use this much memory per node in GB",
    )
    parser.add_argument(
        "--dry_run",
        required=False,
        default=None,
        type=str,
        help="Comma-separated total_procs,node_procs to simulate.",
    )

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    operators = [
        toast.ops.SimSatellite(
            name="sim_satellite",
            detset_key="pixel",
            hwp_rpm=8.0,
            hwp_angle="hwp_angle",
        ),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.PointingDetectorSimple(name="det_pointing"),
        toast.ops.PixelsHealpix(name="pixels", nside=2048),
        toast.ops.StokesWeights(name="weights", mode="IQU"),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(
            name="mapmaker",
            iter_max=10,
            write_map=True,
            write_noiseweighted_map=False,
            write_hits=True,
            write_cov=False,
            write_invcov=False,
            write_rcond=False,
        ),
        toast.ops.MemoryCounter(name="mem_count"),
    ]

    # Templates we want to configure from the command line or a parameter file.
    templates = [toast.templates.Offset(name="baselines", enabled=True)]

    # generates config
    config, args, jobargs = toast.parse_config(
        parser, operators=operators, templates=templates
    )

    # hardcoded arguments
    # ----------------------
    args.case = "auto"

    # Focalplane
    args.sample_rate = 100
    args.max_detector = (
        2054  # Hex-packed 1027 pixels (18 rings) times two dets per pixel.
    )
    # For debugging:
    # args.max_detector = 4
    args.width = 10
    args.obs_minutes = 60
    args.num_obs = 4380
    args.psd_net = 50.0e-6
    args.psd_fmin = 1.0e-5

    # DEBUGGING
    # args.max_detector = 4
    # args.num_obs = 2

    # Schedule
    args.prec_period = 50.0
    args.spin_period = 10.0

    return config, args, jobargs


def make_schedule(args, world_comm):
    """
    Creates a satellite schedule.
    """
    schedule = None
    if (world_comm is None) or (world_comm.rank == 0):
        schedule = create_satellite_schedule(
            prefix="",
            mission_start=datetime(2022, 1, 1),
            observation_time=args.obs_minutes * u.minute,
            num_observations=args.num_obs,
            prec_period=args.prec_period * u.minute,
            spin_period=args.spin_period * u.minute,
        )
    if world_comm is not None:
        schedule = world_comm.bcast(schedule, root=0)
    args.schedule = schedule


@function_timer
def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()

    global_timer = toast.timing.GlobalTimers.get()
    global_timer.start("toast_mini (total)")

    # Define the arguments for the script
    config, args, jobargs = parse_arguments()

    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)
    job_ops = job.operators

    # Get the MPI parameters
    world_comm, n_procs, rank, n_nodes, avail_node_bytes = get_mpi_settings(
        args, log, env
    )
    toast.utils.system_state(comm=world_comm)

    # Log the config
    if rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        outlog = os.path.join(args.out_dir, "config_log.toml")
        toast.config.dump_toml(outlog, config)

    # Create the schedule, so that we know the total maximum observing time.
    make_schedule(args, world_comm)

    # Estimate per-process memory use from map domain objects and
    # other sources.
    nside_final = None
    overhead_bytes = estimate_memory_overhead(
        n_procs, n_nodes, 0.4, job_ops.pixels.nside, world_comm, nside_final=nside_final
    )

    # Select appropriate data volume.
    select_case(
        args,
        jobargs,
        n_procs,
        n_nodes,
        avail_node_bytes,
        job_ops.binner.full_pointing,
        world_comm,
        per_process_overhead_bytes=overhead_bytes,
    )

    # Creates the focalplane file.
    focalplane = make_focalplane(args, world_comm, log)

    # from here on, we start the actual work (unless this is a dry run)
    if args.dry_run is not None:
        log.info_rank("Exit from dry run.", comm=world_comm)
        # We are done!
        sys.exit(0)

    # Create a telescope for the simulation.
    site = toast.instrument.SpaceSite("space")
    telescope = toast.instrument.Telescope(
        "satellite", focalplane=focalplane, site=site
    )

    # Create the toast communicator
    comm = toast.Comm(world=world_comm, groupsize=args.group_procs)

    # Create the (initially empty) data
    data = toast.Data(comm=comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    # Simulate the telescope pointing
    job_ops.sim_satellite.telescope = telescope
    job_ops.sim_satellite.schedule = args.schedule
    job_ops.sim_satellite.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=world_comm, timer=timer)

    # Construct a "perfect" noise model just from the focalplane parameters
    job_ops.default_model.apply(data)
    log.info_rank("Created default noise model in", comm=world_comm, timer=timer)

    # Set the pointing matrix operators to use the detector pointing
    job_ops.pixels.detector_pointing = job_ops.det_pointing
    job_ops.weights.detector_pointing = job_ops.det_pointing
    job_ops.weights.hwp_angle = job_ops.sim_satellite.hwp_angle

    # Set up the binning operator
    job_ops.binner.pixel_pointing = job_ops.pixels
    job_ops.binner.stokes_weights = job_ops.weights
    job_ops.binner.noise_model = job_ops.default_model.noise_model

    # Simulate detector noise
    job_ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    # Mapmaking
    job_ops.mapmaker.binning = job_ops.binner
    job_ops.mapmaker.template_matrix = toast.ops.TemplateMatrix(
        templates=[job.templates.baselines],
        view=job_ops.pixels.view,
    )
    job_ops.mapmaker.det_data = job_ops.sim_noise.det_data
    job_ops.mapmaker.output_dir = args.out_dir
    job_ops.mapmaker.apply(data)
    log.info_rank("Finished map-making in", comm=world_comm, timer=timer)

    # Dump all the timing information
    timer.stop()
    timer.clear()
    timer.start()
    alltimers = gather_timers(comm=world_comm)
    if comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        dump(alltimers, out)
        timer.stop()
        timer.report("toast_mini (gathering and dumping timing info)")
    else:
        timer.stop()

    # display information on GPU data movement when running in debug/verbose mode
    display_datamovement()


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    python_startup_time(rank)
    with toast.mpi.exception_guard(comm=world):
        main()
