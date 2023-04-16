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

import dateutil
from astropy import units as u

import toast
import toast.ops
from toast.accelerator.data_localization import display_datamovement
from toast.schedule_sim_satellite import create_satellite_schedule
from toast.scripts.benchmarking_utilities import (
    compare_output_stats,
    compute_science_metric,
    estimate_memory_overhead,
    get_mpi_settings,
    make_focalplane,
    python_startup_time,
    run_madam,
    run_mapmaker,
    scan_map,
    select_case,
)
from toast.timing import dump, function_timer, gather_timers


def parse_arguments():
    """
    Defines and parses the arguments for the script.
    """
    # defines the parameters of the script
    parser = argparse.ArgumentParser(
        description="Run a TOAST satellite workflow scaled appropriately to the MPI communicator size and available memory."
    )
    parser.add_argument(
        "--out_dir",
        required=False,
        default="toast_benchmark_satellite_out",
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
    parser.add_argument(
        "--case",
        required=False,
        default="auto",
        choices=[
            "auto",
            "tiny",
            "xsmall",
            "small",
            "medium",
            "large",
            "xlarge",
            "heroic",
        ],
        type=str,
        help="Size of the worflow to be run: 'tiny' (1GB), 'xsmall' (10GB), 'small' (100GB), 'medium' (1TB), 'large' (10TB), 'xlarge' (100TB), 'heroic' (1000TB) or 'auto' (deduced from MPI parameters).",
    )
    parser.add_argument(
        "--print_input_map",
        required=False,
        default=False,
        type=bool,
        help="Should the healpy input map be exported as a PNG for debugging purposes.",
    )
    if toast.ops.madam.available():
        parser.add_argument(
            "--madam",
            required=False,
            default=False,
            action="store_true",
            help="Apply the Madam mapmaker in place of the TOAST mapmaker.",
        )

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    operators = [
        toast.ops.SimSatellite(
            name="sim_satellite",
            detset_key="pixel",
        ),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.ScanHealpixMap(name="scan_map"),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.PointingDetectorSimple(name="det_pointing"),
        toast.ops.PixelsHealpix(name="pixels"),
        toast.ops.StokesWeights(name="weights", mode="IQU"),
        toast.ops.SaveHDF5(name="save_hdf5", enabled=False),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(
            name="mapmaker",
            solve_rcond_threshold=1.0e-3,
            map_rcond_threshold=1.0e-3,
            write_map=True,
            write_noiseweighted_map=False,
            write_hits=True,
            write_cov=False,
            write_invcov=False,
            write_rcond=False,
            keep_final_products=True,
            save_cleaned=True,
            overwrite_cleaned=True,
        ),
        toast.ops.PixelsHealpix(name="pixels_final", enabled=False),
        toast.ops.BinMap(
            name="binner_final", enabled=False, pixel_dist="pix_dist_final"
        ),
        toast.ops.MemoryCounter(name="mem_count"),
    ]

    if toast.ops.madam.available():
        operators.append(toast.ops.Madam(name="madam"))

    # Templates we want to configure from the command line or a parameter file.
    templates = [toast.templates.Offset(name="baselines")]

    # generates config
    config, args, jobargs = toast.parse_config(
        parser, operators=operators, templates=templates
    )

    # hardcoded arguments
    # focal plane
    args.sample_rate = 100  # sample_rate is nb sample per second, we do (60 * obs_minutes * sample_rate) sample for one observation of one detector in a minute
    args.max_detector = (
        2054  # Hex-packed 1027 pixels (18 rings) times two dets per pixel.
    )
    args.width = 10
    args.num_obs = 4320  # 6 months of one-hour observations
    args.obs_minutes = 60
    args.psd_net = 50.0e-6
    args.psd_fmin = 1.0e-5
    # schedule
    args.prec_period = 50.0
    args.spin_period = 10.0
    # scan map
    args.nside = 1024
    args.input_map = f"fake_input_sky_nside{args.nside}.fits"

    return config, args, jobargs


def make_schedule(args, world_comm):
    """
    Creates a satellite schedule
    """
    start_str = "2027-01-01 00:00:00"
    mission_start = dateutil.parser.parse(start_str)
    schedule = None
    if (world_comm is None) or (world_comm.rank == 0):
        schedule = create_satellite_schedule(
            prefix="",
            mission_start=mission_start,
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
    env.enable_function_timers()
    global_timer = toast.timing.GlobalTimers.get()
    global_timer.start("toast_benchmark_satellite (total)")

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
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        outlog = os.path.join(args.out_dir, "config_log.toml")
        toast.config.dump_toml(outlog, config)

    # Create the schedule, so that we know the total maximum observing time.
    make_schedule(args, world_comm)

    # Estimate per-process memory use from map domain objects and
    # other sources.
    nside_final = None
    if job_ops.pixels_final.enabled:
        nside_final = job_ops.pixels_final.nside
    overhead_bytes = estimate_memory_overhead(
        n_procs, n_nodes, 0.4, job_ops.pixels.nside, world_comm, nside_final=nside_final
    )

    # selects appropriate case size
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
    global_timer.start("toast_benchmark_satellite (science work)")
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
    job_ops.pixels.nside = args.nside
    job_ops.pixels.detector_pointing = job_ops.det_pointing
    job_ops.weights.detector_pointing = job_ops.det_pointing
    job_ops.pixels_final.nside = args.nside
    job_ops.pixels_final.detector_pointing = job_ops.det_pointing

    # If we are not using a different pointing matrix for our final binning, then
    # use the same one as the solve.
    job_ops.binner.pixel_pointing = job_ops.pixels
    job_ops.binner.stokes_weights = job_ops.weights
    if not job_ops.pixels_final.enabled:
        job_ops.pixels_final = job_ops.pixels

    # If we are not using a different binner for our final binning, use the same one
    # as the solve.
    job_ops.binner_final.pixel_pointing = job_ops.pixels_final
    job_ops.binner_final.stokes_weights = job_ops.weights
    if not job_ops.binner_final.enabled:
        job_ops.binner_final = job_ops.binner

    # Simulate sky signal from a map.
    scan_map(args, rank, job_ops, data, log)
    log.info_rank("Simulated sky signal in", comm=world_comm, timer=timer)

    # Simulate detector noise
    job_ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    # Optionally save to HDF5 in our output directory
    job_ops.save_hdf5.volume = os.path.join(args.out_dir, "data")
    job_ops.save_hdf5.apply(data)
    log.info_rank("Saved data to HDF5 in", comm=world_comm, timer=timer)

    # If we are running the "tiny" case, our inverse condition number thresholds
    # will effectively cut all the data.  Instead, we lift these requirements
    # just to be able to run something.
    if args.case == "tiny":
        job_ops.mapmaker.solve_rcond_threshold = 1.0e-6
        job_ops.mapmaker.map_rcond_threshold = 1.0e-6

    # Destripe and/or bin TOD
    if toast.ops.madam.available() and args.madam:
        run_madam(job_ops, args, job.templates, data)
    else:
        run_mapmaker(job_ops, args, job.templates, data)
    log.info_rank("Finished map-making in", comm=world_comm, timer=timer)

    # end of the computations, sync and computes efficiency
    global_timer.stop_all()
    log.info_rank("Gathering benchmarking metrics.", comm=world_comm)
    if world_comm is not None:
        world_comm.barrier()
    runtime = global_timer.seconds("toast_benchmark_satellite (science work)")
    compute_science_metric(args, runtime, n_nodes, rank, log)

    # Check values against previously computed ones
    compare_output_stats(
        "satellite", args, rank, log, data["mapmaker_hits"], data["mapmaker_map"]
    )

    # dumps all the timing information
    timer.stop()
    timer.clear()
    timer.start()
    alltimers = gather_timers(comm=world_comm)
    if comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        dump(alltimers, out)
        timer.stop()
        timer.report("toast_benchmark_satellite (gathering and dumping timing info)")
    else:
        timer.stop()

    # display information on GPU data movement when running in debug/verbose mode
    display_datamovement()


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    python_startup_time(rank)
    with toast.mpi.exception_guard(comm=world):
        main()
