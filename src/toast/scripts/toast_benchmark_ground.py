#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple ground simulation and makes a map.
The workflow is tailored to the size of the communicator.
"""

import argparse
import os
import shutil
import sys

import numpy as np
from astropy import units as u

import toast
import toast.ops
from toast import spt3g as t3g
from toast.accelerator.data_localization import display_datamovement
from toast.schedule_sim_ground import run_scheduler
from toast.scripts.benchmarking_utilities import (
    compare_output_stats,
    compute_science_metric,
    default_sim_atmosphere,
    estimate_memory_overhead,
    get_mpi_settings,
    get_standard_ground_args,
    make_focalplane,
    python_startup_time,
    run_madam,
    run_mapmaker,
    scan_map,
    select_case,
)
from toast.timing import dump, function_timer, gather_timers

if t3g.available:
    from spt3g import core as c3g


def parse_arguments():
    """
    Defines and parses the arguments for the script.
    """
    # defines the parameters of the script
    parser = argparse.ArgumentParser(
        description="Run a TOAST ground workflow scaled appropriately to the MPI communicator size and available memory."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        default="toast_benchmark_ground_inputs",
        type=str,
        help="The input directory",
    )
    parser.add_argument(
        "--out_dir",
        required=False,
        default="toast_benchmark_ground_out",
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
        "--save_spt3g",
        required=False,
        default=False,
        action="store_true",
        help="Save simulated data to SPT3G format.",
    )
    parser.add_argument(
        "--save_hdf5",
        required=False,
        default=False,
        action="store_true",
        help="Save simulated data to HDF5 format.",
    )
    parser.add_argument(
        "--max_n_det",
        required=False,
        default=None,
        type=int,
        help="Override the maximum number of detectors",
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
        toast.ops.ScanHealpixMap(name="scan_map"),
        toast.ops.SimNoise(name="sim_noise"),
        default_sim_atmosphere(),
        toast.ops.TimeConstant(
            name="convolve_time_constant", deconvolve=False, tau=5 * u.ms
        ),
        toast.ops.PixelsHealpix(name="pixels", view="scanning"),
        toast.ops.StokesWeights(name="weights", mode="IQU"),
        toast.ops.FlagSSO(name="flag_sso"),
        toast.ops.Statistics(name="raw_statistics"),
        toast.ops.TimeConstant(
            name="deconvolve_time_constant",
            deconvolve=True,
            tau=5 * u.ms,
            tau_sigma=0.01,
        ),
        toast.ops.SaveHDF5(name="save_hdf5", enabled=False),
        toast.ops.Statistics(name="filtered_statistics"),
        toast.ops.GroundFilter(name="groundfilter"),
        toast.ops.PolyFilter(name="polyfilter1D"),
        toast.ops.PolyFilter2D(name="polyfilter2D"),
        toast.ops.CommonModeFilter(name="common_mode_filter"),
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
        toast.ops.PixelsHealpix(name="pixels_final", view="scanning", enabled=False),
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

    get_standard_ground_args(args)

    return config, args, jobargs


def load_schedule(args, world_comm):
    """
    Creates a ground schedule
    """
    schedule = None
    if (world_comm is None) or (world_comm.rank == 0):
        schedule_file = os.path.join(args.input_dir, "ground_bench_schedule.txt")
        if not os.path.exists(schedule_file):
            msg = f"Schedule file {schedule_file} does not exist- did you run toast_benchmark_ground_setup?"
            raise RuntimeError(msg)
        schedule = toast.schedule.GroundSchedule()
        schedule.read(schedule_file)
    if world_comm is not None:
        schedule = world_comm.bcast(schedule, root=0)
    args.schedule = schedule


def dump_spt3g(args, job_ops, data):
    """Save data to SPT3G format."""
    if not t3g.available:
        raise RuntimeError("SPT3G is not available, cannot save to that format")
    save_dir = os.path.join(args.out_dir, "data_spt3g")
    meta_exporter = t3g.export_obs_meta(
        noise_models=[
            (job_ops.default_model.noise_model, job_ops.default_model.noise_model),
            (job_ops.elevation_model.out_model, job_ops.elevation_model.out_model),
        ]
    )
    # Note that we export detector flags below to a float64 G3TimestreamMap
    # in order to use FLAC compression.  See documentation for the
    # toast.spt3g.export_obs_data class for a description of the constructor
    # arguments.
    data_exporter = t3g.export_obs_data(
        shared_names=[
            (
                job_ops.sim_ground.boresight_azel,
                job_ops.sim_ground.boresight_azel,
                c3g.G3VectorQuat,
            ),
            (
                job_ops.sim_ground.boresight_radec,
                job_ops.sim_ground.boresight_radec,
                c3g.G3VectorQuat,
            ),
            (job_ops.sim_ground.position, job_ops.sim_ground.position, None),
            (job_ops.sim_ground.velocity, job_ops.sim_ground.velocity, None),
            (job_ops.sim_ground.azimuth, job_ops.sim_ground.azimuth, None),
            (job_ops.sim_ground.elevation, job_ops.sim_ground.elevation, None),
            # (job_ops.sim_ground.hwp_angle, job_ops.sim_ground.hwp_angle, None),
            (job_ops.sim_ground.shared_flags, "telescope_flags", None),
        ],
        det_names=[
            (
                job_ops.sim_noise.det_data,
                job_ops.sim_noise.det_data,
                c3g.G3TimestreamMap,
            ),
            (job_ops.sim_ground.det_flags, "detector_flags", c3g.G3TimestreamMap),
        ],
        interval_names=[
            (job_ops.sim_ground.scan_leftright_interval, "intervals_scan_leftright"),
            (job_ops.sim_ground.turn_leftright_interval, "intervals_turn_leftright"),
            (job_ops.sim_ground.scan_rightleft_interval, "intervals_scan_rightleft"),
            (job_ops.sim_ground.turn_rightleft_interval, "intervals_turn_rightleft"),
            (job_ops.sim_ground.elnod_interval, "intervals_elnod"),
            (job_ops.sim_ground.scanning_interval, "intervals_scanning"),
            (job_ops.sim_ground.turnaround_interval, "intervals_turnaround"),
            (job_ops.sim_ground.sun_up_interval, "intervals_sun_up"),
            (job_ops.sim_ground.sun_close_interval, "intervals_sun_close"),
        ],
        compress=True,
    )
    exporter = t3g.export_obs(
        meta_export=meta_exporter,
        data_export=data_exporter,
        export_rank=0,
    )
    dumper = toast.ops.SaveSpt3g(
        directory=save_dir, framefile_mb=500, obs_export=exporter
    )
    dumper.apply(data)


@function_timer
def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    env.enable_function_timers()
    global_timer = toast.timing.GlobalTimers.get()
    global_timer.start("toast_benchmark_ground (total)")

    # defines and gets the arguments for the script
    config, args, jobargs = parse_arguments()

    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)
    job_ops = job.operators

    # Get the MPI parameters
    world_comm, n_procs, rank, n_nodes, avail_node_bytes = get_mpi_settings(
        args, log, env
    )

    # Check the input map
    args.input_map = os.path.join(
        args.input_dir, f"fake_input_sky_nside{args.nside}.fits"
    )
    if rank == 0 and not os.path.isfile(args.input_map):
        msg = f"Input simulated sky map {args.input_map} does not exist."
        msg += f"  Did you run toast_benchmark_ground_setup?"
        raise RuntimeError(msg)

    # Log the config
    if rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        outlog = os.path.join(args.out_dir, "config_log.toml")
        toast.config.dump_toml(outlog, config)

    # Load the schedule, so that we know the total observing time.
    load_schedule(args, world_comm)

    # Observing site
    site = toast.instrument.GroundSite(
        args.schedule.site_name,
        args.schedule.site_lat,
        args.schedule.site_lon,
        args.schedule.site_alt,
    )

    # Check that we have the atmosphere cache directory set up.
    atm_cache = os.path.join(args.input_dir, "atmosphere")
    if job_ops.sim_atmosphere.enabled:
        if os.path.isdir(atm_cache):
            msg = f"Using cached atmosphere from {atm_cache}."
            job_ops.sim_atmosphere.cache_dir = atm_cache
        else:
            msg = f"Generating atmosphere on the fly."
        log.info_rank(msg, comm=world_comm)

    # Estimate per-process memory use from map domain objects and
    # other sources.
    nside_final = None
    if job_ops.pixels_final.enabled:
        nside_final = job_ops.pixels_final.nside

    # The maximum number of seconds in any scan
    ces_max_time = np.max(
        [(x.stop - x.start).total_seconds() for x in args.schedule.scans]
    )
    overhead_bytes = estimate_memory_overhead(
        n_procs,
        n_nodes,
        0.4,
        job_ops.pixels.nside,
        world_comm,
        nside_final=nside_final,
        sim_atmosphere=job_ops.sim_atmosphere,
        ces_max_time=ces_max_time,
        fov=args.width,
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
    global_timer.start("toast_benchmark_ground (science work)")
    if args.dry_run is not None:
        log.info_rank("Exit from dry run.", comm=world_comm)
        # We are done!
        sys.exit(0)

    # Create a telescope for the simulation.
    telescope = toast.instrument.Telescope(
        args.schedule.telescope_name, focalplane=focalplane, site=site
    )

    # Create the toast communicator
    comm = toast.Comm(world=world_comm, groupsize=args.group_procs)

    # Create the (initially empty) data
    data = toast.Data(comm=comm)

    # Timer for reporting the progress
    timer = toast.timing.Timer()
    timer.start()

    job_ops.mem_count.prefix = "Before Simulation"
    job_ops.mem_count.apply(data)

    # Simulate the telescope pointing
    job_ops.sim_ground.telescope = telescope
    job_ops.sim_ground.schedule = args.schedule
    job_ops.sim_ground.weather = telescope.site.name
    job_ops.sim_ground.apply(data)
    log.info_rank("Simulated telescope pointing in", comm=world_comm, timer=timer)

    job_ops.mem_count.prefix = "After Scan Simulation"
    job_ops.mem_count.apply(data)

    # Set up detector pointing in both Az/El and RA/DEC
    job_ops.det_pointing_azel.boresight = job_ops.sim_ground.boresight_azel
    job_ops.det_pointing_radec.boresight = job_ops.sim_ground.boresight_radec

    # Construct a "perfect" noise model just from the focalplane parameters
    job_ops.default_model.apply(data)
    log.info_rank("Created default noise model in", comm=world_comm, timer=timer)

    # Create the Elevation modulated noise model
    job_ops.elevation_model.noise_model = job_ops.default_model.noise_model
    job_ops.elevation_model.detector_pointing = job_ops.det_pointing_azel
    job_ops.elevation_model.view = job_ops.det_pointing_azel.view
    job_ops.elevation_model.apply(data)
    log.info_rank("Created elevation noise model in", comm=world_comm, timer=timer)

    # Set the pointing matrix operators to use the detector pointing
    job_ops.pixels.nside = args.nside
    job_ops.pixels.detector_pointing = job_ops.det_pointing_radec
    job_ops.weights.detector_pointing = job_ops.det_pointing_radec
    job_ops.pixels_final.nside = args.nside
    job_ops.pixels_final.detector_pointing = job_ops.det_pointing_radec
    job_ops.pixels.view = "scanning"
    job_ops.pixels_final.view = "scanning"

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

    job_ops.mem_count.prefix = "After Noise and Pointing Setup"
    job_ops.mem_count.apply(data)

    # Simulate sky signal from a map.
    args.print_input_map = False
    scan_map(args, rank, job_ops, data, log)
    log.info_rank("Simulated sky signal in", comm=world_comm, timer=timer)

    job_ops.mem_count.prefix = "After Map Scanning"
    job_ops.mem_count.apply(data)

    # Simulate detector noise
    job_ops.sim_noise.noise_model = job_ops.elevation_model.out_model
    job_ops.sim_noise.apply(data)
    log.info_rank("Simulated detector noise in", comm=world_comm, timer=timer)

    job_ops.mem_count.prefix = "After Noise Simulation"
    job_ops.mem_count.apply(data)

    # Simulate atmosphere signal
    job_ops.sim_atmosphere.detector_pointing = job_ops.det_pointing_azel
    job_ops.sim_atmosphere.apply(data)
    log.info_rank("Simulated and observed atmosphere in", comm=world_comm, timer=timer)

    job_ops.mem_count.prefix = "After Atmosphere Simulation"
    job_ops.mem_count.apply(data)

    # Apply a time constant
    job_ops.convolve_time_constant.apply(data)
    log.info_rank("Convolved time constant in", comm=world_comm, timer=timer)

    # Optionally save to HDF5 in our output directory
    job_ops.save_hdf5.volume = os.path.join(args.out_dir, "data")
    job_ops.save_hdf5.apply(data)
    log.info_rank("Saved data to HDF5 in", comm=world_comm, timer=timer)

    # Optionally save data to SPT3G format.  This is expensive and the configuration
    # is specific to this particular workflow, which is why the details are not
    # configurable from the command line / parameter file and why this is not enabled
    # by default.
    if args.save_spt3g:
        dump_spt3g(args, job_ops, data)
        log.info_rank("Saved data to SPT3G format in", comm=world_comm, timer=timer)

    # Collect signal statistics before filtering
    job_ops.raw_statistics.output_dir = args.out_dir
    job_ops.raw_statistics.apply(data)
    log.info_rank("Calculated raw statistics in", comm=world_comm, timer=timer)

    # Deconvolve a time constant
    job_ops.deconvolve_time_constant.apply(data)
    log.info_rank("Deconvolved time constant in", comm=world_comm, timer=timer)

    # Flag Sun, Moon and the planets
    job_ops.flag_sso.detector_pointing = job_ops.det_pointing_azel
    job_ops.flag_sso.apply(data)
    log.info_rank("Flagged Solar system objects in", comm=world_comm, timer=timer)

    # Apply the filter stack
    job_ops.groundfilter.apply(data)
    log.info_rank("Finished ground-filtering in", comm=world_comm, timer=timer)
    job_ops.polyfilter1D.apply(data)
    log.info_rank("Finished 1D-poly-filtering in", comm=world_comm, timer=timer)
    job_ops.polyfilter2D.apply(data)
    log.info_rank("Finished 2D-poly-filtering in", comm=world_comm, timer=timer)
    job_ops.common_mode_filter.apply(data)
    log.info_rank("Finished common-mode-filtering in", comm=world_comm, timer=timer)

    job_ops.mem_count.prefix = "After Filtering"
    job_ops.mem_count.apply(data)

    # Collect signal statistics after filtering
    job_ops.filtered_statistics.output_dir = args.out_dir
    job_ops.filtered_statistics.apply(data)

    # Destripe and/or bin TOD
    if toast.ops.madam.available() and args.madam:
        run_madam(job_ops, args, job.templates, data)
    else:
        run_mapmaker(job_ops, args, job.templates, data)
    log.info_rank("Finished map-making in", comm=world_comm, timer=timer)

    job_ops.mem_count.prefix = "After Map Making"
    job_ops.mem_count.apply(data)

    # end of the computations, sync and computes efficiency
    global_timer.stop_all()
    log.info_rank("Gathering benchmarking metrics.", comm=world_comm)
    if world_comm is not None:
        world_comm.barrier()
    runtime = global_timer.seconds("toast_benchmark_ground (science work)")
    compute_science_metric(args, runtime, n_nodes, rank, log)

    # Check values against previously computed ones
    compare_output_stats(
        "ground", args, rank, log, data["mapmaker_hits"], data["mapmaker_map"]
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
        timer.report("toast_benchmark_ground (gathering and dumping timing info)")
    else:
        timer.stop()

    # display information on GPU data movement when running in debug/verbose mode
    display_datamovement()


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    python_startup_time(rank)
    with toast.mpi.exception_guard(comm=world):
        main()
