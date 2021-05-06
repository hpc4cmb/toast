#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple satellite simulation and makes a map.
The workflow is tailored to the size of the communicator.

total_sample = num_obs * obs_minutes * sample_rate * n_detector
"""

import os
import sys
import traceback
import argparse
import psutil
from datetime import datetime
import numpy as np
from astropy import units as u
import toast
from toast.mpi import MPI
from toast.timing import gather_timers, dump
from toast.instrument_sim import fake_hexagon_focalplane
from toast.schedule_sim_satellite import create_satellite_schedule

# TODO suggest adding this properly to logger ?
def infoMPI(self, com, message):
    """ 
    like `info` but only called if we are in sequential mode on in the rank1 node
    """
    if (com is None) or (com.rank == 0):
        self.info(message)
# adds `infoMPI` member function to the logger class
toast._libtoast.Logger.infoMPI = infoMPI

def parse_arguments():
    """
    Defines and parses the arguments for the script.
    """        
    # defines the parameters of the script
    parser = argparse.ArgumentParser(description="Run a TOAST satellite workflow scaled appropriately to the MPI communicator size and available memory.")
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
        default='auto',
        choices=['auto', 'tiny', 'xsmall', 'small', 'medium', 'large', 'xlarge', 'heroic'],
        type=str,
        help="Size of the worflow to be run: 'tiny' (1GB), 'xsmall' (10GB), 'small' (100GB), 'medium' (1TB), 'large' (10TB), 'xlarge' (100TB), 'heroic' (1000TB) or 'auto' (deduced from MPI parameters).",
    )

    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
    operators = [
        toast.ops.SimSatellite(name="sim_satellite"),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.PointingDetectorSimple(name="det_pointing"),
        toast.ops.PointingHealpix(name="pointing", mode="IQU"),
        toast.ops.BinMap(name="binner", pixel_dist="pix_dist"),
        toast.ops.MapMaker(name="mapmaker"),
        toast.ops.PointingHealpix(name="pointing_final", enabled=False, mode="IQU"),
        toast.ops.BinMap(name="binner_final", enabled=False, pixel_dist="pix_dist_final"),
    ]

    # Templates we want to configure from the command line or a parameter file.
    templates = [toast.templates.Offset(name="baselines")]

    # generates config
    config, args, jobargs = toast.parse_config(parser, operators=operators, templates=templates)

    # hardcoded arguments
    # focal plane
    args.sample_rate = 100 # sample_rate is nb sample per second, we do (60 * obs_minutes * sample_rate) sample for one observation of one detector in a minute
    args.max_detector = 2054 # Hex-packed 1027 pixels (18 rings) times two dets per pixel.
    args.obs_minutes = 60
    args.psd_net = 50.0e-6
    args.psd_fmin = 1.0e-5
    # schedule
    args.prec_period = 50.0
    args.spin_period = 10.0

    return config, args, jobargs

def get_node_mem(world_comm, node_rank):
    avail = 2 ** 62
    if node_rank == 0:
        vmem = psutil.virtual_memory()._asdict()
        avail = vmem["available"]
    if world_comm is not None:
        avail = world_comm.allreduce(avail, op=MPI.MIN)
    return int(avail)

def job_size(world_comm, log):
    procs_per_node = 1
    node_rank = 0
    nodecomm = None
    rank = 0
    procs = 1
    if world_comm is not None:
        rank = world_comm.rank
        procs = world_comm.size
        nodecomm = world_comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        node_rank = nodecomm.rank
        procs_per_node = nodecomm.size
        min_per_node = world_comm.allreduce(procs_per_node, op=MPI.MIN)
        max_per_node = world_comm.allreduce(procs_per_node, op=MPI.MAX)
        if min_per_node != max_per_node:
            raise RuntimeError("Nodes have inconsistent numbers of MPI ranks")
    # One process on each node gets available RAM and communicates it
    avail = get_node_mem(world_comm, node_rank)
    n_node = procs // procs_per_node
    log.infoMPI(world_comm, "Job running on {} nodes each with {} processes ({} total)".format(n_node, procs_per_node, procs))
    return (procs_per_node, avail)

def get_mpi_settings(args, log, env):
    """ 
    Getting the MPI settings
    taking the dry_run parameter into account
    """
    # gets actual MPI information
    world_comm, procs, rank = toast.get_world()
    log.infoMPI(world_comm, "TOAST version = {}".format(env.version()))
    log.infoMPI(world_comm, "Using a maximum of {} threads per process".format(env.max_threads()))
    if world_comm is None:
        log.infoMPI(world_comm, "Running serially with one process at {}".format(str(datetime.now())))
    else:
        log.infoMPI(world_comm, "Running with {} processes at {}".format(procs, str(datetime.now())))

    # is this a dry run that does not use MPI
    if args.dry_run is not None:
        procs, procs_per_node = args.dry_run.split(",")
        procs = int(procs)
        procs_per_node = int(procs_per_node)
        log.infoMPI(world_comm, "DRY RUN simulating {} total processes with {} per node".format(procs, procs_per_node))
        # We are simulating the distribution
        avail_node_bytes = get_node_mem(world_comm, 0)
    else:
        # Get information about the actual job size
        procs_per_node, avail_node_bytes = job_size(world_comm, log)

    # sets per node memory
    log.infoMPI(world_comm, "Minimum detected per-node memory available is {:0.2f} GB".format(avail_node_bytes / (1024 ** 3)))
    if args.node_mem_gb is not None:
        avail_node_bytes = int((1024 ** 3) * args.node_mem_gb)
        log.infoMPI(world_comm, "Setting per-node available memory to {:0.2f} GB as requested".format(avail_node_bytes / (1024 ** 3)))

    # computes the total number of nodes
    n_nodes = procs // procs_per_node
    log.infoMPI(world_comm, "Job has {} total nodes".format(n_nodes))

    return world_comm, rank, n_nodes, avail_node_bytes

def select_case(args, n_nodes, avail_node_bytes, world_comm, log):
    """ 
    Selects the most appropriate case size given the memory available and number of nodes
    sets total_samples and n_detector in args
    """
    # availaibles sizes
    cases_samples = {
        "heroic": 5000000000000,  # O(1000) TB RAM
        "xlarge": 500000000000,  # O(100) TB RAM
        "large" : 50000000000,  # O(10) TB RAM
        "medium": 5000000000,  # O(1) TB RAM
        "small" : 500000000,  # O(100) GB RAM
        "xsmall": 50000000,  # O(10) GB RAM
        "tiny"  : 5000000,  # O(1) GB RAM
    }
    if args.case != 'auto':
        cases_samples = {args.case : cases_samples[args.case]}

    # computes the memory that is currently available
    available_memory_bytes = n_nodes * avail_node_bytes

    # tries the workflow sizes from largest to smalest, until we find one that fits
    for (name, total_samples) in cases_samples.items():
        # sets number of samples
        args.case = name
        args.total_samples = total_samples
        # Minimum time span (one day)
        min_time_samples = int(24 * 3600 * args.sample_rate)
        # For the minimum time span, scale up the number of detectors to reach the requested total sample size.
        args.n_detector = min(args.max_detector, total_samples // min_time_samples)

        det_bytes_per_sample = 2 * (  # At most 2 detector data copies.
            8  # 64 bit float / ints used
            * (1 + 4)  # detector timestream  # pixel index and 3 IQU weights
            + 1  # one byte per sample for flags
        )
        common_bytes_per_sample = (
            8 * (4)  # 64 bit floats  # One quaternion per sample
            + 1  # one byte per sample for common flag
        )

        # group_nodes is the number of nodes in each group, 1 minimum
        # can we fit this case in memory while using the minimum number of groups?
        # group_nodes will be grown later if the minimum fits in memory
        group_nodes = 1
        bytes_per_samp = args.n_detector * det_bytes_per_sample + group_nodes * common_bytes_per_sample
        memory_used_bytes = bytes_per_samp * total_samples

        if available_memory_bytes >= memory_used_bytes:
            # search for maximum group node possible
            # it should fit in memory and be a diviser of the number of nodes
            group_nodes += 1
            while (available_memory_bytes >= memory_used_bytes) and (group_nodes < n_nodes):
                while not (n_nodes % group_nodes == 0): group_nodes += 1
                bytes_per_samp = args.n_detector * det_bytes_per_sample + group_nodes * common_bytes_per_sample
                memory_used_bytes = bytes_per_samp * total_samples
            group_nodes -= 1
            break

    log.infoMPI(world_comm, "Distribution using {} total samples, spread over {} groups, and {} detectors ('{}' workflow size)".format(args.total_samples, group_nodes, args.n_detector, args.case))

def make_focalplane(args, world_comm, log):
    """
    Creates a fake focalplane
    """
    # computes the number of pixels to be used
    n_pixel = 1
    ring = 1
    while 2 * n_pixel < args.n_detector:
        n_pixel += 6 * ring
        ring += 1
    log.infoMPI(world_comm, "Using {} hexagon-packed pixels.".format(n_pixel))
    # creates the focalplane
    focalplane = None
    if (world_comm is None) or (world_comm.rank == 0):
        focalplane = fake_hexagon_focalplane(
                            n_pix=n_pixel,
                            sample_rate=args.sample_rate * u.Hz,
                            psd_net=args.psd_net * u.K * np.sqrt(1 * u.second),
                            psd_fmin=args.psd_fmin * u.Hz)
    if world_comm is not None:
        focalplane = world_comm.bcast(focalplane, root=0)
    return focalplane

def make_schedule(args, world_comm, log):
    """
    Creates a satellite schedule
    """
    num_obs = max(1, (args.obs_minutes * args.sample_rate * args.n_detector) // args.total_samples)
    log.infoMPI(world_comm, "Using {} observations produced at {} observation/minute.".format(num_obs, args.obs_minutes))
    # builds the schedule
    schedule = None
    if (world_comm is None) or (world_comm.rank == 0):
        schedule = create_satellite_schedule(
            prefix="",
            mission_start=datetime.now(),
            observation_time=args.obs_minutes * u.minute,
            num_observations=num_obs,
            prec_period=args.prec_period * u.minute,
            spin_period=args.spin_period * u.minute)
    if world_comm is not None:
        schedule = world_comm.bcast(schedule, root=0)
    return schedule

def scan_map(ops, data):
    """ 
    Simulate sky signal from a map.
    We scan the sky with the "final" pointing model if that is different from the solver pointing model.
    """
    pix_dist = toast.ops.BuildPixelDistribution()
    if ops.binner_final.enabled and ops.pointing_final.enabled:
        pix_dist.pixel_dist = ops.binner_final.pixel_dist
        pix_dist.pointing = ops.pointing_final
        pix_dist.shared_flags = ops.binner_final.shared_flags
        pix_dist.shared_flag_mask = ops.binner_final.shared_flag_mask
        pix_dist.save_pointing = ops.binner_final.full_pointing
    else:
        pix_dist.pixel_dist = ops.binner.pixel_dist
        pix_dist.pointing = ops.pointing
        pix_dist.shared_flags = ops.binner.shared_flags
        pix_dist.shared_flag_mask = ops.binner.shared_flag_mask
        pix_dist.save_pointing = ops.binner.full_pointing
    pix_dist.apply(data)

    ops.scan_map.pixel_dist = pix_dist.pixel_dist
    ops.scan_map.pointing = pix_dist.pointing
    ops.scan_map.apply(data)

def run_mapmaker(ops, args, tmpls, data):
    """ 
    Build up our map-making operation from the pieces- both operators configured from user options and other operators.
    """
    ops.binner.pointing = ops.pointing
    ops.binner.noise_model = ops.default_model.noise_model

    final_bin = None
    if ops.binner_final.enabled:
        final_bin = ops.binner_final
        if ops.pointing_final.enabled:
            final_bin.pointing = ops.pointing_final
        else:
            final_bin.pointing = ops.pointing
        final_bin.noise_model = ops.default_model.noise_model

    # A simple binned map will be made if an empty list of templates is passed to the mapmaker.
    tlist = list()
    if tmpls.baselines.enabled:
        tlist.append(tmpls.baselines)
    tmatrix = toast.ops.TemplateMatrix(templates=tlist)

    ops.mapmaker.binning = ops.binner
    ops.mapmaker.template_matrix = tmatrix
    ops.mapmaker.map_binning = final_bin
    ops.mapmaker.det_data = ops.sim_noise.det_data

    # Run the map making
    ops.mapmaker.apply(data)

    # Write the outputs
    for prod in ["map", "hits", "cov", "rcond"]:
        dkey = "{}_{}".format(ops.mapmaker.name, prod)
        file = os.path.join(args.out_dir, "{}.fits".format(dkey))
        toast.pixels_io.write_healpix_fits(data[dkey], file, nest=ops.pointing.nest)

def compute_science_metric(args, global_timer, n_nodes, rank, log):
    """ 
    Computes the science metric and stores it.
    The metric represents the efficiency of the job in a way that is normalized,
    taking the job size into account
    """
    runtime = global_timer.seconds("toast_benchmark_satellite (science work)")
    prefactor = 1.0e-3
    kilo_samples = 1.0e-3 * args.total_samples
    sample_factor = 1.2
    det_factor = 2.0
    metric = (
        prefactor
        * args.n_detector ** det_factor
        * kilo_samples ** sample_factor
        / (n_nodes * runtime)
    )
    if rank == 0:
        msg = "Science Metric: {:0.1e} * ({:d}**{:0.2f}) * ({:0.3e}**{:0.3f}) / ({:0.1f} * {}) = {:0.2f}".format(
            prefactor,
            args.n_detector,
            det_factor,
            kilo_samples,
            sample_factor,
            runtime,
            n_nodes,
            metric,
        )
        log.info("")
        log.info(msg)
        log.info("")
        with open(os.path.join(args.out_dir, "log"), "a") as f:
            f.write(msg)
            f.write("\n\n")

def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    global_timer = toast.timing.GlobalTimers.get()
    global_timer.start("toast_benchmark_satellite (total)")

    # defines and gets the arguments for the script
    config, args, jobargs = parse_arguments()

    # gets the MPI parameters
    # TODO most of those info do not seem to be used
    world_comm, rank, n_nodes, avail_node_bytes = get_mpi_settings(args, log, env)

    # Log the config
    if rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        outlog = os.path.join(args.out_dir, "config_log.toml")
        toast.config.dump_toml(outlog, config)

    # selects appropriate case size
    select_case(args, n_nodes, avail_node_bytes, world_comm, log)

    # Creates the focalplane file.
    focalplane = make_focalplane(args, world_comm, log)

    # from here on, we start the actual work (unless this is a dry run)
    global_timer.start("toast_benchmark_satellite (science work)")
    if args.dry_run is not None:
        log.infoMPI(world_comm, "Exit from dry run")
        # We are done!
        sys.exit(0)

    # Create a telescope for the simulation.
    site = toast.instrument.SpaceSite("space")
    telescope = toast.instrument.Telescope("satellite", focalplane=focalplane, site=site)

    # Load the schedule file
    schedule = make_schedule(args, world_comm, log)

    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)
    ops = job.operators

    # Find the group size for this job, either from command-line overrides or
    # by estimating the data volume.
    group_size = toast.job_group_size(
        world_comm,
        jobargs,
        schedule=schedule,
        focalplane=focalplane,
        full_pointing=ops.binner.full_pointing,
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

    # Set the pointing matrix operators to use the detector pointing
    ops.pointing.detector_pointing = ops.det_pointing
    ops.pointing_final.detector_pointing = ops.det_pointing

    # Simulate sky signal from a map.
    # TODO reenable by directly instantiating a toast.ops.ScanHealpix class
    # https://github.com/hpc4cmb/toast/blob/fa63ecaec7039377e8c800dd9971170123b47b65/src/toast/unported/pipelines/toast_benchmark.py#L628
    #if ops.scan_map.enabled:
    #    scan_map(ops, data)

    # Simulate detector noise
    if ops.sim_noise.enabled:
        ops.sim_noise.apply(data)

    # Build up our map-making operation.
    if ops.mapmaker.enabled:
        run_mapmaker(ops, args, job.templates, data)

    # end of the computations, sync and computes efficiency
    global_timer.stop_all()
    log.infoMPI(world_comm, "Gathering benchmarking metrics.")
    if world_comm is not None:
        world_comm.barrier()
    compute_science_metric(args, global_timer, n_nodes, rank, log)

    # dumps all the timing information
    timer = toast.timing.GlobalTimers.get()
    timer.start("toast_benchmark_satellite (gathering and dumping timing info)")
    alltimers = gather_timers(comm=world_comm)
    if comm.world_rank == 0:
        out = os.path.join(args.out_dir, "timing")
        dump(alltimers, out)
        with open(os.path.join(args.out_dir, "log"), "a") as f:
            f.write("Copy of Global Timers:\n")
            with open("{}.csv".format(out), "r") as t:
                f.write(t.read())
        timer.stop("toast_benchmark_satellite (gathering and dumping timing info)")
        timer.report()

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
