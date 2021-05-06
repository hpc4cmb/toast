#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple satellite simulation and makes a map.
The workflow is tailored to the size of the communicator.

total_sample = num_obs * obs_minutes * sample_rate * n_detector
TODO ensure that a dry run is trully dry and does not run stuff
TODO compute science metric
"""

import os
import sys
import traceback
import argparse
from datetime import datetime
import numpy as np
from astropy import units as u
import toast
from toast.mpi import MPI
from toast.instrument_sim import fake_hexagon_focalplane
from toast.schedule_sim_satellite import create_satellite_schedule

# TODO suggest adding it properly to logger ?
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
    # TODO possible to enforce that it is in the list of possible options ?
    parser.add_argument(
        "--case",
        required=False,
        default='auto',
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

    # Log the config
    if rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        outlog = os.path.join(args.out_dir, "config_log.toml")
        toast.config.dump_toml(outlog, config)

    return config, args, jobargs

def get_mpi_settings(args):
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
        avail_node_bytes = get_node_mem(world_comm, 0) # TODO
    else:
        # Get information about the actual job size
        procs_per_node, avail_node_bytes = job_size(mpicomm) # TODO

    # sets per node memory
    log.infoMPI(world_com, "Minimum detected per-node memory available is {:0.2f} GB".format(avail_node_bytes / (1024 ** 3)))
    if args.node_mem_gb is not None:
        avail_node_bytes = int((1024 ** 3) * args.node_mem_gb)
        log.infoMPI(world_com, "Setting per-node available memory to {:0.2f} GB as requested".format(avail_node_bytes / (1024 ** 3)))

    # computes the total number of nodes
    n_nodes = procs // procs_per_node
    log.infoMPI(world_com, "Job has {} total nodes".format(n_nodes))

    return world_comm, procs, rank, n_nodes, avail_node_bytes

def select_case(args, n_nodes):
    """ 
    Selects the most appropriate case size given the MPI parameters
    """
    # availaibles sizes
    cases_samples = {
        "tiny": 5000000,  # O(1) GB RAM
        "xsmall": 50000000,  # O(10) GB RAM
        "small": 500000000,  # O(100) GB RAM
        "medium": 5000000000,  # O(1) TB RAM
        "large": 50000000000,  # O(10) TB RAM
        "xlarge": 500000000000,  # O(100) TB RAM
        "heroic": 5000000000000,  # O(1000) TB RAM
    }
    # finds or detects the case to be used
    case = args.case
    if args.case == 'auto':
        args.case = 'tiny' # TODO select the case as a fucntion of the memory available
        # memory use per node as a function of mn samples
        # how many samples can we fit with our current memory
        # what is the corresponding case?

    args.total_samples = cases_samples[case]
    # Minimum time span (one day)
    min_time_samples = int(24 * 3600 * args.sample_rate)
    # For the minimum time span, scale up the number of detectors to reach the requested total sample size.
    args.n_detector = min(args.max_detector, args.total_samples // min_time_samples)
    # TODO we might do that later, where we need it?
    args.num_obs = max(1, (args.obs_minutes * args.sample_rate * args.n_detector) // total_sample)

def make_focalplane(args, world_comm, log):
    """
    Creates a fake focalplane
    """
    # computes the number of pixels to be used
    # TODO could we generate an approximate number of pixels instead of n_detector or are those the same thing?    
    n_pixel = 1
    ring = 1
    while 2 * n_pixel < args.n_detector:
        n_pixel += 6 * ring
        ring += 1
    log.infoMPI(world_com, "Using {} hexagon-packed pixels.".format(n_pixel))
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

def make_schedule(args, world_comm):
    """
    Creates a satellite schedule
    """
    log.infoMPI(world_com, "Using {} observations produced at {} observation/minute.".format(num_obs, obs_minutes))
    # builds the schedule
    schedule = None
    if (world_comm is None) or (world_comm.rank == 0):
        schedule = create_satellite_schedule(
            prefix="",
            mission_start=datetime.now(),
            observation_time=args.obs_minutes * u.minute,
            num_observations=args.num_obs,
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

def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    global_timer = toast.timing.GlobalTimers.get()
    global_timer.start("toast_benchmark_satellite (total)")

    # defines and gets the arguments for the script
    config, args, jobargs = parse_arguments()

    # gets the MPI parameters
    world_comm, procs, rank, n_nodes, avail_node_bytes = get_mpi_settings(args)

    # selects appropriate case size
    select_case(args)

    # Creates the focalplane file.
    focalplane = make_focalplane(args, world_comm, log)

    # Create a telescope for the simulation.
    site = toast.instrument.SpaceSite("space")
    telescope = toast.instrument.Telescope("satellite", focalplane=focalplane, site=site)

    # Load the schedule file
    schedule = make_schedule(num_obs, rank, world_comm)

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

    # TODO compute science metric


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
