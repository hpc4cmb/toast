#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a simple satellite simulation and makes a map.
The workflow is tailored to the size of the communicator.
TODO remove tests where we will always take the same branch?
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

def define_arguments():
    """
    Parses the arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Satellite Simulation Example.")
    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="toast_sim_satellite_out",
        help="The output directory",
    )
    return parser

def make_focalplane(min_pix, rank, world_comm, log):
    """
    Creates a fake focalplane
    """
    focalplane = None
    # computes the number of pixels to be used
    npix = 1
    ring = 1
    while npix < min_pix:
        npix += 6 * ring
        ring += 1
    # creates the focalplane
    if rank == 0:
        msg = "Using {} hexagon-packed pixels, which is >= requested number ({})".format(npix, min_pix)
        log.info(msg)
        # TODO we use some non-default parameters from fake, should we use the defaults instead?
        focalplane = fake_hexagon_focalplane(
                            n_pix=npix,
                            sample_rate=50.0 * u.Hz,
                            psd_net=50.0e-6 * u.K * np.sqrt(1 * u.second),
                            psd_fmin=1.0e-5 * u.Hz)
    if world_comm is not None:
        focalplane = world_comm.bcast(focalplane, root=0)
    return focalplane

def make_schedule(obs_minutes, num_obs, rank, world_comm):
    """
    Creates a satellite schedule
    """
    schedule = None
    if rank == 0:
        # TODO we use some non-default parameters from fake, should we use the defaults instead?
        schedule = create_satellite_schedule(
            prefix="",
            mission_start=datetime.now(),
            observation_time=obs_minutes * u.minute,
            num_observations=num_obs,
            prec_period=50.0 * u.minute,
            spin_period=10.0 * u.minute,
        )
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

def run_mapmaker(ops, tmpls, data):
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
    # TODO do we run that in the bench?
    #for prod in ["map", "hits", "cov", "rcond"]:
    #    dkey = "{}_{}".format(ops.mapmaker.name, prod)
    #    file = os.path.join(args.out_dir, "{}.fits".format(dkey))
    #    toast.pixels_io.write_healpix_fits(data[dkey], file, nest=ops.pointing.nest)

def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_satellite_sim (total)")

    # Get optional MPI parameters
    world_comm, procs, rank = toast.get_world()
    
    # The operators we want to configure from the command line or a parameter file.
    # We will use other operators, but these are the ones that the user can configure.
    # The "name" of each operator instance controls what the commandline and config
    # file options will be called.
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
        toast.ops.BinMap(name="binner_final", enabled=False, pixel_dist="pix_dist_final"),
    ]

    # Templates we want to configure from the command line or a parameter file.
    templates = [toast.templates.Offset(name="baselines")]

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.
    parser = define_arguments()
    config, args, jobargs = toast.parse_config(
        parser,
        operators=operators,
        templates=templates,
    )

    # Log the config that was actually used at runtime.
    if rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        outlog = os.path.join(args.out_dir, "config_log.toml")
        toast.config.dump_toml(outlog, config)

    # Creates the focalplane file.
    minpix = 16 # TODO
    focalplane = make_focalplane(minpix, rank, world_comm, log)

    # Create a telescope for the simulation.  Again, for a specific experiment we
    # would use custom classes for the site.
    site = toast.instrument.SpaceSite("space")
    telescope = toast.instrument.Telescope("satellite", focalplane=focalplane, site=site)

    # Load the schedule file
    obs_minutes = 10 # TODO
    num_obs = 1 # TODO
    schedule = make_schedule(obs_minutes, num_obs, rank, world_comm)

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
    if ops.scan_map.enabled:
        scan_map(ops, data)

    # Simulate detector noise
    if ops.sim_noise.enabled:
        ops.sim_noise.apply(data)

    # Build up our map-making operation.
    if ops.mapmaker.enabled:
        run_mapmaker(ops, job.templates, data)


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
