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
        "--focalplane", required=False, default=None, help="Input fake focalplane"
    )

    parser.add_argument(
        "--schedule", required=False, default=None, help="Input observing schedule"
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

    # Enabled by default
    operators_enabled = [
        toast.ops.SimSatellite(name="sim_satellite"),
        toast.ops.DefaultNoiseModel(name="default_model"),
        toast.ops.ScanHealpix(name="scan_map"),
        toast.ops.SimNoise(name="sim_noise"),
        toast.ops.PointingHealpix(name="pointing"),
        toast.ops.BinMap(name="binner"),
        toast.ops.MapMaker(name="mapmaker"),
    ]

    # Disabled by default
    operators_disabled = [
        toast.ops.PointingHealpix(name="pointing_final"),
        toast.ops.BinMap(name="binner_final"),
        toast.ops.Madam(name="madam"),
    ]

    # Templates we want to configure from the command line or a parameter file.

    templates = [toast.templates.Offset(name="baselines")]

    # Build a config dictionary starting from the operator defaults, overriding with any
    # config files specified with the '--config' commandline option, followed by any
    # individually specified parameter overrides.

    config, args, jobargs = toast.parse_config(
        parser,
        operators_enabled=operators_enabled,
        operators_disabled=operators_disabled,
        templates_enabled=templates,
    )

    # Log the config that was actually used at runtime.
    outlog = os.path.join(args.out_dir, "config_log.toml")
    if rank == 0:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        toast.config.dump_toml(outlog, config)

    # Load or create the focalplane file.  NOTE:  again, this is just using the
    # built-in Focalplane class.  In a workflow for a specific experiment we would
    # have a custom class.

    focalplane = None
    if args.focalplane is None:
        if rank == 0:
            log.info(
                "No focalplane specified.  Will use 2 fake detectors at the boresight"
            )
        focalplane = toast.instrument_sim.fake_hexagon_focalplane(
            n_pix=1,
            sample_rate=50.0 * u.Hz,
            psd_fmin=1.0e-5 * u.Hz,
            psd_net=1.0 * u.K * np.sqrt(1 * u.second),
            psd_fknee=(50.0 * u.Hz / 2000.0),
        )
    else:
        if rank == 0:
            log.info("Loading focalplane from {}".format(args.focalplane))
            focalplane = Focalplane(file=args.focalplane)
        if world_comm is not None:
            focalplane = world_comm.bcast(focalplane, root=0)

    # Create a telescope for the simulation.  Again, for a specific experiment we
    # would use custom classes for the site.

    site = toast.instrument.SpaceSite("space")
    telescope = toast.instrument.Telescope(
        "satellite", focalplane=focalplane, site=site
    )

    # Load the schedule file

    if args.schedule is None:
        log.info("No schedule file specified- nothing to simulate")
        return
    schedule = None
    if rank == 0:
        schedule = toast.schedule.SatelliteSchedule()
        schedule.read(args.schedule)
    if world_comm is not None:
        schedule = world_comm.bcast(schedule, root=0)

    # Find the group size for this job, either from command-line overrides or
    # by estimating the data volume.

    group_size = toast.job_group_size(
        world_comm,
        jobargs,
        schedule=schedule,
        focalplane=focalplane,
        full_pointing=args.enable_madam,
    )

    # Create the toast communicator

    comm = toast.Comm(world=world_comm, groupsize=group_size)

    # Create the (initially empty) data

    data = toast.Data(comm=comm)

    # Instantiate our objects that were configured from the command line / files

    job = toast.create_from_config(config)
    ops = job.operators
    tmpls = job.templates

    # Simulate the telescope pointing

    ops.sim_satellite.telescope = telescope
    ops.sim_satellite.schedule = schedule
    ops.sim_satellite.apply(data)

    # Construct a "perfect" noise model just from the focalplane parameters

    ops.default_model.apply(data)

    # Simulate sky signal from a map.

    if not args.disable_scan_map:
        ops.scan_map.pointing = ops.pointing
        ops.scan_map.apply(data)

    # Simulate detector noise

    if not args.disable_sim_noise:
        ops.sim_noise.apply(data)

    # Build up our map-making operation from the pieces- both operators configured
    # from user options and other operators.

    if not args.disable_mapmaker:
        ops.binner.pointing = ops.pointing
        ops.binner.noise_model = ops.default_model.noise_model

        final_bin = None
        if args.enable_binner_final:
            final_bin = ops.binner_final
            if args.enable_pointing_final:
                final_bin.pointing = ops.pointing_final
            else:
                final_bin.pointing = ops.pointing
            final_bin.noise_model = ops.default_model.noise_model

        # Note that if an empty list of templates is passed to the mapmaker,
        # then a simple binned map will be made.
        tlist = list()
        if not args.disable_baselines:
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

    # Optionally run Madam
    if args.enable_madam:
        ops.madam.apply(data)


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
