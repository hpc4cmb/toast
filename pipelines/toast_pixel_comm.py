#!/usr/bin/env python3

# Copyright (c) 2020-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
Distributed map communication tests.
"""

import os
import sys

import argparse

import traceback

import numpy as np

import healpy as hp

import toast

from toast.mpi import get_world, Comm

from toast.dist import Data

from toast.utils import Logger, Environment

from toast.timing import Timer, GlobalTimers, gather_timers

from toast.timing import dump as dump_timing

from toast import dump_config, parse_config, create

from toast.pixels import PixelDistribution, PixelData

from toast.pixels_io import write_healpix_fits

from toast import future_ops as ops

from toast.future_ops.sim_focalplane import fake_hexagon_focalplane

from toast.instrument import Telescope


def main():
    env = Environment.get()
    log = Logger.get()

    gt = GlobalTimers.get()
    gt.start("toast_pixel_comm (total)")

    mpiworld, procs, rank = get_world()

    # The operators used in this script:
    operators = {"sim_satellite": ops.SimSatellite, "pointing": ops.PointingHealpix}

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="TOAST distributed map communication tests."
    )

    parser.add_argument(
        "--focalplane_pixels",
        required=False,
        type=int,
        default=1,
        help="Number of focalplane pixels",
    )

    parser.add_argument(
        "--group_size",
        required=False,
        type=int,
        default=procs,
        help="Size of a process group assigned to an observation",
    )

    parser.add_argument(
        "--comm_mb",
        required=False,
        type=int,
        default=10,
        help="Size in MB of allreduce buffer",
    )

    config, argvars = parse_config(parser, operators=operators)

    # Communicator
    comm = Comm(world=mpiworld, groupsize=argvars["group_size"])

    # Make a fake focalplane and telescope
    focalplane = fake_hexagon_focalplane(
        argvars["focalplane_pixels"],
        10.0,
        samplerate=10.0,
        epsilon=0.0,
        net=1.0,
        fmin=1.0e-5,
        alpha=1.0,
        fknee=0.05,
    )

    config["operators"]["sim_satellite"]["telescope"] = Telescope(
        name="fake", focalplane=focalplane
    )

    # Specify where to store the pixel distribution
    config["operators"]["pointing"]["create_dist"] = "pixel_dist"

    # Log the config that was actually used at runtime.
    out = "pixel_comm_config_log.toml"
    dump_config(out, config)

    # Instantiate our operators
    run = create(config)

    # Put our operators into a pipeline running all detectors at once.
    pipe_opts = ops.Pipeline.defaults()
    pipe_opts["detector_sets"] = "ALL"
    pipe_opts["operators"] = [
        run["operators"][x] for x in ["sim_satellite", "pointing"]
    ]

    pipe = ops.Pipeline(pipe_opts)

    # Start with empty data
    data = toast.Data(comm=comm)

    # Run the pipeline
    pipe.exec(data)
    pipe.finalize(data)

    # print(data)

    # Get the pixel distribution from the Data object
    pixdist = data["pixel_dist"]

    # print(pixdist)

    # Output file root
    outroot = "pixcomm_nproc-{:04d}_gsize-{:04d}_nobs-{:03d}_ndet-{:03d}_nside-{:04d}_nsub-{:03d}".format(
        procs,
        argvars["group_size"],
        config["operators"]["sim_satellite"]["n_observation"],
        2 * argvars["focalplane_pixels"],
        config["operators"]["pointing"]["nside"],
        config["operators"]["pointing"]["nside_submap"],
    )

    # Print out the total hit map and also the hitmap on rank zero.
    hits = PixelData(pixdist, dtype=np.int32, n_value=1)
    hview = hits.raw.array()
    for obs in data.obs:
        for det in obs.local_detectors:
            global_pixels = obs["pixels"][det]
            # We can do this since n_value == 1
            local_pixels = pixdist.global_pixel_to_local(global_pixels)
            hview[local_pixels] += 1

    if rank == 0:
        fhits = hits.storage_class(pixdist.n_pix)
        fview = fhits.array()
        for lc, sm in enumerate(pixdist.local_submaps):
            offset = sm * pixdist.n_pix_submap
            loffset = lc * pixdist.n_pix_submap
            fview[offset : offset + pixdist.n_pix_submap] = hits.raw[
                loffset : loffset + pixdist.n_pix_submap
            ]
        outfile = "{}_hits-rank0.fits".format(outroot)
        if os.path.isfile(outfile):
            os.remove(outfile)
        hp.write_map(
            outfile,
            fview,
            dtype=np.int32,
            fits_IDL=False,
            nest=config["operators"]["pointing"]["nest"],
        )
        del fview
        fhits.clear()
        del fhits

    hits.sync_allreduce()

    outfile = "{}_hits.fits".format(outroot)
    write_healpix_fits(hits, outfile, nest=config["operators"]["pointing"]["nest"])

    # Create some IQU maps with fake local data
    pixdata = PixelData(pixdist, dtype=np.float64, n_value=3)

    # print(pixdata)

    pixdata.raw[:] = np.random.uniform(0.0, 1.0, len(pixdata.raw))

    # Time the different sync techniques

    niter = 20

    allreduce_seconds = None
    alltoallv_seconds = None
    tm = Timer()

    if mpiworld is not None:
        mpiworld.barrier()
    tm.clear()
    tm.start()
    gt.start("SYNC_ALLREDUCE")

    cbytes = argvars["comm_mb"]*1000000
    for i in range(niter):
        pixdata.sync_allreduce(comm_bytes=cbytes)

    if mpiworld is not None:
        mpiworld.barrier()
    tm.stop()
    gt.stop("SYNC_ALLREDUCE")

    allreduce_seconds = tm.seconds() / niter
    msg = "Allreduce average time = {:0.2f} seconds".format(allreduce_seconds)
    if rank == 0:
        print(msg)

    if mpiworld is not None:
        mpiworld.barrier()
    tm.clear()
    tm.start()
    gt.start("SYNC_ALLTOALLV")

    for i in range(niter):
        pixdata.sync_alltoallv()

    if mpiworld is not None:
        mpiworld.barrier()
    tm.stop()
    gt.stop("SYNC_ALLTOALLV")

    alltoallv_seconds = tm.seconds() / niter
    msg = "Alltoallv average time = {:0.2f} seconds".format(alltoallv_seconds)
    if rank == 0:
        print(msg)

    gt.stop_all()
    alltimers = gather_timers(comm=mpiworld)
    if comm.world_rank == 0:
        dump_timing(alltimers, "{}_timing".format(outroot))

    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        if procs == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None:
            mpiworld.Abort()
