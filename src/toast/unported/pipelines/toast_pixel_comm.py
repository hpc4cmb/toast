#!/usr/bin/env python3

# Copyright (c) 2020-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
Distributed map communication tests.
"""

import argparse
import os
import sys
import traceback

import healpy as hp
import numpy as np
from astropy import units as u

import toast
from toast import future_ops as ops
from toast.config import create, dump_toml, parse_config
from toast.data import Data
from toast.instrument import Telescope
from toast.instrument_sim import fake_hexagon_focalplane
from toast.mpi import Comm, get_world
from toast.pixels import PixelData, PixelDistribution
from toast.pixels_io import write_healpix_fits
from toast.timing import GlobalTimers, Timer
from toast.timing import dump as dump_timing
from toast.timing import gather_timers
from toast.utils import Environment, Logger


def main():
    env = Environment.get()
    log = Logger.get()

    gt = GlobalTimers.get()
    gt.start("toast_pixel_comm (total)")

    mpiworld, procs, rank = get_world()

    # The operators used in this script:
    operators = [
        ops.SimSatellite(name="sim_satellite"),
        ops.PointingHealpix(name="pointing"),
    ]

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

    config, args = parse_config(parser, operators=operators)

    # Make a fake focalplane and telescope
    focalplane = fake_hexagon_focalplane(
        args.focalplane_pixels,
        width=10.0 * u.degree,
        sample_rate=10.0 * u.Hz,
        epsilon=0.0,
        net=1.0,
        f_min=1.0e-5 * u.Hz,
        alpha=1.0,
        f_knee=0.05 * u.Hz,
    )

    # Log the config that was actually used at runtime.
    out = "pixel_comm_config_log.toml"
    if rank == 0:
        dump_toml(out, config)

    # Instantiate our operators
    run = create(config)

    run["operators"]["sim_satellite"].telescope = Telescope(
        name="fake", focalplane=focalplane
    )

    # Specify where to store the pixel distribution
    run["operators"]["pointing"].create_dist = "pixel_dist"

    # Put our operators into a pipeline running all detectors at once.
    pipe = ops.Pipeline(
        detector_sets=["ALL"],
        operators=[run["operators"][x] for x in ["sim_satellite", "pointing"]],
    )

    # Communicator
    comm = Comm(world=mpiworld, groupsize=args.group_size)

    # Start with empty data
    data = toast.Data(comm=comm)

    # Run the pipeline
    pipe.apply(data)

    # print(data)

    # Get the pixel distribution from the Data object
    pixdist = data["pixel_dist"]

    # print(pixdist)

    # Output file root
    outroot = "pixcomm_nproc-{:04d}_gsize-{:04d}_nobs-{:03d}_ndet-{:03d}_nside-{:04d}_nsub-{:03d}".format(
        procs,
        args.group_size,
        run["operators"]["sim_satellite"].num_observations,
        2 * args.focalplane_pixels,
        run["operators"]["pointing"].nside,
        run["operators"]["pointing"].nside_submap,
    )

    # Print out the total hit map and also the hitmap on rank zero.
    hits = PixelData(pixdist, dtype=np.int32, n_value=1)
    hview = hits.raw.array()
    for obs in data.obs:
        for det in obs.local_detectors:
            global_pixels = obs.detdata["pixels"][det]
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
            nest=run["operators"]["pointing"].nest,
        )
        del fview
        fhits.clear()
        del fhits

    hits.sync_allreduce()

    outfile = "{}_hits.fits".format(outroot)
    write_healpix_fits(hits, outfile, nest=run["operators"]["pointing"].nest)

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

    cbytes = args.comm_mb * 1000000
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
