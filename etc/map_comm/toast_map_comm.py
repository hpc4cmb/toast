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
    gt.start("toast_map_comm (total)")

    mpiworld, procs, rank = get_world()

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="TOAST distributed map communication tests."
    )

    parser.add_argument(
        "--nside", required=False, type=int, default=256, help="Map NSIDE"
    )

    parser.add_argument(
        "--nside_submap", required=False, type=int, default=16, help="Submap NSIDE"
    )

    parser.add_argument(
        "--comm_mb",
        required=False,
        type=int,
        default=10,
        help="Size in MB of allreduce buffer",
    )

    args = parser.parse_args()

    # Test different types of submap distribution.

    n_pix = 12 * args.nside ** 2
    n_pix_submap = 12 * args.nside_submap ** 2
    n_sub = n_pix // n_pix_submap

    # Tuples are:
    #  1. Fraction of total submaps with full overlap
    #  2. Fraction of total submaps held empty on all procs
    #  3. Fraction of *remaining* submaps to randomly assign

    fractions = [
        #(0.00, 0.0, 0.25),
        (0.00, 0.0, 0.50),
        (0.00, 0.0, 0.75),
        #(0.25, 0.0, 0.00),
        #(0.25, 0.0, 0.25),
        (0.25, 0.0, 0.50),
        #(0.50, 0.0, 0.00),
        (0.50, 0.0, 0.25),
        #(0.50, 0.0, 0.50),
        #(0.75, 0.0, 0.00),
        (0.75, 0.0, 0.25),
        #(0.75, 0.0, 0.50),
        (1.00, 0.0, 0.00),
    ]

    timing_file_root = "mapcomm_nproc-{:04d}_nside-{:04d}_nsub-{:03d}".format(
        procs, args.nside, args.nside_submap
    )
    timing_file = "{}.csv".format(timing_file_root)

    if os.path.isfile(timing_file):
        if rank == 0:
            print(
                "Skipping completed job (n_proc = {}, nside = {}, nsub = {})".format(
                    procs, args.nside, args.nside_submap
                )
            )
        return

    for full, empty, fill in fractions:
        perc_full = int(100 * full)
        perc_empty = int(100 * empty)
        perc_fill = int(100 * fill)

        n_full = int(full * n_sub)
        n_empty = int((n_sub - n_full) * empty)
        n_fill = int((n_sub - n_full - n_empty) * fill)
        fill_start = n_full + n_empty
        local_submaps = [x for x in range(n_full)]
        # print("loc = {}, fill_start = {}".format(local_submaps, fill_start))
        if n_fill > 0:
            rem = n_sub - n_full - n_empty
            flist = [x + fill_start for x in range(rem)]
            n_remove = rem - n_fill
            for nr in range(n_remove):
                select = np.random.randint(0, high=len(flist), size=1, dtype=np.int32)
                del flist[select[0]]
            local_submaps.extend(flist)
        dist = PixelDistribution(
            n_pix=n_pix, n_submap=n_sub, local_submaps=local_submaps, comm=mpiworld
        )
        # print("rank {} submaps: {}".format(rank, dist.local_submaps))

        # Output file root
        outroot = "mapcomm_nproc-{:04d}_nside-{:04d}_nsub-{:03d}_full-{:03d}_empty-{:03d}_fill-{:03d}".format(
            procs, args.nside, args.nside_submap, perc_full, perc_empty, perc_fill
        )

        # Coverage map
        cover = PixelData(dist, np.int32, n_value=1)

        # Set local submaps
        cover.raw[:] = 1

        # Write coverage info
        if rank == 0:
            fcover = cover.storage_class(dist.n_pix)
            fview = fcover.array()
            for lc, sm in enumerate(dist.local_submaps):
                offset = sm * dist.n_pix_submap
                loffset = lc * dist.n_pix_submap
                fview[offset : offset + dist.n_pix_submap] = cover.raw[
                    loffset : loffset + dist.n_pix_submap
                ]
            outfile = "{}_cover-root.fits".format(outroot)
            if os.path.isfile(outfile):
                os.remove(outfile)
            hp.write_map(outfile, fview, dtype=np.int32, fits_IDL=False, nest=True)
            del fview
            fcover.clear()
            del fcover

        cover.sync_allreduce()

        outfile = "{}_cover.fits".format(outroot)
        write_healpix_fits(cover, outfile, nest=True)

        cover.clear()
        del cover

        # Data map for communication
        pix = PixelData(dist, np.float64, n_value=3)

        # Set local submaps
        pix.raw[:] = 1.0

        # Time the different sync techniques
        niter = 5

        allreduce_seconds = None
        alltoallv_seconds = None
        tm = Timer()

        gtname = "SYNC_ALLREDUCE_{}_{}_{}".format(perc_full, perc_empty, perc_fill)

        if mpiworld is not None:
            mpiworld.barrier()
        tm.clear()
        tm.start()
        gt.start(gtname)

        cbytes = args.comm_mb * 1000000
        for i in range(niter):
            pix.sync_allreduce(comm_bytes=cbytes)

        if mpiworld is not None:
            mpiworld.barrier()
        tm.stop()
        gt.stop(gtname)

        allreduce_seconds = tm.seconds() / niter
        msg = "{} / {} / {}:  Allreduce average time = {:0.2f} seconds".format(
            perc_full, perc_empty, perc_fill, allreduce_seconds
        )
        if rank == 0:
            print(msg)

        gtname = "SYNC_ALLTOALLV_{}_{}_{}".format(perc_full, perc_empty, perc_fill)

        if mpiworld is not None:
            mpiworld.barrier()
        tm.clear()
        tm.start()
        gt.start(gtname)

        for i in range(niter):
            pix.sync_alltoallv()

        if mpiworld is not None:
            mpiworld.barrier()
        tm.stop()
        gt.stop(gtname)

        alltoallv_seconds = tm.seconds() / niter
        msg = "{} / {} / {}:  Alltoallv average time = {:0.2f} seconds".format(
            perc_full, perc_empty, perc_fill, alltoallv_seconds
        )
        if rank == 0:
            print(msg)

        pix.clear()
        del pix

    gt.stop_all()
    alltimers = gather_timers(comm=mpiworld)
    if rank == 0:
        dump_timing(
            alltimers,
            "mapcomm_nproc-{:04d}_nside-{:04d}_nsub-{:03d}".format(
                procs, args.nside, args.nside_submap
            ),
        )

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
