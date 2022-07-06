#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Invert a block diagonal covariance matrix.
"""
import argparse
import os
import re
import sys
import traceback

import healpy as hp
import numpy as np

from toast.dist import distribute_uniform
from toast.map import DistPixels, covariance_rcond
from toast.mpi import get_world
from toast.utils import Logger


def main():
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Read a toast covariance matrix and write the inverse condition number map"
    )

    parser.add_argument(
        "--input", required=True, default=None, help="The input covariance FITS file"
    )

    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="The output inverse condition map FITS file.",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    # get options

    infile = args.input
    outfile = None
    if args.output is not None:
        outfile = args.output
    else:
        inmat = re.match(r"(.*)\.fits", infile)
        if inmat is None:
            log.error("input file should have .fits extension")
            return
        inroot = inmat.group(1)
        outfile = "{}_rcond.fits".format(inroot)

    # Get the default communicator
    mpiworld, procs, rank = get_world()

    # We need to read the header to get the size of the matrix.
    # This would be a trivial function call in astropy.fits or
    # fitsio, but we don't want to bring in a whole new dependency
    # just for that.  Instead, we open the file with healpy in memmap
    # mode so that nothing is actually read except the header.

    nside = 0
    nnz = 0
    if rank == 0:
        fake, head = hp.read_map(infile, h=True, memmap=True)
        for key, val in head:
            if key == "NSIDE":
                nside = int(val)
            if key == "TFIELDS":
                nnz = int(val)
    if mpiworld is not None:
        nside = mpiworld.bcast(nside, root=0)
        nnz = mpiworld.bcast(nnz, root=0)

    npix = 12 * nside**2
    subnside = int(nside / 16)
    if subnside == 0:
        subnside = 1
    subnpix = 12 * subnside**2
    nsubmap = int(npix / subnpix)

    # divide the submaps as evenly as possible among processes

    dist = distribute_uniform(nsubmap, procs)
    local = np.arange(dist[rank][0], dist[rank][0] + dist[rank][1])

    if rank == 0:
        if os.path.isfile(outfile):
            os.remove(outfile)

    if mpiworld is not None:
        mpiworld.barrier()

    # create the covariance and inverse condition number map

    cov = DistPixels(
        comm=mpiworld, dtype=np.float64, size=npix, nnz=nnz, submap=subnpix, local=local
    )

    # read the covariance
    log.info("Reading covariance {}".format(infile))
    cov.read_healpix_fits(infile)

    # every process computes its local piece
    log.info("Computing condition number map")
    rcond = covariance_rcond(cov)

    # write the map
    log.info("Writing condition number map")
    rcond.write_healpix_fits(outfile)
    return


if __name__ == "__main__":
    try:
        main()
    except:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None:
            mpiworld.Abort(6)
