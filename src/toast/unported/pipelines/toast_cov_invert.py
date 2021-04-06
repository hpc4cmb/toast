#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Invert a block diagonal covariance matrix.
"""
import os
import re
import sys
import argparse
import traceback

import numpy as np

import healpy as hp

from toast.mpi import get_world

from toast.utils import Logger

from toast.dist import distribute_uniform

from toast.map import DistPixels, covariance_invert


def main():
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Read a toast covariance matrix and invert it."
    )

    parser.add_argument(
        "--input", required=True, default=None, help="The input covariance FITS file"
    )

    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="The output inverse covariance FITS file.",
    )

    parser.add_argument(
        "--rcond",
        required=False,
        default=None,
        help="Optionally write the inverse condition number map to this file.",
    )

    parser.add_argument(
        "--single",
        required=False,
        default=False,
        action="store_true",
        help="Write the output in single precision.",
    )

    parser.add_argument(
        "--threshold",
        required=False,
        default=1e-3,
        type=np.float,
        help="Reciprocal condition number threshold",
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
        outfile = "{}_inv.fits".format(inroot)

    # Get the default communicator
    mpiworld, procs, rank = get_world()

    # We need to read the header to get the size of the matrix.
    # This would be a trivial function call in astropy.fits or
    # fitsio, but we don't want to bring in a whole new dependency
    # just for that.  Instead, we open the file with healpy in memmap
    # mode so that nothing is actually read except the header.

    nside = 0
    ncovnz = 0
    if rank == 0:
        fake, head = hp.read_map(infile, h=True, memmap=True)
        for key, val in head:
            if key == "NSIDE":
                nside = int(val)
            if key == "TFIELDS":
                ncovnz = int(val)
    if mpiworld is not None:
        nside = mpiworld.bcast(nside, root=0)
        ncovnz = mpiworld.bcast(ncovnz, root=0)

    nnz = int(((np.sqrt(8.0 * ncovnz) - 1.0) / 2.0) + 0.5)

    npix = 12 * nside ** 2
    subnside = int(nside / 16)
    if subnside == 0:
        subnside = 1
    subnpix = 12 * subnside ** 2
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

    cov = None
    invcov = None
    rcond = None

    cov = DistPixels(
        comm=mpiworld,
        dtype=np.float64,
        size=npix,
        nnz=ncovnz,
        submap=subnpix,
        local=local,
    )

    if args.single:
        invcov = DistPixels(
            comm=mpiworld,
            dtype=np.float32,
            size=npix,
            nnz=ncovnz,
            submap=subnpix,
            local=local,
        )
    else:
        invcov = cov

    if args.rcond is not None:
        rcond = DistPixels(
            comm=mpiworld,
            dtype=np.float64,
            size=npix,
            nnz=nnz,
            submap=subnpix,
            local=local,
        )

    # read the covariance
    if rank == 0:
        log.info("Reading covariance from {}".format(infile))
    cov.read_healpix_fits(infile)

    # every process computes its local piece
    if rank == 0:
        log.info("Inverting covariance")
    covariance_invert(cov, args.threshold, rcond=rcond)

    if args.single:
        invcov.data[:] = cov.data.astype(np.float32)

    # write the inverted covariance
    if rank == 0:
        log.info("Writing inverted covariance to {}".format(outfile))
    invcov.write_healpix_fits(outfile)

    # write the condition number

    if args.rcond is not None:
        if rank == 0:
            log.info("Writing condition number map")
        rcond.write_healpix_fits(args.rcond)

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
