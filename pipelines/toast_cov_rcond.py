#!/usr/bin/env python3

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from toast.mpi import MPI, finalize

import os

import sys
import re
import argparse

import numpy as np

import healpy as hp

import toast
import toast.map as tm
import toast.timing as timing


def main():

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("Running with {} processes".format(comm.size))

    global_start = MPI.Wtime()

    parser = argparse.ArgumentParser( description='Read a toast covariance matrix and write the inverse condition number map' )
    parser.add_argument( '--input', required=True, default=None, help='The input covariance FITS file' )
    parser.add_argument( '--output', required=False, default=None, help='The output inverse condition map FITS file.' )

    args = timing.add_arguments_and_parse(parser, timing.FILE(noquotes=True))

    autotimer = timing.auto_timer(timing.FILE())

    # get options

    infile = args.input
    outfile = None
    if args.output is not None:
        outfile = args.output
    else:
        inmat = re.match(r'(.*)\.fits', infile)
        if inmat is None:
            print("input file should have .fits extension")
            sys.exit(0)
        inroot = inmat.group(1)
        outfile = "{}_rcond.fits".format(inroot)

    # We need to read the header to get the size of the matrix.
    # This would be a trivial function call in astropy.fits or
    # fitsio, but we don't want to bring in a whole new dependency
    # just for that.  Instead, we open the file with healpy in memmap
    # mode so that nothing is actually read except the header.

    nside = 0
    nnz = 0
    if comm.rank == 0:
        fake, head = hp.read_map(infile, h=True, memmap=True)
        for key, val in head:
            if key == 'NSIDE':
                nside = int(val)
            if key == 'TFIELDS':
                nnz = int(val)
    nside = comm.bcast(nside, root=0)
    nnz = comm.bcast(nnz, root=0)

    npix = 12 * nside**2
    subnside = int(nside / 16)
    if subnside == 0:
        subnside = 1
    subnpix = 12 * subnside**2
    nsubmap = int( npix / subnpix )

    # divide the submaps as evenly as possible among processes

    dist = toast.distribute_uniform(nsubmap, comm.size)
    local = np.arange(dist[comm.rank][0], dist[comm.rank][0] + dist[comm.rank][1])

    if comm.rank == 0:
        if os.path.isfile(outfile):
            os.remove(outfile)
    comm.barrier()

    # create the covariance and inverse condition number map

    cov = tm.DistPixels(comm=comm, dtype=np.float64, size=npix, nnz=nnz, submap=subnpix, local=local)

    # read the covariance

    cov.read_healpix_fits(infile)

    # every process computes its local piece

    rcond = tm.covariance_rcond(cov)

    # write the map

    rcond.write_healpix_fits(outfile)



if __name__ == "__main__":
    main()
    tman = timing.timing_manager()
    tman.report()
    finalize()
