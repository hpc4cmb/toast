#!/usr/bin/env python

import os
if 'TOAST_NO_MPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

import sys
import re
import argparse

import numpy as np

import healpy as hp

import toast
import toast.map as tm


def main():

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("Running with {} processes".format(comm.size))

    global_start = MPI.Wtime()

    parser = argparse.ArgumentParser( description='Read a toast covariance matrix and invert it.' )
    parser.add_argument( '--input', required=True, default=None, help='The input covariance FITS file' )
    parser.add_argument( '--output', required=False, default=None, help='The output inverse covariance FITS file.' )
    parser.add_argument( '--rcond', required=False, default=None, help='Optionally write the inverse condition number map to this file.' )
    parser.add_argument( '--single', required=False, default=False, action='store_true', help='Write the output in single precision.' )
    parser.add_argument( '--threshold', required=False, default=1e-3, type=np.float, help='Reciprocal condition number threshold' )
    
    args = parser.parse_args()

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
        outfile = "{}_inv.fits".format(inroot)

    # We need to read the header to get the size of the matrix.
    # This would be a trivial function call in astropy.fits or
    # fitsio, but we don't want to bring in a whole new dependency
    # just for that.  Instead, we open the file with healpy in memmap
    # mode so that nothing is actually read except the header.

    nside = 0
    ncovnz = 0
    if comm.rank == 0:
        fake, head = hp.read_map(infile, h=True, memmap=True)
        for key, val in head:
            if key == 'NSIDE':
                nside = int(val)
            if key == 'TFIELDS':
                ncovnz = int(val)
    nside = comm.bcast(nside, root=0)
    ncovnz = comm.bcast(nnz, root=0)

    nnz = int( ( (np.sqrt(8.0*ncovnz) - 1.0) / 2.0 ) + 0.5 )

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

    cov = None
    invcov = None
    rcond = None

    cov = tm.DistPixels(comm=comm, dtype=np.float64, size=npix, nnz=ncovnz, submap=subnpix, local=local)
    if args.single:
        invcov = tm.DistPixels(comm=comm, dtype=np.float32, size=npix, nnz=ncovnz, submap=subnpix, local=local)
    else:
        invcov = cov
    if args.rcond is not None:
        rcond = tm.DistPixels(comm=comm, dtype=np.float64, size=npix, nnz=nnz, submap=subnpix, local=local)

    # read the covariance

    cov.read_healpix_fits(infile)

    # every process computes its local piece

    tm.covariance_invert(cov.data, args.threshold, rcond=rcond)

    if args.single:
        invcov.data[:] = cov.data.astype(np.float32)

    # write the inverted covariance

    invcov.write_healpix_fits(outfile)

    # write the condition number

    if args.rcond is not None:
        rcond.write_healpix_fits(args.rcond)

    return



if __name__ == "__main__":
    main()

