#!/usr/bin/env python3

# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script loads HEALPix maps and noise matrices and calculates
minimum variance averages
"""

import argparse
import os
import sys
import traceback

import h5py
import healpy as hp
import numpy as np

import toast
from toast import PixelData, PixelDistribution
from toast._libtoast import cov_apply_diag, cov_eigendecompose_diag
from toast.covariance import covariance_apply, covariance_invert
from toast.mpi import MPI, Comm, get_world
from toast.pixels_io_healpix import (
    filename_is_fits,
    filename_is_hdf5,
    read_healpix,
    write_healpix,
    write_healpix_fits,
    write_healpix_hdf5,
)
from toast.utils import Environment, Logger, Timer


def main():
    env = Environment.get()
    log = Logger.get()
    comm, ntask, rank = get_world()
    timer0 = Timer()
    timer1 = Timer()
    timer = Timer()
    timer0.start()
    timer1.start()
    timer.start()

    parser = argparse.ArgumentParser(description="Co-add HEALPix maps")

    parser.add_argument(
        "inmap",
        nargs="+",
        help="One or more input maps",
    )

    parser.add_argument(
        "--outmap",
        required=False,
        help="Name of output file",
    )

    parser.add_argument(
        "--rcond",
        required=False,
        help="Name of output rcond file",
    )

    parser.add_argument(
        "--invcov",
        required=False,
        help="Name of output inverse covariance file",
    )

    parser.add_argument(
        "--cov",
        required=False,
        help="Name of output covariance file",
    )

    parser.add_argument(
        "--nside_submap",
        default=16,
        type=int,
        help="Submap size is 12 * nside_submap ** 2.  "
        "Number of submaps is (nside / nside_submap) ** 2",
    )

    parser.add_argument(
        "--rcond_limit",
        default=1e-3,
        type=float,
        help="Reciprocal condition number limit",
    )

    parser.add_argument(
        "--double_precision",
        required=False,
        default=False,
        action="store_true",
        help="Output in double precision",
    )

    parser.add_argument(
        "--scale",
        required=False,
        default=None,
        type=float,
        help="Scale the output map with the provided factor",
    )

    args = parser.parse_args()

    if args.double_precision:
        dtype = np.float64
    else:
        dtype = np.float32

    noiseweighted_sum = None
    invcov_sum = None
    nnz, nnz2, npix = None, None, None
    if len(args.inmap) == 1:
        # Only one file provided, try interpreting it as a text file with a list
        try:
            with open(args.inmap[0], "r") as listfile:
                infiles = listfile.readlines()
            log.info_rank(f"Loaded {args.inmap[0]} in", timer=timer1, comm=comm)
        except UnicodeDecodeError:
            # Didn't work. Assume that user supplied a single map file
            infiles = args.inmap
    else:
        infiles = args.inmap
    nfile = len(infiles)
    for ifile, infile_map in enumerate(infiles):
        infile_map = infile_map.strip()
        if ntask == 1:
            prefix = ""
        else:
            prefix = f"{rank:4} : "
        if ifile % ntask != rank:
            continue
        log.info(f"{prefix}Loading file {ifile + 1} / {nfile} : {infile_map}")
        inmap = read_healpix(infile_map, None, nest=True, dtype=float, verbose=False)
        log.info_rank(f"{prefix}Loaded {infile_map} in", timer=timer1, comm=None)
        if nnz is None:
            nnz, npix = inmap.shape
        else:
            nnz_test, npix_test = inmap.shape
            if nnz != nnz_test:
                raise RuntimeError(f"Mismatch in nnz: {nnz} != {nnz_test}")
            if npix != npix_test:
                raise RuntimeError(f"Mismatch in npix: {npix} != {npix_test}")

        # Did we load a binned or noise-weighted map?
        noiseweighted = "noiseweighted" in infile_map

        # Determine the name of the covariance matrix file
        if "unfiltered_map" in infile_map:
            mapstring = "unfiltered_map"
        elif "filtered_map" in infile_map:
            mapstring = "filtered_map"
        else:
            mapstring = "map"
        if noiseweighted:
            mapstring = f"noiseweighted_{mapstring}"

        infile_invcov = infile_map.replace(mapstring, "invcov")
        if os.path.isfile(infile_invcov):
            log.info(f"{prefix}Loading {infile_invcov}")
            invcov = read_healpix(
                infile_invcov, None, nest=True, dtype=float, verbose=False
            )
            log.info_rank(f"{prefix}Loaded {infile_invcov} in", timer=timer1, comm=None)
        else:
            # Inverse covariance does not exist. Load and invert the
            # covariance matrix
            infile_cov = infile_map.replace(mapstring, "cov")
            if not os.path.isfile(infile_cov):
                msg = (
                    f"Could not find covariance or inverse covariance for {infile_map}"
                )
                raise RuntimeError(msg)
            log.info(f"{prefix}Loading {infile_cov}")
            cov = read_healpix(infile_cov, None, nest=True, dtype=float, verbose=False)
            log.info_rank(f"{prefix}Loaded {infile_cov} in", timer=timer1, comm=None)
            nsubmap = npix
            npix_submap = 1
            rcond = np.zeros(npix, dtype=float)
            log.info(f"{prefix}Inverting matrix")
            cov = cov.T.ravel().astype(float).copy()
            cov_eigendecompose_diag(
                nsubmap, npix_submap, nnz, cov, rcond, args.rcond_limit, True
            )
            invcov = cov.reshape(npix, -1).T.copy()
            log.info_rank(f"{prefix}Inverted matrix in", timer=timer1, comm=None)
            del cov

        # Optionally scale the maps
        if args.scale is not None:
            if noiseweighted:
                inmap /= args.scale
            else:
                inmap *= args.scale
            invcov /= args.scale**2

        if not noiseweighted:
            # Must reverse the multiplication with the
            # white noise covariance matrix
            log.info(f"{prefix}Applying inverse matrix")
            invcov = invcov.T.ravel().astype(float).copy()
            inmap = inmap.T.ravel().astype(float).copy()
            nsubmap = npix
            npix_submap = 1
            cov_apply_diag(nsubmap, npix_submap, nnz, invcov.data, inmap.data)
            inmap = inmap.reshape(npix, -1).T.copy()
            invcov = invcov.reshape(npix, -1).T.copy()
            log.info_rank(f"{prefix}Applied inverse matrix in", timer=timer1, comm=None)

        if nnz2 is None:
            nnz2, npix_test = invcov.shape

        if noiseweighted_sum is None:
            noiseweighted_sum = inmap
            invcov_sum = invcov
        else:
            noiseweighted_sum += inmap
            invcov_sum += invcov
            log.info_rank(f"{prefix}Co-added maps in", timer=timer1, comm=None)
        del invcov
        del inmap

    log.info_rank(f"Processed inputs in", timer=timer, comm=comm)

    if ntask != 1:
        nnz = comm.bcast(nnz)
        nnz2 = comm.bcast(nnz2)
        npix = comm.bcast(npix)
        if noiseweighted_sum is None:
            noiseweighted_sum = np.zeros([nnz, npix], dtype=float)
            invcov_sum = np.zeros([nnz2, npix], dtype=float)
        comm.Allreduce(MPI.IN_PLACE, noiseweighted_sum, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, invcov_sum, op=MPI.SUM)
        log.info_rank(f"Reduced inputs in", timer=timer, comm=comm)

    if args.invcov is not None:
        log.info_rank(f"Writing {args.invcov}", comm=comm)
        if rank == 0:
            write_healpix(
                args.invcov, invcov_sum, nest=True, overwrite=True, dtype=dtype
            )
        log.info_rank(f"Wrote {args.invcov} in", timer=timer, comm=comm)

    # Assign submaps, invert and apply local portions of the matrix

    npix_submap = 12 * args.nside_submap**2
    nsubmap = npix // npix_submap
    local_submaps = [submap for submap in range(nsubmap) if submap % ntask == rank]
    dist = PixelDistribution(
        n_pix=npix, n_submap=nsubmap, local_submaps=local_submaps, comm=comm
    )
    dist_map = PixelData(dist, float, n_value=nnz)
    dist_cov = PixelData(dist, float, n_value=nnz2)
    for local_submap, global_submap in enumerate(local_submaps):
        pix_start = global_submap * npix_submap
        pix_stop = pix_start + npix_submap
        dist_map.data[local_submap] = noiseweighted_sum[:, pix_start:pix_stop].T
        dist_cov.data[local_submap] = invcov_sum[:, pix_start:pix_stop].T
    del noiseweighted_sum
    del invcov_sum

    log.info_rank("Inverting matrix", comm=comm)
    dist_rcond = PixelData(dist, float, n_value=1)
    covariance_invert(dist_cov, args.rcond_limit, rcond=dist_rcond, use_alltoallv=True)
    log.info_rank(f"Inverted matrix in", timer=timer, comm=comm)

    if args.rcond is not None:
        log.info_rank(f"Writing {args.rcond}", comm=comm)
        if filename_is_fits(args.rcond):
            write_healpix_fits(
                dist_rcond,
                args.rcond,
                nest=True,
                single_precision=not args.double_precision,
            )
        else:
            write_healpix_hdf5(
                dist_rcond,
                args.rcond,
                nest=True,
                single_precision=not args.double_precision,
                force_serial=True,
            )
        log.info_rank(f"Wrote {args.rcond}", timer=timer, comm=comm)
    del dist_rcond

    log.info_rank("Applying matrix", comm=comm)
    covariance_apply(dist_cov, dist_map, use_alltoallv=True)
    log.info_rank(f"Applied matrix in", timer=timer, comm=comm)

    if args.cov is not None:
        log.info_rank(f"Writing {args.cov}", comm=comm)
        if filename_is_fits(args.cov):
            write_healpix_fits(
                dist_cov,
                args.cov,
                nest=True,
                single_precision=not args.double_precision,
            )
        else:
            write_healpix_hdf5(
                dist_cov,
                args.cov,
                nest=True,
                single_precision=not args.double_precision,
                force_serial=True,
            )
        log.info_rank(f"Wrote {args.cov}", timer=timer, comm=comm)
    del dist_cov

    log.info_rank(f"Writing {args.outmap}", comm=comm)
    if filename_is_fits(args.outmap):
        write_healpix_fits(
            dist_map,
            args.outmap,
            nest=True,
            single_precision=not args.double_precision,
        )
    else:
        write_healpix_hdf5(
            dist_map,
            args.outmap,
            nest=True,
            single_precision=not args.double_precision,
            force_serial=True,
        )
    log.info_rank(f"Wrote {args.outmap}", timer=timer, comm=comm)

    log.info_rank(f"Co-add done in", timer=timer0, comm=comm)

    if comm is not None:
        comm.Barrier()

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
