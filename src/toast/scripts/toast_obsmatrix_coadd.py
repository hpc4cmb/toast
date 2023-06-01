#!/usr/bin/env python3

# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script co-adds noise-weighted observation matrices and
de-weights the result
"""

import argparse
import os
import sys
import traceback

import h5py
import healpy as hp
import numpy as np
import scipy.sparse

import toast
from toast import PixelData, PixelDistribution
from toast.covariance import covariance_apply, covariance_invert
from toast.mpi import Comm, get_world
from toast.pixels_io_healpix import (
    read_healpix,
    write_healpix_fits,
    write_healpix_hdf5,
    filename_is_fits,
    filename_is_hdf5,
)
from toast.utils import Environment, Logger, Timer


def main():
    env = Environment.get()
    log = Logger.get()
    comm, ntask, rank = get_world()
    timer0 = Timer()
    timer1 = Timer()
    timer0.start()
    timer1.start()

    parser = argparse.ArgumentParser(
        description="Co-add noise-weighted observation matrices and "
        "de-weight the result"
    )

    parser.add_argument(
        "inmatrix",
        nargs="+",
        help="One or more noise-weighted observation matrices",
    )

    parser.add_argument(
        "--outmatrix",
        required=False,
        help="Name of output file",
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

    args = parser.parse_args()

    if args.double_precision:
        dtype = np.float64
    else:
        dtype = np.float32

    if len(args.inmatrix) == 1:
        # Only one file provided, try interpreting it as a text file with a list
        try:
            with open(args.inmatrix[0], "r") as listfile:
                infiles = listfile.readlines()
            log.info_rank(f"Loaded {args.inmatrix[0]} in", timer=timer1, comm=comm)
        except UnicodeDecodeError:
            # Didn't work. Assume that user supplied a single matrix file
            infiles = args.inmatrix
    else:
        infiles = args.inmatrix

    obs_matrix_sum = None
    invcov_sum = None
    nnz = None
    npix = None

    for ifine, infile_matrix in enumerate(infiles):
        infile_matrix = infile_matrix.strip()
        if "noiseweighted" not in infile_matrix:
            msg = f"Observation matrix does not seem to be " \
                  f"noise-weighted: '{infile_matrix}'"
            raise RuntimeError(msg)
        prefix = ""
        log.info(f"{prefix}Loading {infile_matrix}")
        obs_matrix = scipy.sparse.load_npz(infile_matrix)
        if obs_matrix_sum is None:
            obs_matrix_sum = obs_matrix
        else:
            obs_matrix_sum += obs_matrix
        log.info_rank(f"{prefix}Loaded {infile_matrix} in", timer=timer1, comm=None)

        # We'll need the white noise covariance as well
        infile_invcov = infile_matrix.replace("noiseweighted_obs_matrix.npz", "invcov")
        if os.path.isfile(infile_invcov + ".fits"):
            infile_invcov += ".fits"
        elif os.path.isfile(infile_invcov + ".h5"):
            infile_invcov += ".h5"
        else:
            msg = f"Cannot find an inverse covariance matrix to go with '{infile_matrix}'"
            raise RuntimeError(msg)
        log.info(f"{prefix}Loading {infile_invcov}")
        invcov = read_healpix(
            infile_invcov, None, nest=True, dtype=float, verbose=False
        )
        if invcov_sum is None:
            invcov_sum = invcov
            nnzcov, npix = invcov.shape
            nnz = 1
            while (nnz * (nnz + 1)) // 2 != nnzcov:
                nnz += 1
            npixtot = npix * nnz
        else:
            invcov_sum += invcov
        log.info_rank(f"{prefix}Loaded {infile_invcov} in", timer=timer1, comm=None)

    # Put the inverse white noise covariance in a TOAST pixel object

    npix_submap = 12 * args.nside_submap**2
    nsubmap = npix // npix_submap
    local_submaps = [submap for submap in range(nsubmap) if submap % ntask == rank]
    dist = PixelDistribution(
        n_pix=npix, n_submap=nsubmap, local_submaps=local_submaps, comm=comm
    )
    dist_cov = PixelData(dist, float, n_value=nnzcov)
    for local_submap, global_submap in enumerate(local_submaps):
        pix_start = global_submap * npix_submap
        pix_stop = pix_start + npix_submap
        dist_cov.data[local_submap] = invcov_sum[:, pix_start:pix_stop].T
    del invcov_sum

    # Optionally write out the inverse white noise covariance

    if args.invcov is not None:
        log.info_rank(f"Writing {args.invcov}", comm=comm)
        if filename_is_fits(args.invcov):
            write_healpix_fits(
                dist_cov,
                args.invcov,
                nest=True,
                single_precision=not args.double_precision,
            )
        else:
            write_healpix_hdf5(
                dist_cov,
                args.invcov,
                nest=True,
                single_precision=not args.double_precision,
                force_serial=True,
            )
        log.info_rank(f"Wrote {args.invcov}", timer=timer1, comm=comm)

    # Invert the white noise covariance

    log.info_rank("Inverting white noise matrices", comm=comm)
    dist_rcond = PixelData(dist, float, n_value=1)
    covariance_invert(dist_cov, args.rcond_limit, rcond=dist_rcond, use_alltoallv=True)
    log.info_rank(f"Inverted white noise matrices in", timer=timer1, comm=comm)

    # Optionally write out the white noise covariance

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
        log.info_rank(f"Wrote {args.cov} in", timer=timer1, comm=comm)

    # De-weight the observation matrix

    log.info_rank(f"De-weighting obs matrix", comm=comm)
    cc = scipy.sparse.dok_matrix((npixtot, npixtot), dtype=np.float64)
    nsubmap = dist_cov.distribution.n_submap
    npix_submap = dist_cov.distribution.n_pix_submap
    for isubmap_local, isubmap_global in enumerate(
        dist_cov.distribution.local_submaps
    ):
        submap = dist_cov.data[isubmap_local]
        offset = isubmap_global * npix_submap
        for pix_local in range(npix_submap):
            if np.all(submap[pix_local] == 0):
                continue
            pix = pix_local + offset
            icov = 0
            for inz in range(nnz):
                for jnz in range(inz, nnz):
                    cc[pix + inz * npix, pix + jnz * npix] = submap[
                        pix_local, icov
                    ]
                    if inz != jnz:
                        cc[pix + jnz * npix, pix + inz * npix] = submap[
                            pix_local, icov
                        ]
                    icov += 1
    cc = cc.tocsr()
    obs_matrix_sum = cc.dot(obs_matrix_sum)
    log.info_rank(f"De-weighted obs matrix in", timer=timer1, comm=comm)

    # Write out the co-added and de-weighted matrix

    log.info_rank(f"Writing {args.outmatrix}", comm=comm)
    scipy.sparse.save_npz(args.outmatrix, obs_matrix_sum.astype(dtype))
    log.info_rank(f"Wrote {args.outmatrix}.npz in", timer=timer1, comm=comm)

    log.info_rank(
        f"Co-added and de-weighted obs matrix in", timer=timer0, comm=comm
    )

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
