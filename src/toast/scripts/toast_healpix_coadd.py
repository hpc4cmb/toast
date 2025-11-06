#!/usr/bin/env python3

# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
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
from toast.pixels_io_healpix import read_healpix, write_healpix
from toast.timing import Timer
from toast.utils import Environment, Logger


def load_map(fname, prefix=None):
    if prefix is not None:
        log = Logger.get()
        log.info(f"{prefix}Loading {fname}")
    try:
        m = read_healpix(fname, None, nest=True, dtype=float, verbose=False)
    except Exception as e:
        msg = f"Failed to load HEALPix map: {e}"
        raise RuntimeError(msg)
    m = np.atleast_2d(m)
    return m


def main(opts=None, comm=None):
    env = Environment.get()
    log = Logger.get()

    # Get optional MPI parameters
    ntask = 1
    rank = 0
    if comm is not None:
        ntask = comm.size
        rank = comm.rank

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
        "--hits",
        required=False,
        default=None,
        help="Name of output hits file",
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

    args = parser.parse_args(args=opts)

    if args.double_precision:
        dtype = np.float64
    else:
        dtype = np.float32

    noiseweighted_sum = None
    invcov_sum = None
    hits_sum = None
    nnz, nnz2, npix = None, None, None
    hit_pixels = None
    if args.hits is None:
        have_hits = False
    else:
        have_hits = True
    if len(args.inmap) == 1:
        # Only one file provided, try interpreting it as a text file with a list
        try:
            weights = dict()
            infiles = list()
            with open(args.inmap[0], "r") as listfile:
                raw = listfile.readlines()
            for line in raw:
                flds = line.split()
                infiles.append(flds[0])
                if len(flds) > 1:
                    weights[flds[0]] = float(flds[1])
                else:
                    weights[flds[0]] = 1.0
            log.info_rank(f"Loaded {args.inmap[0]} in", timer=timer1, comm=comm)
        except UnicodeDecodeError:
            # Didn't work. Assume that user supplied a single map file
            infiles = args.inmap
            weights = {infiles[0]: 1.0}
    else:
        infiles = args.inmap
        weights = {x: 1.0 for x in infiles}
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
        inmap = load_map(infile_map)
        log.info_rank(
            f"{prefix}Loaded {infile_map} {inmap.shape} in", timer=timer1, comm=None
        )
        if nnz is None:
            nnz, npix = inmap.shape
            hit_pixels = np.zeros(npix, dtype=bool)
        else:
            nnz_test, npix_test = inmap.shape
            if nnz != nnz_test:
                raise RuntimeError(f"Mismatch in nnz: {nnz} != {nnz_test}")
            if npix != npix_test:
                raise RuntimeError(f"Mismatch in npix: {npix} != {npix_test}")

        # Did we load a binned or noise-weighted map?
        noiseweighted = "noiseweighted" in infile_map

        # Determine the name of the covariance matrix file
        if "unfiltered" in infile_map:
            mapstring = "unfiltered_"
        elif "filtered" in infile_map:
            mapstring = "filtered_"
        else:
            mapstring = ""
        if "binmap" in infile_map:
            mapstring += "binmap"
        else:
            mapstring += "map"
        if noiseweighted:
            mapstring = f"noiseweighted_{mapstring}"

        infile_invcov = infile_map.replace(f"_{mapstring}.", "_invcov.")
        if infile_invcov == infile_map:
            raise RuntimeError(
                f"Failed to derive name of a covariance matrix file from {infile_map}."
            )
        infile_hits = infile_map.replace(f"_{mapstring}.", "_hits.")
        if infile_hits == infile_map:
            raise RuntimeError(f"Failed to derive name of hits file from {infile_map}.")

        if os.path.isfile(infile_invcov):
            invcov = load_map(infile_invcov, prefix)
            if nnz2 is None:
                nnz2, npix_test = invcov.shape
            good = invcov[0] != 0
            hit_pixels[good] = True
            ngood = np.sum(good)
            fsky = ngood / good.size
            log.info_rank(
                f"{prefix}Loaded {infile_invcov} {invcov.shape}, fsky = {fsky:.4f} in",
                timer=timer1,
                comm=None,
            )
            invcov = invcov[:, good].copy()
        else:
            # Inverse covariance does not exist. Load and invert the
            # covariance matrix
            log.info(f"{prefix}Inverse covariance not available: {infile_invcov}")
            infile_cov = infile_map.replace(f"_{mapstring}.", "_cov.")
            if not os.path.isfile(infile_cov):
                log.info(f"{prefix}Covariance not available: {infile_cov}")
                msg = (
                    f"Could not find covariance or inverse covariance for {infile_map}"
                )
                raise RuntimeError(msg)
            cov = load_map(infile_cov, prefix)
            if nnz2 is None:
                nnz2, npix_test = cov.shape
            cov_shape = cov.shape
            good = cov[0] != 0
            ngood = np.sum(good)
            hit_pixels[good] = True
            fsky = ngood / good.size
            log.info_rank(
                f"{prefix}Loaded {infile_cov} {cov_shape}, fsky = {fsky:.4f} in",
                timer=timer1,
                comm=None,
            )
            rcond = np.zeros(ngood, dtype=float)
            log.info(f"{prefix}Inverting matrix")
            cov = cov[:, good].T.ravel().copy()
            cov_eigendecompose_diag(ngood, 1, nnz, cov, rcond, args.rcond_limit, True)
            invcov = cov.reshape(ngood, -1).T.copy()
            log.info_rank(
                f"{prefix}Inverted matrix {invcov.shape} in", timer=timer1, comm=None
            )
            del cov

        hits = None
        if have_hits:
            if os.path.isfile(infile_hits):
                hits = load_map(infile_hits, prefix)
                hits = hits[:, good].copy()
            else:
                msg = f"No hits map found for {infile_hits}, disabling "
                msg += "accumulation of hits"
                log.warning(msg)
                have_hits = False

        # Trim off empty pixels
        inmap = inmap[:, good].copy()

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
            cov_apply_diag(ngood, 1, nnz, invcov.data, inmap.data)
            inmap = inmap.reshape(ngood, -1).T.copy()
            invcov = invcov.reshape(ngood, -1).T.copy()
            log.info_rank(f"{prefix}Applied inverse matrix in", timer=timer1, comm=None)

        # Apply per-map weights.  The weights loaded from the file
        # are assumed to be inverse noise weights for each map.
        inmap *= weights[infile_map]
        invcov *= weights[infile_map]

        if noiseweighted_sum is None:
            noiseweighted_sum = np.zeros([nnz, npix], dtype=float)
            invcov_sum = np.zeros([nnz2, npix], dtype=float)
            if have_hits:
                hits_sum = np.zeros([1, npix], dtype=float)

        noiseweighted_sum[:, good] += inmap
        invcov_sum[:, good] += invcov
        if have_hits:
            hits_sum[:, good] += hits
        log.info_rank(f"{prefix}Co-added maps in", timer=timer1, comm=None)

        del hits
        del invcov
        del inmap

    log.info_rank("Processed inputs in", timer=timer, comm=comm)

    if ntask != 1:
        nnz = comm.bcast(nnz)
        nnz2 = comm.bcast(nnz2)
        npix = comm.bcast(npix)
        if hit_pixels is None:
            hit_pixels = np.zeros(npix, dtype=bool)
        comm.Allreduce(MPI.IN_PLACE, hit_pixels, op=MPI.LOR)
        good = hit_pixels
        ngood = np.sum(hit_pixels)
        fsky = ngood / npix
        if noiseweighted_sum is None:
            noiseweighted_sum = np.zeros([nnz, ngood], dtype=float)
            invcov_sum = np.zeros([nnz2, ngood], dtype=float)
            if have_hits:
                hits_sum = np.zeros([1, ngood], dtype=float)
        else:
            noiseweighted_sum = noiseweighted_sum[:, good].copy()
            invcov_sum = invcov_sum[:, good].copy()
            if have_hits:
                hits_sum = hits_sum[:, good].copy()
        comm.Allreduce(MPI.IN_PLACE, noiseweighted_sum, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, invcov_sum, op=MPI.SUM)
        if have_hits:
            comm.Allreduce(MPI.IN_PLACE, hits_sum, op=MPI.SUM)
        log.info_rank("Reduced inputs in", timer=timer, comm=comm)
    else:
        good = hit_pixels
        ngood = np.sum(hit_pixels)
        noiseweighted_sum = noiseweighted_sum[:, good]
        invcov_sum = invcov_sum[:, good]
        if have_hits:
            hits_sum = hits_sum[:, good]
    fsky = ngood / npix
    log.info_rank(f"fsky = {fsky:.4f}", comm=comm)

    if args.invcov is not None:
        log.info_rank(f"Writing {args.invcov}", comm=comm)
        if rank == 0:
            full_invcov = np.zeros([nnz2, npix])
            full_invcov[:, good] = invcov_sum
            write_healpix(
                args.invcov, full_invcov, nest=True, overwrite=True, dtype=dtype
            )
            del full_invcov
        log.info_rank(f"Wrote {args.invcov} in", timer=timer, comm=comm)

    if have_hits:
        log.info_rank(f"Writing {args.hits}", comm=comm)
        if rank == 0:
            full_hits = np.zeros([1, npix])
            full_hits[:, good] = hits_sum
            write_healpix(args.hits, full_hits, nest=True, overwrite=True, dtype=dtype)
            del full_hits
        log.info_rank(f"Wrote {args.hits} in", timer=timer, comm=comm)

    # Each task processes a segment of hit pixels

    npix_task = ngood // ntask + 1
    first_pix = rank * npix_task
    last_pix = min(first_pix + npix_task, ngood)
    if first_pix < last_pix:
        my_npix = last_pix - first_pix
        ind = slice(first_pix, last_pix)
        my_map = noiseweighted_sum[:, ind].T.ravel().copy()
        my_cov = invcov_sum[:, ind].T.ravel().copy()
        my_rcond = np.zeros(my_npix, dtype=float)
        log.debug(f"{prefix}Inverting {my_npix} pixels")
        cov_eigendecompose_diag(
            my_npix, 1, nnz, my_cov, my_rcond, args.rcond_limit, True
        )
        log.debug(f"{prefix}Multiplying {my_npix} pixels")
        cov_apply_diag(my_npix, 1, nnz, my_cov.data, my_map.data)
        my_map = my_map.reshape(my_npix, -1).T.copy()
        my_cov = my_cov.reshape(my_npix, -1).T.copy()
    else:
        my_map = np.zeros([nnz, 0], dtype=float)
        my_cov = np.zeros([nnz2, 0], dtype=float)
        my_rcond = np.zeros([0], dtype=float)

    log.info_rank("Inverted and applied covariance in", timer=timer, comm=comm)

    # Gather to root process and write

    if comm is None:
        total_map = [my_map]
        total_cov = [my_cov]
        total_rcond = [my_rcond]
    else:
        total_map = comm.gather(my_map, root=0)
        total_cov = comm.gather(my_cov, root=0)
        total_rcond = comm.gather(my_rcond, root=0)
        log.info_rank("Gathered map and covariance in", timer=timer, comm=comm)

    if rank == 0:
        if args.double_precision:
            dtype = np.float64
        else:
            dtype = np.float32
        if args.rcond is not None:
            log.info(f"Writing {args.rcond}")
            total_rcond = np.hstack(total_rcond)
            full_rcond = np.zeros(npix, dtype=dtype)
            full_rcond[good] = total_rcond
            write_healpix(
                args.rcond, full_rcond, nest=True, dtype=dtype, overwrite=True
            )
            del full_rcond
            del total_rcond
            log.info_rank(f"Wrote {args.rcond}", timer=timer, comm=None)
        if args.cov is not None:
            log.info(f"Writing {args.cov}")
            total_cov = np.hstack(total_cov)
            full_cov = np.zeros([nnz2, npix])
            full_cov[:, good] = total_cov
            write_healpix(args.cov, full_cov, nest=True, dtype=dtype, overwrite=True)
            del full_cov
            del total_cov
            log.info_rank(f"Wrote {args.cov}", timer=timer, comm=None)
        log.info(f"Writing {args.outmap}")
        total_map = np.hstack(total_map)
        full_map = np.zeros([nnz, npix])
        full_map[:, good] = total_map
        write_healpix(args.outmap, full_map, nest=True, dtype=dtype, overwrite=True)
        log.info_rank(f"Wrote {args.outmap}", timer=timer, comm=None)

    log.info_rank("Co-add done in", timer=timer0, comm=comm)

    if comm is not None:
        comm.Barrier()

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(comm=world)
