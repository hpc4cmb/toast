#!/usr/bin/env python3

# Copyright (c) 2015-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script loads HEALPix maps and noise matrices and calculates
minimum variance averages
"""

import argparse
import os
import re
import sys
import traceback

import h5py
import healpy as hp
import numpy as np
from toast._libtoast import cov_apply_diag, cov_eigendecompose_diag
from toast.covariance import covariance_apply, covariance_invert
from toast.mpi import MPI, Comm, get_world
from toast.pixels_io_healpix import read_healpix, write_healpix
from toast.timing import Timer
from toast.utils import Environment, Logger

import toast


def load_map(fname, prefix="", cache=None, dtype=float):
    log = Logger.get()
    timer = Timer()
    timer.start()
    if cache is not None:
        if fname not in cache:
            msg = prefix + f"Cache was provided but {fname} is not in cache"
            log.error(msg)
        log.info(prefix + f"Loading {fname} from cache")
        m, good, npix = cache[fname]
        if len(m.shape) != 2:
            msg = f"Cached map '{fname}' does not have the right dimensions: {m.shape}"
            raise RuntimeError(msg)
        nmap, ngood = m.shape
    else:
        log.info(prefix + f"Loading {fname}")
        try:
            m = read_healpix(fname, None, nest=True, dtype=dtype)
        except Exception as e:
            msg = prefix + f"Failed to load HEALPix map: {e}"
            raise RuntimeError(msg)
        m = np.atleast_2d(m)
        nmap, npix = m.shape
        good = np.argwhere(m[0] != 0).ravel()
        # Discard empty pixels
        m = m[:, good]
    log.info_rank(prefix + f"Loaded {fname} {m.shape} in", timer=timer, comm=None)
    return m, nmap, npix, good


def find_covariance(infile_map, prefix="", cache=None):
    """Determine the name of the covariance matrix file"""
    log = Logger.get()

    for naming_scheme in range(2):
        if "unfiltered" in infile_map and naming_scheme == 1:
            mapstring = "unfiltered_"
        elif "filtered" in infile_map and naming_scheme == 1:
            mapstring = "filtered_"
        else:
            mapstring = ""
        if "binmap" in infile_map:
            mapstring += "binmap"
        else:
            mapstring += "map"

        pattern = re.compile(".*(_signflip[0-9]{4}).*")
        match = pattern.match(infile_map)
        if match is not None:
            infile_map = infile_map.replace(match.groups()[0], "")

        infile_invcov = infile_map.replace("noiseweighted_", "").replace(
            f"_{mapstring}.", "_invcov."
        )
        infile_cov = infile_map.replace("noiseweighted_", "").replace(
            f"_{mapstring}.", "_cov."
        )
        infile_hits = infile_map.replace("noiseweighted_", "").replace(
            f"_{mapstring}.", "_hits."
        )

        if infile_invcov == infile_map:
            # Try another naming scheme
            continue
        if cache is not None:
            if infile_invcov in cache or infile_cov in cache:
                # This naming scheme worked
                break
        else:
            if os.path.isfile(infile_invcov) or os.path.isfile(infile_cov):
                # This naming scheme worked
                break

    # Confirm success

    if (cache is not None and infile_invcov in cache) or os.path.isfile(infile_invcov):
        log.info(
            prefix + f"Found inverse covariance for {infile_map} : {infile_invcov}"
        )
        infile_cov = None  # No need for this input
    else:
        log.info(prefix + f"No inverse covariance for {infile_map} : {infile_invcov}")
        infile_invcov = None
        if (cache is not None and infile_cov in cache) or os.path.isfile(infile_cov):
            log.info(prefix + f"Found covariance for {infile_map} : {infile_cov}")
        else:
            log.info(prefix + f"No covariance for {infile_map} : {infile_cov}")
            msg = f"No covariance available for {infile_map}"
            raise RuntimeError(msg)

    if (cache is None or infile_hits not in cache) and not os.path.isfile(infile_hits):
        log.info(prefix + f"No hits for {infile_map} : {infile_hits}")
        infile_hits = None

    for fname in infile_invcov, infile_cov, infile_hits:
        if fname is not None and fname == infile_map:
            raise RuntimeError(
                f"Failed to derive name of an auxiliary file from {infile_map}."
            )

    return infile_invcov, infile_cov, infile_hits


def parse_args(opts):
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
        "--outmap_noiseweighted",
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
        "--nside_out",
        required=False,
        default=None,
        type=int,
        help="Output map resolution",
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

    parser.add_argument(
        "--ring",
        required=False,
        default=False,
        action="store_true",
        help="Output in RING ordering (default is NESTED)",
    )

    args = parser.parse_args(args=opts)

    return args


def parse_input_maps(args, comm, weights):
    log = Logger.get()
    timer = Timer()
    timer.start()

    if weights is not None:
        # Ignore command line arguments for input maps, use the provided
        # dictionary instead
        infiles = list(weights.keys())
    elif len(args.inmap) == 1:
        # Only one file provided, try interpreting it as a text file with a list
        try:
            weights = dict()
            infiles = list()
            with open(args.inmap[0], "r") as listfile:
                raw = listfile.readlines()
            if len(raw) == 0:
                msg = f"Did not find any maps listed in {args.inmap[0]}"
                raise RuntimeError(msg)
            for line in raw:
                fields = line.split()
                infiles.append(fields[0])
                if len(fields) == 1:
                    weights[fields[0]] = (1.0, 1.0)
                elif len(fields) == 2:
                    weights[fields[0]] = (float(fields[1]), float(fields[1]))
                elif len(fields) == 3:
                    weights[fields[0]] = (float(fields[1]), float(fields[2]))
                else:
                    msg = f"Failed to parse entry in {args.inmap[0]} : '{line}'"
                    raise RuntimeError(msg)
            log.info_rank(f"Loaded {args.inmap[0]} in", timer=timer, comm=comm)
        except UnicodeDecodeError:
            # Didn't work. Assume that user supplied a single map file
            infiles = args.inmap
            weights = {infiles[0]: (1.0, 1.0)}
    else:
        infiles = args.inmap
        weights = {x: (1.0, 1.0) for x in infiles}

    return infiles, weights


def main(
        opts=None,
        comm=None,
        cache=None,
        result=None,
        prefix=None,
        weights=None,
):
    """Coadd the specified HEALPix maps

    Args:
        comm : MPI communicator to use. If not None, each rank will load
            different maps
        cache (dict) : Cache of preloaded maps. The file path is the key and
            the value is a tuple of nonzero map elements and the indices of
            nonzero elements
        result (dict) : If nonzero, the coadded products are NOT written to
            disk but rather entered in the `result` dictionary with their
            intended paths used as keys
        weights (dict) : dictionary of paths to read and the weights to
             apply to them and their inver covariance. Will override command
             line arguments for input maps.
    """
    env = Environment.get()
    log = Logger.get()

    # Get optional MPI parameters
    ntask = 1
    rank = 0
    if comm is not None:
        ntask = comm.size
        rank = comm.rank

    if prefix is None:
        if ntask == 1:
            prefix = ""
        else:
            prefix = f"{rank:4} : "

    timer0 = Timer()
    timer1 = Timer()
    timer = Timer()
    timer0.start()
    timer1.start()
    timer.start()

    args = parse_args(opts)

    if args.double_precision:
        dtype = np.float64
        dtype_hits = np.int64
    else:
        dtype = np.float32
        dtype_hits = np.int32

    noiseweighted_sum = None
    invcov_sum = None
    hits_sum = None
    nnz, nnz2, nnz3, npix = None, None, None, None
    if args.hits is None:
        have_hits = False
    else:
        have_hits = True

    infiles, weights = parse_input_maps(args, comm, weights)

    nfile = len(infiles)
    for ifile, infile_map in enumerate(infiles):
        infile_map = infile_map.strip()
        if ifile % ntask != rank:
            continue
        log.info(prefix + f"Processing file {ifile + 1} / {nfile}")
        inmap, nnz_test, npix_test, good = load_map(
            infile_map, prefix=prefix, cache=cache, dtype=float,
        )
        if nnz is None:
            nnz = nnz_test
            npix = npix_test
        else:
            if nnz != nnz_test:
                raise RuntimeError(f"Mismatch in nnz: {nnz} != {nnz_test}")
            if npix != npix_test:
                raise RuntimeError(f"Mismatch in npix: {npix} != {npix_test}")

        # Did we load a binned or noise-weighted map?
        noiseweighted = "noiseweighted" in infile_map

        infile_invcov, infile_cov, infile_hits = find_covariance(
            infile_map, prefix=prefix, cache=cache
        )

        if infile_invcov is not None:
            invcov, nnz2, npix_test, good_cov = load_map(
                infile_invcov, prefix=prefix, cache=cache, dtype=float
            )
            ngood = good_cov.size
            fsky = ngood / npix
            log.info_rank(
                prefix + f"Loaded {infile_invcov} {invcov.shape}, fsky = {fsky:.4f} in",
                timer=timer1,
                comm=None,
            )
        else:
            # Inverse covariance does not exist. Load and invert the
            # covariance matrix
            cov, nnz2, npix_test, good_cov = load_map(
                infile_cov, prefix=prefix, cache=cache, dtype=float
            )
            ngood = good_cov.size
            fsky = ngood / npix
            log.info_rank(
                prefix + f"Loaded {infile_cov} {cov.shape}, fsky = {fsky:.4f} in",
                timer=timer1,
                comm=None,
            )
            rcond = np.zeros(ngood, dtype=float)
            log.info(prefix + f"Inverting matrix")
            cov = cov.T.ravel().copy()
            cov_eigendecompose_diag(ngood, 1, nnz, cov, rcond, args.rcond_limit, True)
            invcov = cov.reshape(ngood, -1).T.copy()
            log.info_rank(
                prefix + f"Inverted matrix {invcov.shape} in", timer=timer1, comm=None
            )
            del cov

        if good.size != good_cov.size or np.any(good != good_cov):
            # Inverse covariance can include pixels that fail matrix
            # inversion.  We must discard those to be able to use
            # compressed maps

            keep_cov = np.zeros(good_cov.size, dtype=bool)
            map_set = set(good)
            for i, pix in enumerate(good_cov):
                if pix in map_set:
                    keep_cov[i] = True
            ndiscard = np.sum(np.logical_not(keep_cov))
            if ndiscard != 0:
                msg = f"Discarding {ndiscard} / {good_cov.size} pixels in "
                msg += f"{infile_invcov}/{infile_cov} "
                msg += f"that are not present in {infile_map}"
                log.warning(msg)
            good_cov = good_cov[keep_cov]
            invcov = invcov[:, keep_cov]

            keep_map = np.zeros(good.size, dtype=bool)
            cov_set = set(good_cov)
            for i, pix in enumerate(good):
                if pix in cov_set:
                    keep_map[i] = True
            ndiscard = np.sum(np.logical_not(keep_map))
            if ndiscard != 0:
                msg = f"Discarding {ndiscard} / {good.size} pixels in {infile_map} "
                msg += f"that are not present in {infile_invcov}/{infile_cov}"
                log.warning(msg)
            good = good[keep_map]
            inmap = inmap[:, keep_map]

        if np.any(good != good_cov):
            raise RuntimeError("Map and covariance disagree on nonzeros")

        hits = None
        if have_hits:
            if infile_hits is not None:
                hits, nnz3, npix_test, good_hits = load_map(
                    infile_hits, prefix=prefix, cache=cache, dtype=int
                )
                if good_hits.size != good.size:
                    # Hits can include pixels that fail matrix
                    # inversion.  We must discard those to be able to use
                    # compressed maps
                    keep_hits = np.zeros(good_hits.size, dtype=bool)
                    map_set = set(good)
                    for i, pix in enumerate(good_hits):
                        if pix in map_set:
                            keep_hits[i] = True
                    good_hits = good_hits[keep_hits]
                    hits = hits[keep_hits]
                    raise RuntimeError(msg)
                if np.any(good_hits != good):
                    raise RuntimeError("Map and hits disagree on nonzeros")
            else:
                msg = prefix + f"No hits map found for {infile_map}, disabling "
                msg += "accumulation of hits"
                log.warning(msg)
                have_hits = False

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
            log.info(prefix + f"Applying inverse matrix")
            invcov = invcov.T.ravel().astype(float).copy()
            inmap = inmap.T.ravel().astype(float).copy()
            cov_apply_diag(ngood, 1, nnz, invcov.data, inmap.data)
            inmap = inmap.reshape(ngood, -1).T.copy()
            invcov = invcov.reshape(ngood, -1).T.copy()
            log.info_rank(
                prefix + f"Applied inverse matrix in", timer=timer1, comm=None
            )

        # Apply per-map weights to the noise-weighted map. The
        # additional weights loaded from the file are assumed to be
        # inverse noise weights for each map.
        map_weight, invcov_weight = weights[infile_map]
        inmap *= map_weight
        invcov *= invcov_weight

        if noiseweighted_sum is None:
            noiseweighted_sum = np.zeros([nnz, npix], dtype=float)
            invcov_sum = np.zeros([nnz2, npix], dtype=float)
            if have_hits:
                hits_sum = np.zeros([1, npix], dtype=int)

        noiseweighted_sum[:, good] += inmap
        invcov_sum[:, good] += invcov
        if have_hits:
            hits_sum[:, good] += hits
        log.info_rank(prefix + f"Co-added maps in", timer=timer1, comm=None)

        del hits
        del invcov
        del inmap

    if args.nside_out is not None and noiseweighted_sum is not None:
        # If needed, adjust the co-add resolution.
        nside_in = hp.get_nside(noiseweighted_sum)
        if nside_in < args.nside_out:
            msg = f"Don't know how to increase Nside from {nside_in} to {args.nside_out}"
            raise RuntimeError(msg)
        noiseweighted_sum = hp.ud_grade(
            noiseweighted_sum,
            args.nside_out,
            order_in="NEST",
            order_out="NEST",
            power=-2,
        )
        invcov_sum = hp.ud_grade(
            invcov_sum,
            args.nside_out,
            order_in="NEST",
            order_out="NEST",
            power=-2,
        )
        if have_hits:
            hits_sum = hp.ud_grade(
                hits_sum,
                args.nside_out,
                order_in="NEST",
                order_out="NEST",
                power=-2,
            )
        npix = 12 * args.nside_out**2

    log.info_rank(prefix + "Processed inputs in", timer=timer, comm=comm)

    if invcov_sum is None:
        hit_pixels = None
    else:
        hit_pixels = invcov_sum[0] != 0

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
        log.info_rank(prefix + "Reduced inputs in", timer=timer, comm=comm)
    else:
        good = hit_pixels
        ngood = np.sum(hit_pixels)
        noiseweighted_sum = noiseweighted_sum[:, good]
        invcov_sum = invcov_sum[:, good]
        if have_hits:
            hits_sum = hits_sum[:, good]
    fsky = ngood / npix
    log.info_rank(prefix + f"fsky = {fsky:.4f}", comm=comm)

    if args.outmap_noiseweighted is not None:
        log.info_rank(prefix + f"Writing {args.outmap_noiseweighted}", comm=comm)
        if rank == 0:
            full_noiseweighted = np.zeros([nnz, npix])
            full_noiseweighted[:, good] = noiseweighted_sum
            if args.ring:
                noiseweighted_out = hp.reorder(full_noiseweighted, n2r=True)
                nest = False
            else:
                noiseweighted_out = full_noiseweighted
                nest = True
            if result is None:
                # Write to disk
                write_healpix(
                    args.outmap_noiseweighted,
                    noiseweighted_out,
                    nest=nest,
                    overwrite=True,
                    dtype=dtype,
                )
            else:
                # Write to the result dictionary
                result[args.outmap_noiseweighted] = noiseweighted_out
            del noiseweighted_out
            del full_noiseweighted
        log.info_rank(
            prefix + f"Wrote {args.outmap_noiseweighted} in", timer=timer, comm=comm
        )

    if args.invcov is not None:
        log.info_rank(prefix + f"Writing {args.invcov}", comm=comm)
        if rank == 0:
            full_invcov = np.zeros([nnz2, npix])
            full_invcov[:, good] = invcov_sum
            if args.ring:
                invcov_out = hp.reorder(full_invcov, n2r=True)
                nest = False
            else:
                invcov_out = full_invcov
                nest = True
            if result is None:
                # Write to disk
                write_healpix(
                    args.invcov, invcov_out, nest=nest, overwrite=True, dtype=dtype
                )
            else:
                # Write to the result dictionary
                result[args.invcov] = invcov_out
            del invcov_out
            del full_invcov
        log.info_rank(prefix + f"Wrote {args.invcov} in", timer=timer, comm=comm)

    if have_hits:
        log.info_rank(prefix + f"Writing {args.hits}", comm=comm)
        if rank == 0:
            full_hits = np.zeros([1, npix])
            full_hits[:, good] = hits_sum
            if args.ring:
                hits_out = hp.reorder(full_hits, n2r=True)
                nest = False
            else:
                hits_out = full_hits
                nest = True
            if result is None:
                # Write to disk
                write_healpix(
                    args.hits, hits_out, nest=nest, overwrite=True, dtype=dtype_hits
                )
            else:
                # Write to the result dictionary
                result[args.hits] = hits_out
            del hits_out
            del full_hits
        log.info_rank(prefix + f"Wrote {args.hits} in", timer=timer, comm=comm)

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
        log.debug(prefix + f"Inverting {my_npix} pixels")
        cov_eigendecompose_diag(
            my_npix, 1, nnz, my_cov, my_rcond, args.rcond_limit, True
        )
        log.debug(prefix + f"Multiplying {my_npix} pixels")
        cov_apply_diag(my_npix, 1, nnz, my_cov.data, my_map.data)
        my_map = my_map.reshape(my_npix, -1).T.copy()
        my_cov = my_cov.reshape(my_npix, -1).T.copy()
    else:
        my_map = np.zeros([nnz, 0], dtype=float)
        my_cov = np.zeros([nnz2, 0], dtype=float)
        my_rcond = np.zeros([0], dtype=float)

    log.info_rank(prefix + "Inverted and applied covariance in", timer=timer, comm=comm)

    # Gather to root process and write

    if comm is None:
        total_map = [my_map]
        total_cov = [my_cov]
        total_rcond = [my_rcond]
    else:
        total_map = comm.gather(my_map, root=0)
        total_cov = comm.gather(my_cov, root=0)
        total_rcond = comm.gather(my_rcond, root=0)
        log.info_rank(prefix + "Gathered map and covariance in", timer=timer, comm=comm)

    if rank == 0:
        if args.double_precision:
            dtype = np.float64
        else:
            dtype = np.float32
        if args.rcond is not None:
            log.info(prefix + f"Writing {args.rcond}")
            total_rcond = np.hstack(total_rcond)
            full_rcond = np.zeros(npix, dtype=dtype)
            full_rcond[good] = total_rcond
            del total_rcond
            if args.ring:
                rcond_out = hp.reorder(full_rcond, n2r=True)
                nest = False
            else:
                rcond_out = full_rcond
                nest = True
            if result is None:
                # Write to disk
                write_healpix(
                    args.rcond, rcond_out, nest=nest, dtype=dtype, overwrite=True
                )
            else:
                # Write to the result dictionary
                result[args.rcond] = rcond_out
            del rcond_out
            del full_rcond
            log.info_rank(prefix + f"Wrote {args.rcond}", timer=timer, comm=None)
        if args.cov is not None:
            log.info(prefix + f"Writing {args.cov}")
            total_cov = np.hstack(total_cov)
            full_cov = np.zeros([nnz2, npix])
            full_cov[:, good] = total_cov
            del total_cov
            if args.ring:
                cov_out = hp.reorder(full_cov, n2r=True)
                nest = False
            else:
                cov_out = full_cov
                nest = True
            if result is None:
                # Write to disk
                write_healpix(
                    args.cov, cov_out, nest=nest, dtype=dtype, overwrite=True
                )
            else:
                # Write to the result dictionary
                result[args.cov] = cov_out
            del cov_out
            del full_cov
            log.info_rank(prefix + f"Wrote {args.cov}", timer=timer, comm=None)

        if args.outmap is not None:
            log.info(prefix + f"Writing {args.outmap}")
            total_map = np.hstack(total_map)
            full_map = np.zeros([nnz, npix])
            full_map[:, good] = total_map
            if args.ring:
                map_out = hp.reorder(full_map, n2r=True)
                nest = False
            else:
                map_out = full_map
                nest = True
            if result is None:
                # Write to disk
                write_healpix(args.outmap, map_out, nest=nest, dtype=dtype, overwrite=True)
            else:
                # Write to the result dictionary
                result[args.outmap] = map_out
            del map_out
            del full_map
            log.info_rank(prefix + f"Wrote {args.outmap}", timer=timer, comm=None)

        if result is not None:
            result["nest"] = not args.ring

    log.info_rank(prefix + "Co-add done in", timer=timer0, comm=comm)

    if comm is not None:
        comm.Barrier()

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(comm=world)
