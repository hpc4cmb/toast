# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np
from healpy import nside2npix

from ..timing import function_timer, Timer
from ..utils import Logger, Environment
from ..map import covariance_apply, covariance_invert, DistPixels
from ..todmap import OpSimDipole, OpAccumDiag


def add_binner_args(parser):
    try:
        parser.add_argument(
            "--hits",
            required=False,
            action="store_true",
            help="Write hit maps [default]",
            dest="write_hits",
        )
        parser.add_argument(
            "--no-hits",
            required=False,
            action="store_false",
            help="Do not write hit maps",
            dest="write_hits",
        )
        parser.set_defaults(write_hits=True)
    except argparse.ArgumentError:
        pass

    try:
        parser.add_argument(
            "--wcov",
            required=False,
            action="store_true",
            help="Write white noise covariance [default]",
            dest="write_wcov",
        )
        parser.add_argument(
            "--no-wcov",
            required=False,
            action="store_false",
            help="Do not write white noise covariance",
            dest="write_wcov",
        )
        parser.set_defaults(write_wcov=True)
    except argparse.ArgumentError:
        pass

    try:
        parser.add_argument(
            "--wcov-inv",
            required=False,
            action="store_true",
            help="Write inverse white noise covariance [default]",
            dest="write_wcov_inv",
        )
        parser.add_argument(
            "--no-wcov-inv",
            required=False,
            action="store_false",
            help="Do not write inverse white noise covariance",
            dest="write_wcov_inv",
        )
        parser.set_defaults(write_wcov_inv=True)
    except argparse.ArgumentError:
        pass

    try:
        parser.add_argument(
            "--zip",
            required=False,
            action="store_true",
            help="Compress the map outputs",
            dest="zip_maps",
        )
        parser.add_argument(
            "--no-zip",
            required=False,
            action="store_false",
            help="Do not compress the map outputs",
            dest="zip_maps",
        )
        parser.set_defaults(zip_maps=True)
    except argparse.ArgumentError:
        pass

    # `nside` may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    return


def init_binner(args, comm, data, detweights, verbose=True, pixels="pixels"):
    """construct distributed maps to store the covariance,
    noise weighted map, and hits

    Args:
        detweights (dict) :  A dictionary of the dimensional, inverse
             variance detector weights that form the diagonal of N^-1.
    Returns:
        white_noise_cov_matrix (DistPixels) :  The white noise
            covariance matrices needed to bin signal onto the
             distributed map `dist_map`.
        dist_map (DistPixels) :  An empty distributed map compatible
            with `white_noise_cov_matrix` ready for binning TOD.  Can be
            used repeatedly in calls to `apply_binner`.

    Outputs:
        '{args.outdir}/hits.fits' : the hit map, if `args.write_hits` is True.
        '{args.outdir}/npp.fits' : the inverse white noise covariance,
             if `args.write_wcov_inv` is True.
        '{args.outdir}/npp.fits' : the white noise covariance,
             if `args.write_wcov` is True.
    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    npix = nside2npix(args.nside)

    white_noise_cov_matrix = DistPixels(data, nnz=6, dtype=np.float64, pixels=pixels)
    hits = DistPixels(data, nnz=1, dtype=np.int64, pixels=pixels)
    dist_map = DistPixels(data, nnz=3, dtype=np.float64, pixels=pixels)

    # compute the hits and covariance once, since the pointing and noise
    # weights are fixed.

    if white_noise_cov_matrix.data is not None:
        white_noise_cov_matrix.data.fill(0.0)

    if hits.data is not None:
        hits.data.fill(0)

    build_wcov = OpAccumDiag(
        detweights=detweights, invnpp=white_noise_cov_matrix, hits=hits
    )
    build_wcov.exec(data)

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Accumulate N_pp'^1")

    white_noise_cov_matrix.allreduce()
    hits.allreduce()

    if comm.world_rank == 0 and verbose:
        timer.report_clear("All reduce N_pp'^1")

    if args.write_hits:
        fname = os.path.join(args.outdir, "hits.fits")
        if args.zip_maps:
            fname += ".gz"
        hits.write_healpix_fits(fname)
        if comm.world_rank == 0 and verbose:
            log.info("Wrote hits to {}".format(fname))
    if args.write_wcov_inv:
        fname = os.path.join(args.outdir, "invnpp.fits")
        if args.zip_maps:
            fname += ".gz"
        white_noise_cov_matrix.write_healpix_fits(fname)
        if comm.world_rank == 0 and verbose:
            log.info("Wrote inverse white noise covariance to {}".format(fname))

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Writing hits and N_pp'^1")

    # invert it
    covariance_invert(white_noise_cov_matrix, 1.0e-3)

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Invert N_pp'^1")

    if args.write_wcov:
        fname = os.path.join(args.outdir, "npp.fits")
        if args.zip_maps:
            fname += ".gz"
        white_noise_cov_matrix.write_healpix_fits(fname)
        if comm.world_rank == 0 and verbose:
            log.info("Wrote white noise covariance to {}".format(fname))

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Write N_pp'")

    # in debug mode, print out data distribution information
    if args.debug:
        handle = None
        if rank == 0:
            handle = open(os.path.join(args.outdir, "distdata.txt"), "w")
        data.info(handle)
        if rank == 0:
            handle.close()
        if comm.world_rank == 0 and verbose:
            timer.report_clear("Dumping data distribution")
    return white_noise_cov_matrix, dist_map


def apply_binner(
    args,
    comm,
    data,
    white_noise_cov_matrix,
    dist_map,
    detweights,
    outpath,
    cache_prefix=None,
    prefix="binned",
    verbose=True,
):
    """ Bin the signal in `cache_prefix` onto `dist_map`
    using the noise weights in `white_noise_cov_matrix`.

    Args :
        white_noise_cov_matrix (DistPixels) :  Pre-computed white noise
             covariance matrices
        dist_map (DistPixels) :  Pre-allocated distributed map
            compatible with `white_noise_cov_matrix`.  Will be cleared
            upon entry and will contain the binned map upon exit.
        detweights (dict) :  A dictionary of inverse variance weights
            that form the diagonal of N^-1, the inverse sample-sample
            white noise covariance matrix.
        outpath (str) :  Output directory to contain the binned map.
        cache_prefix (str) :  Select which signal to bin
        prefix (str) :  Identifier to append to output map name

    Outputs :
        '`outpath`/`prefix` + .fits' : The binned map.
    """
    log = Logger.get()
    timer = Timer()

    if comm.world_rank == 0:
        os.makedirs(outpath, exist_ok=True)

    if dist_map.data is not None:
        dist_map.data.fill(0.0)
    build_dist_map = OpAccumDiag(
        zmap=dist_map, name=cache_prefix, detweights=detweights
    )
    build_dist_map.exec(data)
    dist_map.allreduce()

    if comm.world_rank == 0 and verbose:
        timer.report_clear("  Building noise-weighted map")

    covariance_apply(white_noise_cov_matrix, dist_map)

    if (comm is None or comm.world_rank == 0) and verbose:
        timer.report_clear("  Computing binned map")

    fname = os.path.join(outpath, prefix + ".fits")
    if args.zip_maps:
        fname += ".gz"
    dist_map.write_healpix_fits(fname)

    if comm.world_rank == 0 and verbose:
        log.info("Wrote binned map to {}".format(fname))
        timer.report_clear("  Writing binned map")
    return
