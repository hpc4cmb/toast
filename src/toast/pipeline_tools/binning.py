# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..tod import OpSimDipole
from ..utils import Logger, Environment
from ..map import (
    OpAccumDiag,
    OpLocalPixels,
    covariance_apply,
    covariance_invert,
    DistPixels,
)
from .pointing import get_submaps


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

    # `nside` may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    return


def init_binner(args, comm, data, detweights, subnpix=None, localsm=None, verbose=True):
    """construct distributed maps to store the covariance,
    noise weighted map, and hits
    """
    log = Logger.get()
    timer = Timer()

    if subnpix is None or localsm is None:
        localpix, localsm, subnpix = get_submaps(args, comm, data, verbose=verbose)

    npix = 12 * args.nside * args.nside

    invnpp = DistPixels(
        comm=comm.comm_world,
        size=npix,
        nnz=6,
        dtype=np.float64,
        submap=subnpix,
        local=localsm,
    )
    hits = DistPixels(
        comm=comm.comm_world,
        size=npix,
        nnz=1,
        dtype=np.int64,
        submap=subnpix,
        local=localsm,
    )
    zmap = DistPixels(
        comm=comm.comm_world,
        size=npix,
        nnz=3,
        dtype=np.float64,
        submap=subnpix,
        local=localsm,
    )

    # compute the hits and covariance once, since the pointing and noise
    # weights are fixed.

    if invnpp.data is not None:
        invnpp.data.fill(0.0)

    if hits.data is not None:
        hits.data.fill(0)

    build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits)
    build_invnpp.exec(data)

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Accumulate N_pp'^1")

    invnpp.allreduce()
    hits.allreduce()

    if comm.world_rank == 0 and verbose:
        timer.report_clear("All reduce N_pp'^1")

    if args.write_hits:
        fname = os.path.join(args.outdir, "hits.fits")
        hits.write_healpix_fits(fname)
        if comm.world_rank == 0 and verbose:
            log.info("Wrote hits to {}".format(fname))
    if args.write_wcov_inv:
        fname = os.path.join(args.outdir, "invnpp.fits")
        invnpp.write_healpix_fits(fname)
        if comm.world_rank == 0 and verbose:
            log.info("Wrote inverse white noise covariance to {}".format(fname))

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Writing hits and N_pp'^1")

    # invert it
    covariance_invert(invnpp, 1.0e-3)

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Invert N_pp'^1")

    if args.write_wcov:
        fname = os.path.join(args.outdir, "npp.fits")
        invnpp.write_healpix_fits(fname)
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
    return invnpp, zmap


def apply_binner(
    args, comm, data, invnpp, zmap, detweights, outpath, cache_prefix=None, verbose=True
):
    """ Bin the signal in `cache_prefix` onto `zmap`
    using the noise weights in `invnpp`.
    """
    log = Logger.get()
    timer = Timer()

    if comm is None or comm.world_rank == 0:
        if not os.path.isdir(outpath):
            os.makedirs(outpath, exist_ok=True)

    if zmap.data is not None:
        zmap.data.fill(0.0)
    build_zmap = OpAccumDiag(zmap=zmap, name=cache_prefix, detweights=detweights)
    build_zmap.exec(data)
    zmap.allreduce()

    if (comm is None or comm.world_rank == 0) and verbose:
        timer.report_clear("  Building noise weighted map")

    covariance_apply(invnpp, zmap)

    if (comm is None or comm.world_rank == 0) and verbose:
        timer.report_clear("  Computing binned map")

    fname = os.path.join(outpath, "binned.fits")
    zmap.write_healpix_fits(fname)

    if (comm is None or comm.world_rank == 0) and verbose:
        log.info("Wrote binned map to {}".format(fname))
        timer.report_clear("  Writing binned map")
    return
