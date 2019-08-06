# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..map import OpMadam, OpLocalPixels, DistPixels

from ..tod import OpSimPySM, OpSimScan


def add_sky_map_args(parser):
    """ Add the sky arguments
    """

    parser.add_argument("--input-map", required=False, help="Input map for signal")

    # The nside may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # The coordinate system may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass

    return


def add_pysm_args(parser):
    """ Add the sky arguments
    """

    parser.add_argument(
        "--pysm-model",
        required=False,
        help="Comma separated models for on-the-fly PySM "
        'simulation, e.g. s3,d6,f1,a2"',
    )
    parser.add_argument(
        "--apply-beam",
        required=False,
        action="store_true",
        help="Apply beam convolution to input map with "
        "gaussian beam parameters defined in focalplane",
    )
    parser.add_argument(
        "--pysm-precomputed-cmb-K_CMB",
        required=False,
        help="Precomputed CMB map for PySM in K_CMB"
        'it overrides any model defined in pysm_model"',
    )

    # The nside may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # The coordinate system may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass

    return


@function_timer
def scan_sky_signal(
    args, comm, data, localsm, subnpix, cache_prefix="signal", verbose=True
):
    """ Scan sky signal from a map.

    """
    if not args.input_map:
        return None

    log = Logger.get()
    timer = Timer()
    timer.start()

    if comm.world_rank == 0 and verbose:
        log.info("Scanning input map")

    npix = 12 * args.nside ** 2

    # Scan the sky signal
    if comm.world_rank == 0 and not os.path.isfile(args.input_map):
        raise RuntimeError("Input map does not exist: {}".format(args.input_map))
    distmap = DistPixels(
        comm=comm.comm_world,
        size=npix,
        nnz=3,
        dtype=np.float32,
        submap=subnpix,
        local=localsm,
    )
    distmap.read_healpix_fits(args.input_map)
    scansim = OpSimScan(distmap=distmap, out=cache_prefix)
    scansim.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Read and sample map")

    return cache_prefix


@function_timer
def simulate_sky_signal(
    args, comm, data, focalplanes, subnpix, localsm, cache_prefix, verbose=False
):
    """ Use PySM to simulate smoothed sky signal.

    """
    if not args.pysm_model:
        return
    timer = Timer()
    timer.start()
    # Convolve a signal TOD from PySM
    op_sim_pysm = OpSimPySM(
        comm=comm.comm_rank,
        out=cache_prefix,
        pysm_model=args.pysm_model.split(","),
        pysm_precomputed_cmb_K_CMB=args.pysm_precomputed_cmb_K_CMB,
        focalplanes=focalplanes,
        nside=args.nside,
        subnpix=subnpix,
        localsm=localsm,
        apply_beam=args.apply_beam,
        coord=args.coord,
    )
    op_sim_pysm.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("PySM")

    return cache_prefix


@function_timer
def expand_pointing(args, comm, data):
    """ Expand boresight pointing to every detector.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    hwprpm = args.hwprpm
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)
    hwpsteptime = args.hwpsteptime

    if comm.world_rank == 0:
        log.info("Expanding pointing")

    pointing = OpPointingHpix(
        nside=args.nside,
        nest=True,
        mode="IQU",
        hwprpm=hwprpm,
        hwpstep=hwpstep,
        hwpsteptime=hwpsteptime,
    )

    pointing.exec(data)

    # Only purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if (args.tidas is None) and (args.spt3g is None):
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_radec_quats()

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Pointing generation")

    return
