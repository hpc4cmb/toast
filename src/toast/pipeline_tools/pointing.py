# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..map import DistPixels
from ..todmap import OpMadam, OpLocalPixels, OpPointingHpix, get_submaps_nested


def add_pointing_args(parser):
    """ Add the pointing arguments
    """

    # `nside` may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # `coord` may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass

    parser.add_argument(
        "--single-precision-pointing",
        required=False,
        action="store_true",
        help="Use single precision for pointing in memory.",
        dest="single_precision_pointing",
    )
    parser.add_argument(
        "--no-single-precision-pointing",
        required=False,
        action="store_false",
        help="Use single precision for pointing in memory.",
        dest="single_precision_pointing",
    )
    parser.set_defaults(single_precision_pointing=False)

    # Common flag mask may already be added
    try:
        parser.add_argument(
            "--common-flag-mask",
            required=False,
            default=1,
            type=np.uint8,
            help="Common flag mask",
        )
    except argparse.ArgumentError:
        pass
    # `flush` may already be added
    try:
        parser.add_argument(
            "--flush",
            required=False,
            default=False,
            action="store_true",
            help="Flush every print statement.",
        )
    except argparse.ArgumentError:
        pass
    return


@function_timer
def expand_pointing(args, comm, data):
    """ Expand boresight pointing to every detector.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    hwprpm = args.hwp_rpm
    hwpstep = None
    if args.hwp_step_deg is not None:
        hwpstep = float(args.hwp_step_deg)
    hwpsteptime = args.hwp_step_time_s

    if comm.world_rank == 0:
        log.info("Expanding pointing")

    pointing = OpPointingHpix(
        nside=args.nside,
        nest=True,
        mode="IQU",
        single_precision=args.single_precision_pointing,
    )

    pointing.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Pointing generation")

    return


@function_timer
def get_submaps(args, comm, data):
    """ Get a list of locally hit pixels and submaps on every process.

    """
    log = Logger.get()
    if comm.world_rank == 0:
        log.info("Scanning local submaps")
    timer = Timer()
    timer.start()

    localpix, localsm, subnpix = get_submaps_nested(data, args.nside)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0:
        timer.report_clear("Identify local submaps")

    return localpix, localsm, subnpix
