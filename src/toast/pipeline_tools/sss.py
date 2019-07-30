# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..tod import OpSimScanSynchronousSignal


def add_sss_args(parser):
    """ Add the sky arguments
    """

    parser.add_argument(
        "--ground-map", required=False, help="Fixed ground template map"
    )
    parser.add_argument(
        "--ground-nside",
        required=False,
        default=128,
        type=np.int,
        help="Ground template resolution",
    )
    parser.add_argument(
        "--ground-fwhm-deg",
        required=False,
        default=10,
        type=np.float,
        help="Ground template smoothing in degrees",
    )
    parser.add_argument(
        "--ground-lmax",
        required=False,
        default=256,
        type=np.int,
        help="Ground template expansion order",
    )
    parser.add_argument(
        "--ground-scale",
        required=False,
        default=1e-3,
        type=np.float,
        help="Ground template RMS at el=45 deg",
    )
    parser.add_argument(
        "--ground-power",
        required=False,
        default=-1,
        type=np.float,
        help="Exponential for suppressing ground pick-up at "
        "higher observing elevation",
    )

    parser.add_argument(
        "--simulate-ground",
        required=False,
        action="store_true",
        help="Enable simulating ground pickup.",
        dest="simulate_ground",
    )
    parser.add_argument(
        "--no-simulate-ground",
        required=False,
        action="store_true",
        help="Enable simulating ground pickup.",
        dest="simulate_ground",
    )
    parser.set_defaults(simulate_ground=False)

    return


@function_timer
def simulate_sss(args, comm, data, mc, cache_prefix=None, verbose=False):
    if not args.simulate_ground:
        return
    log = Logger.get()
    timer = Timer()
    if comm.world_rank == 0 and verbose:
        log.info("Simulating sss")
    timer.start()
    nse = OpSimScanSynchronousSignal(
        out=cache_prefix,
        realization=mc,
        nside=args.ground_nside,
        fwhm=args.ground_fwhm_deg,
        scale=args.ground_scale,
        lmax=args.ground_lmax,
        power=args.ground_power,
        path=args.ground_map,
        report_timing=verbose,
    )
    nse.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("sss simulation")
    return
