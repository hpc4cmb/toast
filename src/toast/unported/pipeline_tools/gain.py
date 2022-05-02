# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import Timer, function_timer
from ..tod import OpGainScrambler
from ..utils import Environment, Logger


def add_gainscrambler_args(parser):
    """Add the noise simulation arguments"""
    parser.add_argument(
        "--gainscrambler",
        required=False,
        default=False,
        action="store_true",
        help="Add simulated noise",
        dest="apply_gainscrambler",
    )
    parser.add_argument(
        "--no-gainscrambler",
        required=False,
        action="store_false",
        help="Do not add simulated noise",
        dest="apply_gainscrambler",
    )

    parser.add_argument(
        "--gain-sigma",
        required=False,
        default=0.01,
        type=np.float,
        help="Simulated gain fluctuation amplitude",
    )
    return


@function_timer
def scramble_gains(args, comm, data, mc, cache_name=None, verbose=False):
    if not args.apply_gainscrambler or args.gain_sigma == 0:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("Scrambling gains")
    scrambler = OpGainScrambler(sigma=args.gain_sigma, name=cache_name, realization=mc)
    scrambler.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Scramble gains")
    return
