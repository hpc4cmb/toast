# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse

import numpy as np

from toast.timing import function_timer, Timer
from toast.utils import Logger, Environment

from ..tod import OpPolyFilter, OpPolyFilter2D
from ..todmap import OpGroundFilter

#
# Polynomial filter
#


def add_polyfilter2D_args(parser):
    """Add the polynomial filter arguments to argparser"""
    parser.add_argument(
        "--polyfilter2D",
        required=False,
        default=False,
        action="store_true",
        help="Apply 2D polynomial filter",
        dest="apply_polyfilter2D",
    )
    parser.add_argument(
        "--no-polyfilter2D",
        required=False,
        action="store_false",
        help="Do not apply 2D polynomial filter",
        dest="apply_polyfilter2D",
    )
    parser.set_defaults(apply_polyfilter2D=False)

    parser.add_argument(
        "--poly-order2D",
        required=False,
        default=0,
        type=np.int,
        help="Polynomial order for the 2D polyfilter",
    )
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
    return


def add_polyfilter_args(parser):
    """Add the polynomial filter arguments to argparser"""
    parser.add_argument(
        "--polyfilter",
        required=False,
        default=False,
        action="store_true",
        help="Apply polynomial filter",
        dest="apply_polyfilter",
    )
    parser.add_argument(
        "--no-polyfilter",
        required=False,
        action="store_false",
        help="Do not apply polynomial filter",
        dest="apply_polyfilter",
    )
    parser.set_defaults(apply_polyfilter=False)

    parser.add_argument(
        "--poly-order",
        required=False,
        default=0,
        type=np.int,
        help="Polynomial order for the polyfilter",
    )
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
    return


@function_timer
def apply_polyfilter2D(args, comm, data, cache_name=None, verbose=True):
    """Apply the 2D polynomial filter to data under `cache_name`."""
    if not args.apply_polyfilter2D:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("2D Polyfiltering signal")
    polyfilter = OpPolyFilter2D(
        order=args.poly_order2D, name=cache_name, common_flag_mask=args.common_flag_mask
    )
    polyfilter.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("2D Polynomial filtering")
    return


#
# Ground filter
#


@function_timer
def apply_polyfilter(args, comm, data, cache_name=None, verbose=True):
    """Apply the polynomial filter to data under `cache_name`."""
    if not args.apply_polyfilter:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("Polyfiltering signal")
    polyfilter = OpPolyFilter(
        order=args.poly_order, name=cache_name, common_flag_mask=args.common_flag_mask
    )
    polyfilter.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Polynomial filtering")
    return


#
# Ground filter
#


def add_groundfilter_args(parser):
    """Add the ground filter arguments to argparser"""
    parser.add_argument(
        "--groundfilter",
        required=False,
        default=False,
        action="store_true",
        help="Apply ground filter",
        dest="apply_groundfilter",
    )
    parser.add_argument(
        "--no-groundfilter",
        required=False,
        action="store_false",
        help="Do not apply ground filter",
        dest="apply_groundfilter",
    )
    parser.set_defaults(apply_groundfilter=False)

    parser.add_argument(
        "--ground-order",
        required=False,
        default=0,
        type=np.int,
        help="Ground template order",
    )
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
    return


@function_timer
def apply_groundfilter(args, comm, data, cache_name=None, verbose=True):
    if not args.apply_groundfilter:
        return
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("Ground-filtering signal")
    groundfilter = OpGroundFilter(
        filter_order=args.ground_order,
        name=cache_name,
        common_flag_mask=args.common_flag_mask,
    )
    groundfilter.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Ground filtering")
    return
