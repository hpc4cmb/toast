# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import copy
import os
import re

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..todmap import OpMapMaker


def add_mapmaker_args(parser):
    """ Add mapmaker arguments
    """
    parser.add_argument(
        "--mapmaker-prefix",
        required=False,
        default="toast",
        help="Output map prefix",
        dest="mapmaker_prefix",
    )
    parser.add_argument(
        "--mapmaker-iter-max",
        required=False,
        default=1000,
        type=np.int,
        help="Maximum number of CG iterations",
        dest="mapmaker_iter_max",
    )
    parser.add_argument(
        "--mapmaker-precond-width",
        required=False,
        default=100,
        type=np.int,
        help="Width of the Madam band preconditioner",
        dest="mapmaker_precond_width",
    )
    parser.add_argument(
        "--mapmaker-baseline-length",
        required=False,
        default=10000.0,
        type=np.float,
        help="Destriping baseline length (seconds)",
        dest="mapmaker_baseline_length",
    )
    parser.add_argument(
        "--mapmaker-noisefilter",
        required=False,
        default=False,
        action="store_true",
        help="Destripe with the noise filter enabled",
        dest="mapmaker_noisefilter",
    )
    try:
        parser.add_argument(
            "--binmap",
            required=False,
            action="store_true",
            help="Write binned maps [default]",
            dest="write_binmap",
        )
        parser.add_argument(
            "--no-binmap",
            required=False,
            action="store_false",
            help="Do not write binned maps",
            dest="write_binmap",
        )
        parser.set_defaults(write_binmap=True)
    except argparse.ArgumentError:
        pass

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

    return


@function_timer
def apply_mapmaker(
    args,
    comm,
    data,
    outpath,
    cache_name,
    time_comms=None,
    telescope_data=None,
    first_call=True,
    extra_prefix=None,
    verbose=True,
    bin_only=False,
):
    log = Logger.get()
    timer = Timer()

    if outpath is None:
        outpath = args.out

    mapmaker = OpMapMaker(
        nside=args.nside,
        nnz=3,
        name=cache_name,
        outdir=outpath,
        outprefix=args.mapmaker_prefix,
        write_hits=args.write_hits,
        zip_maps=args.zip_maps,
        write_wcov_inv=args.write_wcov_inv,
        write_wcov=args.write_wcov,
        write_binned=args.write_binmap,
        write_destriped=True,
        write_rcond=True,
        rcond_limit=1e-3,
        baseline_length=args.mapmaker_baseline_length,
        maskfile=None,
        weightmapfile=None,
        common_flag_mask=args.common_flag_mask,
        flag_mask=1,
        intervals="intervals",
        subharmonic_order=None,
        iter_min=3,
        iter_max=args.mapmaker_iter_max,
        use_noise_prior=args.mapmaker_noisefilter,
        precond_width=args.mapmaker_precond_width,
        pixels="pixels",
    )

    mapmaker.exec(data)

    if comm.world_rank == 0 and verbose:
        timer.report_clear("  OpMapMaker")

    return
