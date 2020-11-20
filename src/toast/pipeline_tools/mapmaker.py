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
from ..tod import OpPolyFilter


def add_mapmaker_args(parser):
    """Add mapmaker arguments"""
    parser.add_argument(
        "--mapmaker-prefix",
        required=False,
        default="toast",
        help="Output map prefix",
        dest="mapmaker_prefix",
    )
    parser.add_argument(
        "--mapmaker-mask",
        required=False,
        help="Destriping mask",
        dest="mapmaker_mask",
    )
    parser.add_argument(
        "--mapmaker-weightmap",
        required=False,
        help="Destriping weight map",
        dest="mapmaker_weightmap",
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
        "--mapmaker-prefilter-order",
        required=False,
        type=np.int,
        help="Polynomial prefiltering for mapmaker",
        dest="mapmaker_prefilter_order",
    )
    parser.add_argument(
        "--mapmaker-fourier2D-order",
        required=False,
        type=np.int,
        help="Per sample 2D Fourier template order",
        dest="mapmaker_fourier2D_order",
    )
    parser.add_argument(
        "--mapmaker-fourier2D-subharmonics",
        required=False,
        action="store_true",
        help="Fit linear modes along with Fourier templates",
        dest="mapmaker_fourier2D_subharmonics",
    )
    parser.add_argument(
        "--no-mapmaker-fourier2D-subharmonics",
        required=False,
        action="store_false",
        help="Do not fit linear modes along with Fourier templates",
        dest="mapmaker_fourier2D_subharmonics",
    )
    parser.add_argument(
        "--mapmaker-gain-poly-order",
        required=False,
        help="Fit gain template with Legendre Polynomials",
        dest="mapmaker_gain_poly_order",
        type=np.int
    )
    parser.add_argument(
        "--mapmaker-calibration",
        required=False,
        action="store_true",
        help="Calibrate for the fitted gain amplitudes before destriping",
        dest="mapmaker_calibration",
    )
    parser.add_argument(
        "--no-mapmaker-calibration",
        required=False,
        action="store_false",
        help="Do not calibrate for the fitted gain amplitudes before destriping",
        dest="mapmaker_calibration",
    )
    parser.set_defaults(mapmaker_fourier2D_subharmonics=False)
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
    gain_templatename=None,
):
    log = Logger.get()
    timer = Timer()

    if outpath is None:
        outpath = args.out

    file_root = args.mapmaker_prefix
    if extra_prefix is not None:
        if len(file_root) > 0 and not file_root.endswith("_"):
            file_root += "_"
        file_root += "{}".format(extra_prefix)

    if time_comms is None:
        time_comms = [("all", comm.comm_world)]

    if telescope_data is None:
        telescope_data = [("all", data)]

    if not bin_only and args.mapmaker_prefilter_order is not None:
        timer.start()
        if comm.world_rank == 0 and verbose:
            print(
                "Applying polynomial prefilter, order = {}".format(
                    args.mapmaker_prefilter_order
                ),
                flush=True,
            )
        polyfilter = OpPolyFilter(
            order=args.mapmaker_prefilter_order,
            name=cache_name,
            common_flag_mask=args.common_flag_mask,
        )
        polyfilter.exec(data)
        timer.stop()
        if comm.world_rank == 0 and verbose:
            timer.report("Polynomial prefilter")

    for time_name, time_comm in time_comms:
        for tele_name, tele_data in telescope_data:

            write_hits = args.write_hits and first_call
            write_wcov_inv = args.write_wcov_inv and first_call
            write_wcov = args.write_wcov and first_call
            if bin_only:
                baseline_length = None
                write_binned = True
                write_destriped = False
                fourier2D_order = None
            else:
                baseline_length = args.mapmaker_baseline_length
                write_binned = args.write_binmap
                write_destriped = True
                fourier2D_order = args.mapmaker_fourier2D_order

            if len(time_name.split("-")) == 3:
                # Special rules for daily maps
                if not args.do_daymaps:
                    continue
                if len(telescope_data) > 1 and tele_name == "all":
                    # Skip daily maps over multiple telescopes
                    continue
                # Do not destripe daily maps
                baseline_length = None
                write_binned = True
                write_destriped = False

            timer.clear()
            timer.start()

            if len(file_root) > 0 and not file_root.endswith("_"):
                file_root += "_"
            prefix = "{}telescope_{}_time_{}_".format(file_root, tele_name, time_name)
            if args.mapmaker_calibration :
                if gain_templatename is None:
                    raise ValueError("Can't calibrate if the template signal is not specified")
                calibrator = OpMapMaker(
                    nside=args.nside,
                    nnz=3,
                    name=cache_name,
                    outdir=outpath,
                    outprefix=prefix,
                    write_hits=False ,
                    write_wcov_inv=False,
                    write_wcov=False,
                    write_binned=False,
                    write_destriped=False,
                    write_rcond=False,
                    rcond_limit=1e-3,
                    baseline_length=args.obs_time_h, ## we calibrate for the whole observation length
                    maskfile=args.mapmaker_mask,
                    weightmapfile=args.mapmaker_weightmap,
                    common_flag_mask=args.common_flag_mask,
                    flag_mask=1,
                    intervals="intervals",
                    gain_templatename=gain_templatename  ,
                    gain_poly_order= args.mapmaker_gain_poly_order,
                    iter_min=3,
                    iter_max=args.mapmaker_iter_max,
                    use_noise_prior=False,
                    pixels="pixels",
                )

                calibrator.exec(tele_data, time_comm)

            mapmaker = OpMapMaker(
                nside=args.nside,
                nnz=3,
                name=cache_name,
                outdir=outpath,
                outprefix=prefix,
                write_hits=(args.write_hits and first_call),
                zip_maps=args.zip_maps,
                write_wcov_inv=(args.write_wcov_inv and first_call),
                write_wcov=(args.write_wcov and first_call),
                write_binned=write_binned,
                write_destriped=write_destriped,
                write_rcond=True,
                rcond_limit=1e-3,
                baseline_length=baseline_length,
                maskfile=args.mapmaker_mask,
                weightmapfile=args.mapmaker_weightmap,
                common_flag_mask=args.common_flag_mask,
                flag_mask=1,
                intervals="intervals",
                subharmonic_order=None,
                fourier2D_order=fourier2D_order,
                fourier2D_subharmonics=args.mapmaker_fourier2D_subharmonics,
                iter_min=3,
                iter_max=args.mapmaker_iter_max,
                use_noise_prior=args.mapmaker_noisefilter,
                precond_width=args.mapmaker_precond_width,
                pixels="pixels",
            )

            mapmaker.exec(tele_data, time_comm)

            # User needs to set TOAST_FUNCTIME to see timing results
            if "TOAST_FUNCTIME" in os.environ and os.environ["TOAST_FUNCTIME"]:
                mapmaker.report_timing()

    if comm.world_rank == 0 and verbose:
        timer.report_clear("  OpMapMaker")

    return
