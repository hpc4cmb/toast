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

from ..todmap import OpFilterBin


def add_filterbin_args(parser):
    """Add mapmaker arguments"""
    parser.add_argument(
        "--filterbin-prefix",
        required=False,
        default="toast",
        help="Output map prefix",
        dest="filterbin_prefix",
    )
    parser.add_argument(
        "--filterbin-mask",
        required=False,
        help="Filtering mask",
        dest="filterbin_mask",
    )
    parser.add_argument(
        "--filterbin-ground-order",
        required=False,
        type=np.int,
        help="Ground filter order",
        dest="filterbin_ground_order",
    )
    parser.add_argument(
        "--filterbin-split-ground-template",
        required=False,
        action="store_true",
        help="Split ground template by scan direction",
        dest="filterbin_split_ground_template",
    )
    parser.add_argument(
        "--no-filterbin-split-ground-template",
        required=False,
        action="store_false",
        help="Do not split ground template",
        dest="filterbin_split_ground_template",
    )
    parser.set_defaults(filterbin_split_ground_template=False)
    parser.add_argument(
        "--filterbin-poly-order",
        required=False,
        type=np.int,
        help="Polynomial filter order",
        dest="filterbin_poly_order",
    )
    parser.add_argument(
        "--filterbin-obs-matrix",
        required=False,
        action="store_true",
        help="Write observation_matrix",
        dest="filterbin_write_obs_matrix",
    )
    parser.add_argument(
        "--no-filterbin-obs-matrix",
        required=False,
        action="store_false",
        help="Do not write observation_matrix",
        dest="filterbin_write_obs_matrix",
    )
    parser.set_defaults(filterbin_write_obs_matrix=False)
    parser.add_argument(
        "--filterbin-deproject-map",
        required=False,
        help="Deprojection template file",
    )
    parser.add_argument(
        "--filterbin-deproject-map",
        required=False,
        help="Deprojection template file",
    )
    parser.add_argument(
        "--filterbin-deproject-pattern",
        default=".*",
        help="Deprojection pattern (which detectors to filter)",
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
def apply_filterbin(
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
):
    log = Logger.get()
    timer = Timer()

    if outpath is None:
        outpath = args.out

    file_root = args.filterbin_prefix
    if extra_prefix is not None:
        if len(file_root) > 0 and not file_root.endswith("_"):
            file_root += "_"
        file_root += "{}".format(extra_prefix)

    if time_comms is None:
        time_comms = [("all", comm.comm_world)]

    if telescope_data is None:
        telescope_data = [("all", data)]

    for time_name, time_comm in time_comms:
        for tele_name, tele_data in telescope_data:

            write_hits = args.write_hits and first_call
            write_wcov_inv = args.write_wcov_inv and first_call
            write_wcov = args.write_wcov and first_call
            write_binned = args.write_binmap

            if len(time_name.split("-")) == 3:
                # Special rules for daily maps
                if not args.do_daymaps:
                    continue
                if len(telescope_data) > 1 and tele_name == "all":
                    # Skip daily maps over multiple telescopes
                    continue

            timer.clear()
            timer.start()

            deproject_nnz = None
            if args.filterbin_deproject_map:
                deproject_nnz = None
                if comm is None or comm.comm_world.rank == 0:
                    hdulist = pf.open(args.filterbin_deproject_map, "r")
                    deproject_nnz = hdulist[1].header["tfields"]
                    hdulist.close()
                if comm is not None:
                    deproject_nnz = comm.comm_world.bcast(deproject_nnz)

            if len(file_root) > 0 and not file_root.endswith("_"):
                file_root += "_"
            prefix = "{}telescope_{}_time_{}_".format(file_root, tele_name, time_name)

            filterbin = OpFilterBin(
                nside=args.nside,
                nnz=3,
                name=cache_name,
                outdir=outpath,
                outprefix=prefix,
                write_hits=write_hits,
                zip_maps=args.zip_maps,
                write_wcov_inv=write_wcov_inv,
                write_wcov=write_wcov,
                write_binned=write_binned,
                rcond_limit=1e-3,
                maskfile=args.mapmaker_mask,
                common_flag_mask=args.common_flag_mask,
                flag_mask=1,
                intervals="intervals",
                pixels_name="pixels",
                ground_filter_order=args.filterbin_ground_order,
                split_ground_template=args.filterbin_split_ground_template,
                poly_filter_order=args.filterbin_poly_order,
                write_obs_matrix=args.filterbin_write_obs_matrix,
                deproject_map=args.filterbin_deproject_map,
                deproject_pattern=args.filterbin_deproject_pattern,
                deproject_nnz=deproject_nnz,
            )

            filterbin.exec(tele_data, time_comm)

    if comm.world_rank == 0 and verbose:
        timer.report_clear("  OpFilterBin")

    return
