#!/usr/bin/env python3

# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Simple wrapper around the corresponding visualization function"""

import argparse

from toast.mpi import exception_guard, get_world
from toast.vis import plot_wcs_maps


def main(opts=None):
    parser = argparse.ArgumentParser(
        description="This program plots output maps in WCS projections.",
        usage="toast_plot_wcs <options>",
    )

    parser.add_argument(
        "--hit_file",
        required=False,
        type=str,
        default=None,
        help="The path to the hit map file",
    )

    parser.add_argument(
        "--map_file",
        required=False,
        type=str,
        default=None,
        help="The path to the map file",
    )

    parser.add_argument(
        "--truth_file",
        required=False,
        type=str,
        default=None,
        help="If data is simulated, the optional path to the input true sky map",
    )

    parser.add_argument(
        "--min_I",
        required=False,
        type=float,
        default=None,
        help="Minimum data value for the Intensity map",
    )
    parser.add_argument(
        "--max_I",
        required=False,
        type=float,
        default=None,
        help="Maximum data value for the Intensity map",
    )

    parser.add_argument(
        "--min_Q",
        required=False,
        type=float,
        default=None,
        help="Minimum data value for the Q map",
    )
    parser.add_argument(
        "--max_Q",
        required=False,
        type=float,
        default=None,
        help="Maximum data value for the Q map",
    )

    parser.add_argument(
        "--min_U",
        required=False,
        type=float,
        default=None,
        help="Minimum data value for the U map",
    )
    parser.add_argument(
        "--max_U",
        required=False,
        type=float,
        default=None,
        help="Maximum data value for the U map",
    )
    parser.add_argument(
        "--max_hits",
        required=False,
        type=int,
        default=None,
        help="Maximum data value for the hit map",
    )

    parser.add_argument(
        "--Xmin",
        required=False,
        type=float,
        default=None,
        help="Minimum X viewport fraction (0.0 - 1.0)",
    )
    parser.add_argument(
        "--Xmax",
        required=False,
        type=float,
        default=None,
        help="Maximum X viewport fraction (0.0 - 1.0)",
    )

    parser.add_argument(
        "--Ymin",
        required=False,
        type=float,
        default=None,
        help="Minimum Y viewport fraction (0.0 - 1.0)",
    )
    parser.add_argument(
        "--Ymax",
        required=False,
        type=float,
        default=None,
        help="Maximum Y viewport fraction (0.0 - 1.0)",
    )

    parser.add_argument(
        "--cmap",
        required=False,
        type=str,
        default="viridis",
        help="The colormap name (e.g. 'inferno')",
    )

    parser.add_argument(
        "--azimuth",
        required=False,
        default=False,
        action="store_true",
        help="Data is Azimuth / Elevation, so invert the X-axis",
    )

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default=None,
        help="Place plots in this directory",
    )

    args = parser.parse_args(args=opts)

    range_I = None
    if (args.min_I is not None) and (args.max_I is not None):
        range_I = (args.min_I, args.max_I)
    range_Q = None
    if (args.min_Q is not None) and (args.max_Q is not None):
        range_Q = (args.min_Q, args.max_Q)
    range_U = None
    if (args.min_U is not None) and (args.max_U is not None):
        range_U = (args.min_U, args.max_U)

    plot_wcs_maps(
        hitfile=args.hit_file,
        mapfile=args.map_file,
        range_I=range_I,
        range_Q=range_Q,
        range_U=range_U,
        max_hits=args.max_hits,
        truth=args.truth_file,
        xmin=args.Xmin,
        xmax=args.Xmax,
        ymin=args.Ymin,
        ymax=args.Ymax,
        is_azimuth=args.azimuth,
        cmap=args.cmap,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    world, procs, rank = get_world()
    with exception_guard(comm=world):
        main()
