# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import dateutil.parser
import os

import healpy as hp
import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

# from ..tod import OpSimAtmosphere, atm_available_utils


def add_todsatellite_args(parser):
    parser.add_argument(
        "--start-time",
        required=False,
        type=float,
        default=0.0,
        help="The overall start time of the simulation",
    )

    parser.add_argument(
        "--spin-period-min",
        required=False,
        type=float,
        default=10.0,
        help="The period (in minutes) of the rotation about the spin axis",
    )
    parser.add_argument(
        "--spin-angle-deg",
        required=False,
        type=float,
        default=30.0,
        help="The opening angle (in degrees) of the boresight from the spin axis",
    )

    parser.add_argument(
        "--prec-period-min",
        required=False,
        type=float,
        default=50.0,
        help="The period (in minutes) of the rotation about the precession axis",
    )
    parser.add_argument(
        "--prec-angle-deg",
        required=False,
        type=float,
        default=65.0,
        help="The opening angle (in degrees) of the spin axis "
        "from the precession axis",
    )

    parser.add_argument(
        "--obs-time-h",
        required=False,
        type=float,
        default=1.0,
        help="Number of hours in one science observation",
    )
    parser.add_argument(
        "--gap-h",
        required=False,
        type=float,
        default=0.0,
        help="Cooler cycle time in hours between science obs",
    )
    parser.add_argument(
        "--obs-num",
        required=False,
        type=int,
        default=1,
        help="Number of complete observations",
    )

    # The HWP arguments may also be added by other TOD classes
    try:
        parser.add_argument(
            "--hwp-rpm",
            required=False,
            default=0,
            type=np.float,
            help="The rate (in RPM) of the HWP rotation",
        )
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument(
            "--hwp-step-deg",
            required=False,
            default=None,
            help="For stepped HWP, the angle in degrees of each step",
        )
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument(
            "--hwp-step-time-s",
            required=False,
            default=0,
            type=np.float,
            help="For stepped HWP, the time in seconds between steps",
        )
    except argparse.ArgumentError:
        pass

    return
