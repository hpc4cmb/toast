# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..todmap import OpSimDipole
from ..utils import Logger, Environment


def add_dipole_args(parser):
    parser.add_argument(
        "--dipole",
        required=False,
        action="store_true",
        help="Add simulated dipole",
        dest="simulate_dipole",
    )
    parser.add_argument(
        "--no-dipole",
        required=False,
        action="store_false",
        help="Do not add simulated dipole",
        dest="simulate_dipole",
    )
    parser.set_defaults(simulate_dipole=False)
    parser.add_argument(
        "--dipole-mode",
        required=False,
        default="total",
        help="Dipole mode is 'total', 'orbital' or 'solar'",
    )
    parser.add_argument(
        "--dipole-solar-speed-kms",
        required=False,
        help="Solar system speed [km/s]",
        type=float,
        default=369.0,
    )
    parser.add_argument(
        "--dipole-solar-gal-lat-deg",
        required=False,
        help="Solar system speed galactic latitude [degrees]",
        type=float,
        default=48.26,
    )
    parser.add_argument(
        "--dipole-solar-gal-lon-deg",
        required=False,
        help="Solar system speed galactic longitude[degrees]",
        type=float,
        default=263.99,
    )

    # `coord` may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass
    return


def simulate_dipole(args, comm, data, cache_prefix, freq=0, verbose=True):
    """ Simulate CMB dipole and add it to `cache_prefix`

    Args:
        freq (float) :  Observing frequency in GHz.  If non-zero, the
            relativistic quadrupole will include frequency corrections.
    """
    if not args.simulate_dipole:
        return None
    log = Logger.get()
    timer = Timer()
    timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("Simulating dipole")
    has_signal = True
    op_sim_dipole = OpSimDipole(
        mode=args.dipole_mode,
        solar_speed=args.dipole_solar_speed_kms,
        solar_gal_lat=args.dipole_solar_gal_lat_deg,
        solar_gal_lon=args.dipole_solar_gal_lon_deg,
        out=cache_prefix,
        keep_quats=False,
        keep_vel=False,
        subtract=False,
        coord=args.coord,
        freq=freq,
        flag_mask=255,
        common_flag_mask=255,
    )
    op_sim_dipole.exec(data)
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Simulate dipole")
    return cache_prefix
