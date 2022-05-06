#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script creates a schedule file compatible with the SatelliteSchedule class.
"""

import argparse
import sys
from datetime import datetime

from astropy import units as u

import toast
from toast.mpi import get_world
from toast.schedule_sim_satellite import create_satellite_schedule
from toast.utils import Logger


def main():
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Create a satellite observing schedule."
    )

    parser.add_argument(
        "--out",
        required=False,
        default="schedule.ecsv",
        help="The output schedule file.",
    )

    parser.add_argument(
        "--start",
        required=False,
        default=None,
        help="The start time of the mission as an ISO-8601 string.",
    )

    parser.add_argument(
        "--num_obs",
        required=False,
        type=int,
        default=1,
        help="The number of observations.",
    )

    parser.add_argument(
        "--obs_minutes",
        required=False,
        default=10.0,
        type=float,
        help="The length of each observation in minutes.",
    )

    parser.add_argument(
        "--gap_minutes",
        required=False,
        default=0.0,
        type=float,
        help="The length of the gaps between observations in minutes.",
    )

    parser.add_argument(
        "--prec_minutes",
        required=False,
        default=50.0,
        type=float,
        help="The precession period in minutes.",
    )

    parser.add_argument(
        "--spin_minutes",
        required=False,
        default=10.0,
        type=float,
        help="The spin period in minutes.",
    )

    args = parser.parse_args()

    mission_start = None
    if args.start is None:
        # start now
        mission_start = datetime.now()
    else:
        # convert
        mission_start = datetime.fromisoformat(args.start)

    # This is a serial script, guard against being called with multiple processes
    mpiworld, procs, rank = get_world()

    if rank == 0:
        sch = create_satellite_schedule(
            prefix="",
            mission_start=mission_start,
            observation_time=args.obs_minutes * u.minute,
            gap_time=args.gap_minutes * u.minute,
            num_observations=args.num_obs,
            prec_period=args.prec_minutes * u.minute,
            spin_period=args.spin_minutes * u.minute,
        )
        sch.write(args.out)


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
