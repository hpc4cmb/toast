# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import dateutil.parser
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment
from ..weather import Weather

from ..tod import (
    OpSimAtmosphere,
    atm_available_utils,
)


def add_todground_args(parser):
    """ Add TODGround arguments
    """
    parser.add_argument(
        "--scan-rate",
        required=False,
        default=1.0,
        type=np.float,
        help="Scanning rate [deg / s]",
    )
    parser.add_argument(
        "--scan-accel",
        required=False,
        default=1.0,
        type=np.float,
        help="Scanning rate change [deg / s^2]",
    )
    parser.add_argument(
        "--sun-angle-min",
        required=False,
        default=30.0,
        type=np.float,
        help="Minimum azimuthal distance between the Sun and the bore sight [deg]",
    )
    parser.add_argument(
        "--schedule",
        required=True,
        help="Comma-separated list CES schedule files "
        "(from toast_ground_schedule.py)",
    )
    parser.add_argument(
        "--weather",
        required=False,
        help="Comma-separated list of TOAST weather files for "
        "every schedule.  Repeat the same file if the "
        "schedules share observing site.",
    )
    parser.add_argument(
        "--timezone",
        required=False,
        type=np.int,
        default=0,
        help="Offset to apply to MJD to separate days [hours]",
    )

    # `sample-rate` may be already added
    try:
        parser.add_argument(
            "--sample-rate",
            required=False,
            default=100.0,
            type=np.float,
            help="Detector sample rate (Hz)",
        )
    except argparse.ArgumentError:
        pass
    # `coord` may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass
    return


def _parse_line(line, all_ces):
    """ Parse one line of the schedule file
    """
    if line.startswith("#"):
        return
    
    (start_date,
     start_time,
     stop_date,
     stop_time,
     mjdstart,
     mjdstop,
     name,
     azmin,
     azmax,
     el,
     rs,
     sun_el1,
     sun_az1,
     sun_el2,
     sun_az2,
     moon_el1,
     moon_az1,
     moon_el2,
     moon_az2,
     moon_phase,
     scan,
     subscan,
    ) = line.split()
    start_time = start_date + " " + start_time
    stop_time = stop_date + " " + stop_time
    # Define season as a calendar year.  This can be
    # changed later and could even be in the schedule file.
    season = int(start_date.split("-")[0])
    try:
        start_time = dateutil.parser.parse(start_time + " +0000")
        stop_time = dateutil.parser.parse(stop_time + " +0000")
    except Exception:
        start_time = dateutil.parser.parse(start_time)
        stop_time = dateutil.parser.parse(stop_time)
    start_timestamp = start_time.timestamp()
    stop_timestamp = stop_time.timestamp()
    all_ces.append([
        start_timestamp,
        stop_timestamp,
        name,
        float(mjdstart),
        int(scan),
        int(subscan),
        float(azmin),
        float(azmax),
        float(el),
        season,
        start_date,
    ])
    return

@function_timer
def load_schedule(args, comm, verbose=False):
    """ Load the observing schedule(s).

    """
    schedules = []
    timer = Timer()
    timer.start()

    if comm is None or comm.world_rank == 0:
        ftimer = Timer()
        for fn in args.schedule.split(","):
            if not os.path.isfile(fn):
                raise RuntimeError("No such schedule file: {}".format(fn))
            ftimer.start()
            with open(fn, "r") as f:
                while True:
                    line = f.readline()
                    if line.startswith("#"):
                        continue
                    (site_name, telescope, site_lat, site_lon, site_alt) = line.split()
                    site_alt = float(site_alt)
                    site = (site_name, telescope, site_lat, site_lon, site_alt)
                    break
                all_ces = []
                for line in f:
                    _parse_line(line, all_ces)
            schedules.append([site, all_ces])
            ftimer.stop()
            if verbose:
                ftimer.report_clear("Load {}".format(fn))

    if comm is not None:
        schedules = comm.comm_world.bcast(schedules)

    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Loading schedule")
    return schedules


@function_timer
def load_weather(args, comm, schedules, verbose=False):
    """ Load TOAST weather file(s) and attach them to the schedules.

    """
    if args.weather is None:
        return
    timer = Timer()
    timer.start()

    if comm.world_rank == 0:
        weathers = []
        weatherdict = {}
        ftimer = Timer()
        for fname in args.weather.split(","):
            if fname not in weatherdict:
                if not os.path.isfile(fname):
                    raise RuntimeError("No such weather file: {}".format(fname))
                ftimer.start()
                weatherdict[fname] = Weather(fname)
                ftimer.stop()
                ftimer.report_clear("Load {}".format(fname))
            weathers.append(weatherdict[fname])
    else:
        weathers = None

    if comm.comm_world is not None:
        weathers = comm.comm_world.bcast(weathers)
    if len(weathers) == 1 and len(schedules) > 1:
        weathers *= len(schedules)
    if len(weathers) != len(schedules):
        raise RuntimeError("Number of weathers must equal number of schedules or be 1.")

    for schedule, weather in zip(schedules, weathers):
        schedule.append(weather)

    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Loading weather")
    return
