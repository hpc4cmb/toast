# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import dateutil.parser
import os

import healpy as hp
import numpy as np

from .. import qarray
from ..timing import function_timer, Timer
from ..utils import Logger, Environment
from ..weather import Weather

from .classes import Telescope, Focalplane
from .debug import add_debug_args

# Schedule, Site and CES are small helper classes for building
# ground observations


class Schedule:
    def __init__(self, telescope=None, ceslist=None, sort=False):
        self.telescope = telescope
        self.ceslist = ceslist
        if sort:
            self.sort_ceslist()
        return

    def sort_ceslist(self):
        """ Sort the list of CES by name
        """
        nces = len(self.ceslist)
        for i in range(nces - 1):
            for j in range(i + 1, nces):
                if self.ceslist[j].name < self.ceslist[j - 1].name:
                    temp = self.ceslist[j]
                    self.ceslist[j] = self.ceslist[j - 1]
                    self.ceslist[j - 1] = temp
        return


class Site:
    def __init__(self, name, lat, lon, alt, weather=None):
        """ Instantiate a Site object

        args:
            name (str)
            lat (str) :  Site latitude as a pyEphem string
            lon (str) :  Site longitude as a pyEphem string
            alt (float) :  Site altitude in meters
            telescope (str) :  Optional telescope instance at the site
        """
        self.name = name
        # Strings get interpreted correctly pyEphem.
        # Floats must be in radians
        self.lat = str(lat)
        self.lon = str(lon)
        self.alt = alt
        self.id = 0
        self.weather = weather

    def __repr__(self):
        value = (
            "(Site '{}' : ID = {}, lon = {}, lat = {}, alt = {} m, "
            "weather = {})"
            "".format(self.name, self.id, self.lon, self.lat, self.alt, self.weather)
        )
        return value


class CES:
    def __init__(
        self,
        start_time,
        stop_time,
        name,
        mjdstart,
        scan,
        subscan,
        azmin,
        azmax,
        el,
        season,
        start_date,
        rising,
        mindist_sun,
        mindist_moon,
        el_sun,
    ):
        self.start_time = start_time
        self.stop_time = stop_time
        self.name = name
        self.mjdstart = mjdstart
        self.scan = scan
        self.subscan = subscan
        self.azmin = azmin
        self.azmax = azmax
        self.el = el
        self.season = season
        self.start_date = start_date
        self.rising = rising
        self.mindist_sun = mindist_sun
        self.mindist_moon = mindist_moon
        self.el_sun = el_sun


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

    add_debug_args(parser)

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
    parser.add_argument(
        "--split-schedule",
        required=False,
        help="Only use a subset of the schedule.  The argument is a string "
        'of the form "[isplit],[nsplit]" and only observations that satisfy '
        "scan modulo nsplit == isplit are included",
    )
    parser.add_argument(
        "--sort-schedule",
        required=False,
        action="store_true",
        help="Reorder the observing schedule so that observations of the"
        "same patch are consecutive.  This will reduce the sky area observed "
        "by individual process groups.",
        dest="sort_schedule",
    )
    parser.add_argument(
        "--no-sort-schedule",
        required=False,
        action="store_false",
        help="Do not reorder the observing schedule so that observations of the"
        "same patch are consecutive.",
        dest="sort_schedule",
    )
    parser.set_defaults(sort_schedule=True)

    # The HWP arguments may also be added by other TOD classes
    try:
        parser.add_argument(
            "--hwp-rpm",
            required=False,
            type=np.float,
            help="The rate (in RPM) of the HWP rotation",
        )
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument(
            "--hwp-step-deg",
            required=False,
            type=np.float,
            help="For stepped HWP, the angle in degrees of each step",
        )
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument(
            "--hwp-step-time-s",
            required=False,
            type=np.float,
            help="For stepped HWP, the time in seconds between steps",
        )
    except argparse.ArgumentError:
        pass
    # Modulate noise PSD by observing elevation
    parser.add_argument(
        "--elevation-noise-a",
        default=0,
        type=np.float,
        help="Evaluate noise PSD as (a / sin(el) + b) ** 2 * fsample * 1e-12",
    )
    parser.add_argument(
        "--elevation-noise-b",
        default=0,
        type=np.float,
        help="Evaluate noise PSD as (a / sin(el) + b) ** 2 * fsample * 1e-12",
    )

    return


@function_timer
def get_elevation_noise(args, comm, data, key="noise"):
    """ Insert elevation-dependent noise
    """
    if args.elevation_noise_a == 0 and args.elevation_noise_b == 0:
        return
    timer = Timer()
    timer.start()
    a = args.elevation_noise_a
    b = args.elevation_noise_b
    fsample = args.sample_rate
    for obs in data.obs:
        tod = obs["tod"]
        noise = obs[key]
        for det in tod.local_dets:
            if det not in noise.keys:
                raise RuntimeError(
                    'Detector "{}" does not have a PSD in the noise object'.format(det)
                )
            # freq = noise.freq[det]
            psd = noise.psd(det)
            try:
                # Some TOD classes provide a shortcut to Az/El
                _, el = tod.read_azel(detector=det)
            except Exception as e:
                nlocal = tod.local_samples[1]
                local_start = nlocal // 2 - nlocal // 20
                n = nlocal // 10
                azelquat = tod.read_pntg(
                    detector=det, local_start=local_start, n=n, azel=True
                )
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, _ = qarray.to_position(azelquat)
                el = np.pi / 2 - theta
            # The model evaluates to uK / sqrt(Hz)
            # Translate it to K_CMB ** 2
            el = np.median(el)
            old_net = np.median(psd[-10:])
            new_net = (a / np.sin(el) + b) ** 2 * fsample * 1e-12
            psd[:] *= new_net / old_net
    if comm.comm_world is None or comm.world_rank == 0:
        timer.report_clear("Elevation noise")
    return


@function_timer
def get_breaks(comm, all_ces, args, verbose=True):
    """ List operational day limits in the list of CES:s.

    """
    breaks = []
    if not args.do_daymaps:
        return breaks
    do_break = False
    nces = len(all_ces)
    for i in range(nces - 1):
        # If current and next CES are on different days, insert a break
        tz = args.timezone / 24
        start1 = all_ces[i][3]  # MJD start
        start2 = all_ces[i + 1][3]  # MJD start
        scan1 = all_ces[i][4]
        scan2 = all_ces[i + 1][4]
        if scan1 != scan2 and do_break:
            breaks.append(nces + i + 1)
            do_break = False
            continue
        day1 = int(start1 + tz)
        day2 = int(start2 + tz)
        if day1 != day2:
            if scan1 == scan2:
                # We want an entire CES, even if it crosses the day bound.
                # Wait until the scan number changes.
                do_break = True
            else:
                breaks.append(nces + i + 1)

    nbreak = len(breaks)
    if nbreak < comm.ngroups - 1:
        if comm.comm_world is None or comm.world_rank == 0:
            print(
                "WARNING: there are more process groups than observing days. "
                "Will try distributing by observation.",
                flush=True,
            )
        breaks = []
        for i in range(nces - 1):
            scan1 = all_ces[i][4]
            scan2 = all_ces[i + 1][4]
            if scan1 != scan2:
                breaks.append(nces + i + 1)
        nbreak = len(breaks)

    if nbreak != comm.ngroups - 1:
        raise RuntimeError(
            "Number of observing days ({}) does not match number of process "
            "groups ({}).".format(nbreak + 1, comm.ngroups)
        )
    return breaks


def _parse_line(line, all_ces):
    """ Parse one line of the schedule file
    """
    if line.startswith("#"):
        return

    (
        start_date,
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
    all_ces.append(
        [
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
        ]
    )


@function_timer
def min_sso_dist(el, azmin, azmax, sso_el1, sso_az1, sso_el2, sso_az2):
    """ Return a rough minimum angular distance between the bore sight
    and a solar system object"""
    sso_vec1 = hp.dir2vec(sso_az1, sso_el1, lonlat=True)
    sso_vec2 = hp.dir2vec(sso_az2, sso_el2, lonlat=True)
    az1 = azmin
    az2 = azmax
    if az2 < az1:
        az2 += 360
    n = 100
    az = np.linspace(az1, az2, n)
    el = np.ones(n) * el
    vec = hp.dir2vec(az, el, lonlat=True)
    dist1 = np.degrees(np.arccos(np.dot(sso_vec1, vec)))
    dist2 = np.degrees(np.arccos(np.dot(sso_vec2, vec)))
    return min(np.amin(dist1), np.amin(dist2))


@function_timer
def load_schedule(args, comm):
    """ Load the observing schedule(s).

    Returns:
        schedules (list): List of tuples of the form
            (`site`, `all_ces`) where `all_ces` is
            a list of individual CES objects for `site`.
    """
    schedules = []
    timer0 = Timer()
    timer0.start()

    if comm.comm_world is None or comm.world_rank == 0:
        timer1 = Timer()
        isplit, nsplit = None, None
        if args.split_schedule is not None:
            isplit, nsplit = args.split_schedule.split(",")
            isplit = np.int(isplit)
            nsplit = np.int(nsplit)
            scan_counters = {}
        for fn in args.schedule.split(","):
            if not os.path.isfile(fn):
                raise RuntimeError("No such schedule file: {}".format(fn))
            timer1.start()
            with open(fn, "r") as f:
                while True:
                    line = f.readline()
                    if line.startswith("#"):
                        continue
                    (
                        site_name,
                        telescope_name,
                        site_lat,
                        site_lon,
                        site_alt,
                    ) = line.split()
                    site = Site(site_name, site_lat, site_lon, float(site_alt))
                    telescope = Telescope(telescope_name, site=site)
                    break
                all_ces = []
                last_name = None
                for line in f:
                    if line.startswith("#"):
                        continue
                    (
                        start_date,
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
                    if nsplit:
                        # Only accept 1 / `nsplit` of the rising and setting
                        # scans in patch `name`.  Selection is performed
                        # during the first subscan.
                        if name != last_name:
                            if name not in scan_counters:
                                scan_counters[name] = {}
                            counter = scan_counters[name]
                            # Separate counters for rising and setting scans
                            if rs not in counter:
                                counter[rs] = 0
                            else:
                                counter[rs] += 1
                            iscan = counter[rs]
                        last_name = name
                        if iscan % nsplit != isplit:
                            continue
                    start_time = start_date + " " + start_time
                    stop_time = stop_date + " " + stop_time
                    # Define season as a calendar year.  This can be
                    # changed later and could even be in the schedule file.
                    season = int(start_date.split("-")[0])
                    # Gather other useful metadata
                    mindist_sun = min_sso_dist(
                        *np.array(
                            [el, azmin, azmax, sun_el1, sun_az1, sun_el2, sun_az2]
                        ).astype(np.float)
                    )
                    mindist_moon = min_sso_dist(
                        *np.array(
                            [el, azmin, azmax, moon_el1, moon_az1, moon_el2, moon_az2]
                        ).astype(np.float)
                    )
                    el_sun = max(float(sun_el1), float(sun_el2))
                    try:
                        start_time = dateutil.parser.parse(start_time + " +0000")
                        stop_time = dateutil.parser.parse(stop_time + " +0000")
                    except Exception:
                        start_time = dateutil.parser.parse(start_time)
                        stop_time = dateutil.parser.parse(stop_time)
                    start_timestamp = start_time.timestamp()
                    stop_timestamp = stop_time.timestamp()
                    all_ces.append(
                        CES(
                            start_time=start_timestamp,
                            stop_time=stop_timestamp,
                            name=name,
                            mjdstart=float(mjdstart),
                            scan=int(scan),
                            subscan=int(subscan),
                            azmin=float(azmin),
                            azmax=float(azmax),
                            el=float(el),
                            season=season,
                            start_date=start_date,
                            rising=(rs.upper() == "R"),
                            mindist_sun=mindist_sun,
                            mindist_moon=mindist_moon,
                            el_sun=el_sun,
                        )
                    )
            schedules.append(Schedule(telescope, all_ces, sort=args.sort_schedule))
            timer1.report_clear("Load {} (sub)scans in {}".format(len(all_ces), fn))

    if comm.comm_world is not None:
        schedules = comm.comm_world.bcast(schedules)

    if comm.world_rank == 0:
        timer0.report_clear("Loading schedule(s)")
    return schedules


@function_timer
def load_weather(args, comm, schedules, verbose=False):
    """ Load TOAST weather file(s) and attach them to the sites in the
     schedules.

    Args:
        schedules (iterable) :  list of observing schedules.
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
        schedule.telescope.site.weather = weather

    if comm.world_rank == 0 and verbose:
        timer.report_clear("Loading weather")
