#!/usr/bin/env python3

# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a ground simulation and makes a map.
"""

import os

if "TOAST_STARTUP_DELAY" in os.environ:
    import numpy as np
    import time

    delay = np.float(os.environ["TOAST_STARTUP_DELAY"])
    wait = np.random.rand() * delay
    # print('Sleeping for {} seconds before importing TOAST'.format(wait),
    #      flush=True)
    time.sleep(wait)

import sys
import re
import argparse
import traceback
import copy
import pickle

import dateutil.parser

import numpy as np

from toast.mpi import get_world, Comm

from toast.dist import distribute_uniform, Data

from toast.utils import Logger, Environment

import toast.qarray as qa

from toast.weather import Weather

from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing

from toast.map import OpMadam, OpLocalPixels, DistPixels

from toast.tod import (
    AnalyticNoise,
    OpSimNoise,
    OpPointingHpix,
    OpSimPySM,
    OpMemoryCounter,
    TODGround,
    OpSimScan,
    OpCacheCopy,
    OpGainScrambler,
    OpPolyFilter,
    OpGroundFilter,
    OpSimAtmosphere,
    atm_available_utils,
)

if atm_available_utils:
    from toast.tod.atm import (
        atm_atmospheric_loading,
        atm_absorption_coefficient,
        atm_absorption_coefficient_vec,
    )

# FIXME: put these back into the import statement above after porting.
tidas_available = False
spt3g_available = False

if tidas_available:
    from toast.tod.tidas import OpTidasExport, TODTidas

if spt3g_available:
    from toast.tod.spt3g import Op3GExport, TOD3G


# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

XAXIS, YAXIS, ZAXIS = np.eye(3)


def parse_arguments(comm):
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "atmosphere and make maps for some number of noise Monte Carlos.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--groupsize",
        required=False,
        type=np.int,
        help="Size of a process group assigned to a CES",
    )

    parser.add_argument(
        "--timezone",
        required=False,
        type=np.int,
        default=0,
        help="Offset to apply to MJD to separate days [hours]",
    )

    parser.add_argument(
        "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
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
        "--samplerate",
        required=False,
        default=100.0,
        type=np.float,
        help="Detector sample rate (Hz)",
    )
    parser.add_argument(
        "--scanrate",
        required=False,
        default=1.0,
        type=np.float,
        help="Scanning rate [deg / s]",
    )
    parser.add_argument(
        "--scan_accel",
        required=False,
        default=1.0,
        type=np.float,
        help="Scanning rate change [deg / s^2]",
    )
    parser.add_argument(
        "--sun_angle_min",
        required=False,
        default=30.0,
        type=np.float,
        help="Minimum azimuthal distance between the Sun and " "the bore sight [deg]",
    )

    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        required=False,
        action="store_true",
        help="Conserve memory",
    )
    parser.add_argument(
        "--no_conserve_memory",
        dest="conserve_memory",
        required=False,
        action="store_false",
        help="Do not conserve memory",
    )
    parser.set_defaults(conserve_memory=True)

    parser.add_argument(
        "--polyorder",
        required=False,
        type=np.int,
        help="Polynomial order for the polyfilter",
    )

    parser.add_argument(
        "--wbin_ground",
        required=False,
        type=np.float,
        help="Ground template bin width [degrees]",
    )

    parser.add_argument(
        "--gain_sigma", required=False, type=np.float, help="Gain error distribution"
    )

    parser.add_argument(
        "--hwprpm",
        required=False,
        default=0.0,
        type=np.float,
        help="The rate (in RPM) of the HWP rotation",
    )
    parser.add_argument(
        "--hwpstep",
        required=False,
        default=None,
        help="For stepped HWP, the angle in degrees " "of each step",
    )
    parser.add_argument(
        "--hwpsteptime",
        required=False,
        default=0.0,
        type=np.float,
        help="For stepped HWP, the the time in seconds " "between steps",
    )

    parser.add_argument("--input_map", required=False, help="Input map for signal")
    parser.add_argument(
        "--input_pysm_model",
        required=False,
        help="Comma separated models for on-the-fly PySM "
        'simulation, e.g. s3,d6,f1,a2"',
    )
    parser.add_argument(
        "--apply_beam",
        required=False,
        action="store_true",
        help="Apply beam convolution to input map with "
        "gaussian beam parameters defined in focalplane",
    )

    parser.add_argument(
        "--skip_atmosphere",
        required=False,
        default=False,
        action="store_true",
        help="Disable simulating the atmosphere.",
    )
    parser.add_argument(
        "--skip_noise",
        required=False,
        default=False,
        action="store_true",
        help="Disable simulating detector noise.",
    )
    parser.add_argument(
        "--skip_bin",
        required=False,
        default=False,
        action="store_true",
        help="Disable binning the map.",
    )
    parser.add_argument(
        "--skip_hits",
        required=False,
        default=False,
        action="store_true",
        help="Do not save the 3x3 matrices and hitmaps",
    )
    parser.add_argument(
        "--skip_destripe",
        required=False,
        default=False,
        action="store_true",
        help="Do not destripe the data",
    )
    parser.add_argument(
        "--skip_daymaps",
        required=False,
        default=False,
        action="store_true",
        help="Do not bin daily maps",
    )

    parser.add_argument(
        "--atm_lmin_center",
        required=False,
        default=0.01,
        type=np.float,
        help="Kolmogorov turbulence dissipation scale center",
    )
    parser.add_argument(
        "--atm_lmin_sigma",
        required=False,
        default=0.001,
        type=np.float,
        help="Kolmogorov turbulence dissipation scale sigma",
    )
    parser.add_argument(
        "--atm_lmax_center",
        required=False,
        default=10.0,
        type=np.float,
        help="Kolmogorov turbulence injection scale center",
    )
    parser.add_argument(
        "--atm_lmax_sigma",
        required=False,
        default=10.0,
        type=np.float,
        help="Kolmogorov turbulence injection scale sigma",
    )
    parser.add_argument(
        "--atm_gain",
        required=False,
        default=1e-4,
        type=np.float,
        help="Atmospheric gain factor.",
    )
    parser.add_argument(
        "--atm_zatm",
        required=False,
        default=40000.0,
        type=np.float,
        help="atmosphere extent for temperature profile",
    )
    parser.add_argument(
        "--atm_zmax",
        required=False,
        default=200.0,
        type=np.float,
        help="atmosphere extent for water vapor integration",
    )
    parser.add_argument(
        "--atm_xstep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in X direction",
    )
    parser.add_argument(
        "--atm_ystep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in Y direction",
    )
    parser.add_argument(
        "--atm_zstep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in Z direction",
    )
    parser.add_argument(
        "--atm_nelem_sim_max",
        required=False,
        default=1000,
        type=np.int,
        help="controls the size of the simulation slices",
    )
    parser.add_argument(
        "--atm_gangsize",
        required=False,
        default=1,
        type=np.int,
        help="size of the gangs that create slices",
    )
    parser.add_argument(
        "--atm_wind_time",
        required=False,
        default=36000.0,
        type=np.float,
        help="Maximum time to simulate without discontinuity",
    )
    parser.add_argument(
        "--atm_z0_center",
        required=False,
        default=2000.0,
        type=np.float,
        help="central value of the water vapor distribution",
    )
    parser.add_argument(
        "--atm_z0_sigma",
        required=False,
        default=0.0,
        type=np.float,
        help="sigma of the water vapor distribution",
    )
    parser.add_argument(
        "--atm_T0_center",
        required=False,
        default=280.0,
        type=np.float,
        help="central value of the temperature distribution",
    )
    parser.add_argument(
        "--atm_T0_sigma",
        required=False,
        default=10.0,
        type=np.float,
        help="sigma of the temperature distribution",
    )
    parser.add_argument(
        "--atm_cache",
        required=False,
        default="atm_cache",
        help="Atmosphere cache directory",
    )

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )
    parser.add_argument(
        "--zip",
        required=False,
        default=False,
        action="store_true",
        help="Compress the output fits files",
    )
    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Write diagnostics",
    )
    parser.add_argument(
        "--flush",
        required=False,
        default=False,
        action="store_true",
        help="Flush every print statement.",
    )
    parser.add_argument(
        "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
    )
    parser.add_argument(
        "--madam_prefix", required=False, default="toast", help="Output map prefix"
    )
    parser.add_argument(
        "--madam_iter_max",
        required=False,
        default=1000,
        type=np.int,
        help="Maximum number of CG iterations in Madam",
    )
    parser.add_argument(
        "--madam_baseline_length",
        required=False,
        default=10000.0,
        type=np.float,
        help="Destriping baseline length (seconds)",
    )
    parser.add_argument(
        "--madam_baseline_order",
        required=False,
        default=0,
        type=np.int,
        help="Destriping baseline polynomial order",
    )
    parser.add_argument(
        "--madam_precond_width",
        required=False,
        default=1,
        type=np.int,
        help="Madam preconditioner width",
    )
    parser.add_argument(
        "--madam_noisefilter",
        required=False,
        default=False,
        action="store_true",
        help="Destripe with the noise filter enabled",
    )
    parser.add_argument(
        "--madampar", required=False, default=None, help="Madam parameter file"
    )
    parser.add_argument(
        "--no_madam_allreduce",
        required=False,
        default=False,
        action="store_true",
        help="Do not use allreduce communication in Madam",
    )
    parser.add_argument(
        "--common_flag_mask",
        required=False,
        default=1,
        type=np.uint8,
        help="Common flag mask",
    )
    parser.add_argument(
        "--MC_start",
        required=False,
        default=0,
        type=np.int,
        help="First Monte Carlo noise realization",
    )
    parser.add_argument(
        "--MC_count",
        required=False,
        default=1,
        type=np.int,
        help="Number of Monte Carlo noise realizations",
    )
    parser.add_argument(
        "--fp",
        required=False,
        default=None,
        help="Pickle file containing a dictionary of detector "
        "properties.  The keys of this dict are the detector "
        "names, and each value is also a dictionary with keys "
        '"quat" (4 element ndarray), "fwhm" (float, arcmin), '
        '"fknee" (float, Hz), "alpha" (float), and '
        '"NET" (float).',
    )
    parser.add_argument(
        "--focalplane_radius",
        required=False,
        type=np.float,
        help="Override focal plane radius [deg]",
    )
    parser.add_argument(
        "--freq",
        required=True,
        help="Comma-separated list of frequencies with " "identical focal planes",
    )
    parser.add_argument(
        "--tidas", required=False, default=None, help="Output TIDAS export path"
    )
    parser.add_argument(
        "--spt3g", required=False, default=None, help="Output SPT3G export path"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    if args.tidas is not None:
        if not tidas_available:
            raise RuntimeError("TIDAS not found- cannot export")

    if args.spt3g is not None:
        if not spt3g_available:
            raise RuntimeError("SPT3G not found- cannot export")

    if len(args.freq.split(",")) != 1:
        # Multi frequency run.  We don't support multiple copies of
        # scanned signal.
        if args.input_map:
            raise RuntimeError(
                "Multiple frequencies are not supported when scanning from a map"
            )

    if not args.skip_atmosphere and args.weather is None:
        raise RuntimeError("Cannot simulate atmosphere without a TOAST weather file")

    if comm.world_rank == 0:
        log.info("All parameters:")
        for ag in vars(args):
            log.info("{} = {}".format(ag, getattr(args, ag)))

    if args.groupsize:
        comm = Comm(world=comm.comm_world, groupsize=args.groupsize)

    if comm.world_rank == 0:
        if not os.path.isdir(args.outdir):
            try:
                os.makedirs(args.outdir)
            except FileExistsError:
                pass

    return args, comm


def name2id(name, maxval=2 ** 16):
    """ Map a name into an index.

    """
    value = 0
    for c in name:
        value += ord(c)
    return value % maxval


@function_timer
def load_weather(args, comm, schedules):
    """ Load TOAST weather file(s) and attach them to the schedules.

    """
    if args.weather is None:
        return
    tmr = Timer()
    tmr.start()

    if comm.world_rank == 0:
        weathers = []
        weatherdict = {}
        ftmr = Timer()
        for fname in args.weather.split(","):
            if fname not in weatherdict:
                if not os.path.isfile(fname):
                    raise RuntimeError("No such weather file: {}".format(fname))
                ftmr.start()
                weatherdict[fname] = Weather(fname)
                ftmr.stop()
                ftmr.report_clear("Load {}".format(fname))
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

    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Loading weather")
    return


@function_timer
def load_schedule(args, comm):
    """ Load the observing schedule(s).

    """
    schedules = []
    tmr = Timer()
    tmr.start()

    if comm.world_rank == 0:
        ftmr = Timer()
        for fn in args.schedule.split(","):
            if not os.path.isfile(fn):
                raise RuntimeError("No such schedule file: {}".format(fn))
            ftmr.start()
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
            schedules.append([site, all_ces])
            ftmr.stop()
            ftmr.report_clear("Load {}".format(fn))

    schedules = comm.comm_world.bcast(schedules)

    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Loading schedule")
    return schedules


@function_timer
def get_focalplane_radius(args, focalplane, rmin=1.0):
    """ Find the furthest angular distance from the boresight

    """
    if args.focalplane_radius:
        return args.focalplane_radius

    cosangs = []
    for det in focalplane:
        quat = focalplane[det]["quat"]
        vec = qa.rotate(quat, ZAXIS)
        cosangs.append(np.dot(ZAXIS, vec))
    mincos = np.amin(cosangs)
    maxdist = max(np.degrees(np.arccos(mincos)), rmin)
    return maxdist * 1.001


@function_timer
def load_focalplanes(args, comm, schedules):
    """ Attach a focalplane to each of the schedules.

    """
    tmr = Timer()
    tmr.start()

    # Load focalplane information

    focalplanes = []
    if comm.world_rank == 0:
        ftmr = Timer()
        for fpfile in args.fp.split(","):
            ftmr.start()
            with open(fpfile, "rb") as picklefile:
                focalplane = pickle.load(picklefile)
                focalplanes.append(focalplane)
                ftmr.report_clear("Load {}".format(fpfile))
        ftmr.stop()
    focalplanes = comm.comm_world.bcast(focalplanes)

    if len(focalplanes) == 1 and len(schedules) > 1:
        focalplanes *= len(schedules)
    if len(focalplanes) != len(schedules):
        raise RuntimeError(
            "Number of focalplanes must equal number of schedules or be 1."
        )

    detweights = {}
    for schedule, focalplane in zip(schedules, focalplanes):
        schedule.append(focalplane)
        for detname, det in focalplane.items():
            net = det["NET"]
            detweight = 1.0 / (args.samplerate * net * net)
            if detname in detweights and detweights[detname] != detweight:
                raise RuntimeError("Detector weight for {} changes".format(detname))
            detweights[detname] = detweight

    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Loading focalplanes")
    return detweights


@function_timer
def get_analytic_noise(args, comm, focalplane):
    """Create a TOAST noise object.

    Create a noise object from the 1/f noise parameters contained in the
    focalplane database.

    """
    tmr = Timer()
    tmr.start()
    detectors = sorted(focalplane.keys())
    fmin = {}
    fknee = {}
    alpha = {}
    NET = {}
    rates = {}
    for d in detectors:
        rates[d] = args.samplerate
        fmin[d] = focalplane[d]["fmin"]
        fknee[d] = focalplane[d]["fknee"]
        alpha[d] = focalplane[d]["alpha"]
        NET[d] = focalplane[d]["NET"]
    nse = AnalyticNoise(
        rate=rates, fmin=fmin, detectors=detectors, fknee=fknee, alpha=alpha, NET=NET
    )
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Creating noise model")
    return nse


@function_timer
def get_breaks(comm, all_ces, nces, args):
    """ List operational day limits in the list of CES:s.

    """
    breaks = []
    if args.skip_daymaps:
        return breaks
    do_break = False
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
        if comm.world_rank == 0:
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


@function_timer
def create_observation(args, comm, all_ces_tot, ices, noise):
    """ Create a TOAST observation.

    Create an observation for the CES scan defined by all_ces_tot[ices].

    """
    ces, site, fp, fpradius, detquats, weather = all_ces_tot[ices]

    (
        CES_start,
        CES_stop,
        CES_name,
        mjdstart,
        scan,
        subscan,
        azmin,
        azmax,
        el,
        season,
        date,
    ) = ces

    _, _, site_lat, site_lon, site_alt = site

    totsamples = int((CES_stop - CES_start) * args.samplerate)

    # create the TOD for this observation

    try:
        tod = TODGround(
            comm.comm_group,
            detquats,
            totsamples,
            detranks=comm.comm_group.size,
            firsttime=CES_start,
            rate=args.samplerate,
            site_lon=site_lon,
            site_lat=site_lat,
            site_alt=site_alt,
            azmin=azmin,
            azmax=azmax,
            el=el,
            scanrate=args.scanrate,
            scan_accel=args.scan_accel,
            CES_start=None,
            CES_stop=None,
            sun_angle_min=args.sun_angle_min,
            coord=args.coord,
            sampsizes=None,
        )
    except RuntimeError as e:
        raise RuntimeError(
            'Failed to create TOD for {}-{}-{}: "{}"'
            "".format(CES_name, scan, subscan, e)
        )

    # Create the observation

    site_name = site[0]
    telescope_name = site[1]
    site_id = name2id(site_name)
    telescope_id = name2id(telescope_name)

    obs = {}
    obs["name"] = "CES-{}-{}-{}-{}-{}".format(
        site_name, telescope_name, CES_name, scan, subscan
    )
    obs["tod"] = tod
    obs["baselines"] = None
    obs["noise"] = noise
    obs["id"] = int(mjdstart * 10000)
    obs["intervals"] = tod.subscans
    obs["site"] = site_name
    obs["telescope"] = telescope_name
    obs["site_id"] = site_id
    obs["telescope_id"] = telescope_id
    obs["fpradius"] = fpradius
    obs["weather"] = weather
    obs["start_time"] = CES_start
    obs["altitude"] = site_alt
    obs["season"] = season
    obs["date"] = date
    obs["MJD"] = mjdstart
    obs["focalplane"] = fp
    return obs


@function_timer
def create_observations(args, comm, schedules, mem_counter):
    """ Create and distribute TOAST observations for every CES in schedules.

    """
    log = Logger.get()
    tmr = Timer()
    tmr.start()

    data = Data(comm)

    # Loop over the schedules, distributing each schedule evenly across
    # the process groups.  For now, we'll assume that each schedule has
    # the same number of operational days and the number of process groups
    # matches the number of operational days.  Relaxing these constraints
    # will cause the season break to occur on different process groups
    # for different schedules and prevent splitting the communicator.

    for schedule in schedules:

        if args.weather is None:
            site, all_ces, focalplane = schedule
            weather = None
        else:
            site, all_ces, weather, focalplane = schedule

        fpradius = get_focalplane_radius(args, focalplane)

        # Focalplane information for this schedule
        detectors = sorted(focalplane.keys())
        detquats = {}
        for d in detectors:
            detquats[d] = focalplane[d]["quat"]

        # Noise model for this schedule
        noise = get_analytic_noise(args, comm, focalplane)

        all_ces_tot = []
        nces = len(all_ces)
        for ces in all_ces:
            all_ces_tot.append((ces, site, focalplane, fpradius, detquats, weather))

        breaks = get_breaks(comm, all_ces, nces, args)

        groupdist = distribute_uniform(nces, comm.ngroups, breaks=breaks)
        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for ices in range(group_firstobs, group_firstobs + group_numobs):
            obs = create_observation(args, comm, all_ces_tot, ices, noise)
            data.obs.append(obs)

    if args.skip_atmosphere:
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_azel_quats()

    if comm.group_rank == 0:
        log.info("Group # {:4} has {} observations.".format(comm.group, len(data.obs)))

    if len(data.obs) == 0:
        raise RuntimeError(
            "Too many tasks. Every MPI task must "
            "be assigned to at least one observation."
        )

    mem_counter.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Simulated scans")

    # Split the data object for each telescope for separate mapmaking.
    # We could also split by site.

    if len(schedules) > 1:
        telescope_data = data.split("telescope")
        if len(telescope_data) == 1:
            # Only one telescope available
            telescope_data = []
    else:
        telescope_data = []
    telescope_data.insert(0, ("all", data))
    return data, telescope_data


@function_timer
def expand_pointing(args, comm, data, mem_counter):
    """ Expand boresight pointing to every detector.

    """
    log = Logger.get()
    tmr = Timer()
    tmr.start()

    hwprpm = args.hwprpm
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)
    hwpsteptime = args.hwpsteptime

    if comm.world_rank == 0:
        log.info("Expanding pointing")

    pointing = OpPointingHpix(
        nside=args.nside,
        nest=True,
        mode="IQU",
        hwprpm=hwprpm,
        hwpstep=hwpstep,
        hwpsteptime=hwpsteptime,
    )

    pointing.exec(data)

    # Only purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if (args.tidas is None) and (args.spt3g is None):
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_radec_quats()

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Pointing generation")

    mem_counter.exec(data)
    return


@function_timer
def get_submaps(args, comm, data):
    """ Get a list of locally hit pixels and submaps on every process.

    """
    log = Logger.get()

    if not args.skip_bin or args.input_map:
        if comm.world_rank == 0:
            log.info("Scanning local pixels")
        tmr = Timer()
        tmr.start()

        # Prepare for using distpixels objects
        nside = args.nside
        subnside = 16
        if subnside > nside:
            subnside = nside
        subnpix = 12 * subnside * subnside

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(data)
        if localpix is None:
            raise RuntimeError(
                "Process {} has no hit pixels. Perhaps there are fewer "
                "detectors than processes in the group?".format(comm.world_rank)
            )

        # find the locally hit submaps.
        localsm = np.unique(np.floor_divide(localpix, subnpix))

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        tmr.stop()
        if comm.world_rank == 0:
            tmr.report("Identify local submaps")
    else:
        localpix, localsm = None, None
    return localpix, localsm, subnpix


@function_timer
def add_sky_signal(data, totalname_freq, signalname):
    """ Add previously simulated sky signal to the atmospheric noise.

    """
    if signalname is not None:
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                cachename_in = "{}_{}".format(signalname, det)
                cachename_out = "{}_{}".format(totalname_freq, det)
                ref_in = tod.cache.reference(cachename_in)
                if tod.cache.exists(cachename_out):
                    ref_out = tod.cache.reference(cachename_out)
                    ref_out += ref_in
                else:
                    ref_out = tod.cache.put(cachename_out, ref_in)
                del ref_in, ref_out
    return


@function_timer
def simulate_sky_signal(args, comm, data, mem_counter, schedules, subnpix, localsm):
    """ Use PySM to simulate smoothed sky signal.

    """
    tmr = Timer()
    tmr.start()
    # Convolve a signal TOD from PySM
    signalname = "signal"
    op_sim_pysm = OpSimPySM(
        comm=comm.comm_rank,
        out=signalname,
        pysm_model=args.input_pysm_model.split(","),
        focalplanes=[s[3] for s in schedules],
        nside=args.nside,
        subnpix=subnpix,
        localsm=localsm,
        apply_beam=args.apply_beam,
        coord=args.coord,
    )
    op_sim_pysm.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("PySM")

    mem_counter.exec(data)
    return signalname


@function_timer
def scan_sky_signal(args, comm, data, mem_counter, localsm, subnpix):
    """ Scan sky signal from a map.

    """
    log = Logger.get()
    tmr = Timer()
    tmr.start()

    signalname = None

    if args.input_map:
        if comm.world_rank == 0:
            log.info("Scanning input map")

        npix = 12 * args.nside ** 2

        # Scan the sky signal
        if comm.world_rank == 0 and not os.path.isfile(args.input_map):
            raise RuntimeError("Input map does not exist: {}".format(args.input_map))
        distmap = DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=3,
            dtype=np.float32,
            submap=subnpix,
            local=localsm,
        )
        mem_counter._objects.append(distmap)
        distmap.read_healpix_fits(args.input_map)
        scansim = OpSimScan(distmap=distmap, out="signal")
        scansim.exec(data)
        signalname = "signal"
        mem_counter.exec(data)

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        tmr.stop()
        if comm.world_rank == 0:
            tmr.report("Read and sample map")

    return signalname


def setup_sigcopy(args):
    """ Determine if an extra copy of the atmospheric signal is needed.

    When we simulate multichroic focal planes, the frequency-independent
    part of the atmospheric noise is simulated first and then the
    frequency scaling is applied to a copy of the atmospheric noise.
    """
    if len(args.freq.split(",")) == 1:
        totalname = "total"
        totalname_freq = "total"
    else:
        totalname = "total"
        totalname_freq = "total_freq"

    return totalname, totalname_freq


@function_timer
def setup_madam(args):
    """ Create a Madam parameter dictionary.

    Initialize the Madam parameters from the command line arguments.

    """
    pars = {}

    cross = args.nside // 2
    submap = 16
    if submap > args.nside:
        submap = args.nside

    pars["temperature_only"] = False
    pars["force_pol"] = True
    pars["kfirst"] = not args.skip_destripe
    pars["write_map"] = not args.skip_destripe
    pars["write_binmap"] = not args.skip_bin
    pars["write_matrix"] = not args.skip_hits
    pars["write_wcov"] = not args.skip_hits
    pars["write_hits"] = not args.skip_hits
    pars["nside_cross"] = cross
    pars["nside_submap"] = submap
    if args.no_madam_allreduce:
        pars["allreduce"] = False
    else:
        pars["allreduce"] = True
    pars["reassign_submaps"] = True
    pars["pixlim_cross"] = 1e-3
    pars["pixmode_cross"] = 2
    pars["pixlim_map"] = 1e-2
    pars["pixmode_map"] = 2
    # Instead of fixed detector weights, we'll want to use scaled noise
    # PSD:s that include the atmospheric noise
    pars["radiometers"] = True
    pars["noise_weights_from_psd"] = True

    if args.madampar is not None:
        pat = re.compile(r"\s*(\S+)\s*=\s*(\S+(\s+\S+)*)\s*")
        comment = re.compile(r"^#.*")
        with open(args.madampar, "r") as f:
            for line in f:
                if comment.match(line) is None:
                    result = pat.match(line)
                    if result is not None:
                        key, value = result.group(1), result.group(2)
                        pars[key] = value

    pars["base_first"] = args.madam_baseline_length
    pars["basis_order"] = args.madam_baseline_order
    pars["nside_map"] = args.nside
    if args.madam_noisefilter:
        if args.madam_baseline_order != 0:
            raise RuntimeError(
                "Madam cannot build a noise filter when baseline"
                "order is higher than zero."
            )
        pars["kfilter"] = True
    else:
        pars["kfilter"] = False
    pars["precond_width"] = args.madam_precond_width
    pars["fsample"] = args.samplerate
    pars["iter_max"] = args.madam_iter_max
    pars["file_root"] = args.madam_prefix
    return pars


@function_timer
def scale_atmosphere_by_frequency(args, comm, data, freq, totalname_freq, mc):
    """Scale atmospheric fluctuations by frequency.

    Assume that cached signal under totalname_freq is pure atmosphere
    and scale the absorption coefficient according to the frequency.

    If the focalplane is included in the observation and defines
    bandpasses for the detectors, the scaling is computed for each
    detector separately.

    """
    if args.skip_atmosphere:
        return

    tmr = Timer()
    tmr.start()
    for obs in data.obs:
        tod = obs["tod"]
        todcomm = tod.mpicomm
        site_id = obs["site_id"]
        weather = obs["weather"]
        if "focalplane" in obs:
            focalplane = obs["focalplane"]
        else:
            focalplane = None
        start_time = obs["start_time"]
        weather.set(site_id, mc, start_time)
        altitude = obs["altitude"]
        air_temperature = weather.air_temperature
        surface_pressure = weather.surface_pressure
        pwv = weather.pwv
        # Use the entire processing group to sample the absorption
        # coefficient as a function of frequency
        freqmin = 0
        freqmax = 2 * freq
        nfreq = 1001
        freqstep = (freqmax - freqmin) / (nfreq - 1)
        nfreq_task = int(nfreq // todcomm.size) + 1
        my_ifreq_min = nfreq_task * todcomm.rank
        my_ifreq_max = min(nfreq, nfreq_task * (todcomm.rank + 1))
        my_nfreq = my_ifreq_max - my_ifreq_min
        if my_nfreq > 0:
            if atm_available_utils:
                my_freqs = freqmin + np.arange(my_ifreq_min, my_ifreq_max) * freqstep
                my_absorption = atm_absorption_coefficient_vec(
                    altitude,
                    air_temperature,
                    surface_pressure,
                    pwv,
                    my_freqs[0],
                    my_freqs[-1],
                    my_nfreq,
                )
            else:
                raise RuntimeError(
                    "Atmosphere utilities from libaatm are not available"
                )
        else:
            my_freqs = np.array([])
            my_absorption = np.array([])
        freqs = np.hstack(todcomm.allgather(my_freqs))
        absorption = np.hstack(todcomm.allgather(my_absorption))
        # loading = atm_atmospheric_loading(altitude, pwv, freq)
        for det in tod.local_dets:
            try:
                # Use detector bandpass from the focalplane
                center = focalplane[det]["bandcenter_ghz"]
                width = focalplane[det]["bandwidth_ghz"]
            except Exception:
                # Use default values for the entire focalplane
                center = freq
                width = 0.2 * freq
            nstep = 101
            # Interpolate the absorption coefficient to do a top hat
            # integral across the bandpass
            det_freqs = np.linspace(center - width / 2, center + width / 2, nstep)
            absorption_det = np.mean(np.interp(det_freqs, freqs, absorption))
            cachename = "{}_{}".format(totalname_freq, det)
            ref = tod.cache.reference(cachename)
            ref *= absorption_det
            del ref

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Atmosphere scaling")
    return


@function_timer
def update_atmospheric_noise_weights(args, comm, data, freq, mc):
    """Update atmospheric noise weights.

    Estimate the atmospheric noise level from weather parameters and
    encode it as a noise_scale in the observation.  Madam will apply
    the noise_scale to the detector weights.  This approach assumes
    that the atmospheric noise dominates over detector noise.  To be
    more precise, we would have to add the squared noise weights but
    we do not have their relative calibration.

    """
    tmr = Timer()
    tmr.start()
    if args.weather and not args.skip_atmosphere:
        if atm_available_utils:
            for obs in data.obs:
                site_id = obs["site_id"]
                weather = obs["weather"]
                start_time = obs["start_time"]
                weather.set(site_id, mc, start_time)
                altitude = obs["altitude"]
                absorption = atm_absorption_coefficient(
                    altitude,
                    weather.air_temperature,
                    weather.surface_pressure,
                    weather.pwv,
                    freq,
                )
                obs["noise_scale"] = absorption * weather.air_temperature
        else:
            raise RuntimeError("Atmosphere utilities from libaatm are not available")
    else:
        for obs in data.obs:
            obs["noise_scale"] = 1.0

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Atmosphere weighting")
    return


@function_timer
def simulate_atmosphere(args, comm, data, mc, mem_counter, totalname):
    log = Logger.get()
    tmr = Timer()
    tmr.start()
    if not args.skip_atmosphere:
        if comm.world_rank == 0:
            log.info("Simulating atmosphere")
            if args.atm_cache and not os.path.isdir(args.atm_cache):
                try:
                    os.makedirs(args.atm_cache)
                except FileExistsError:
                    pass

        # Simulate the atmosphere signal
        atm = OpSimAtmosphere(
            out=totalname,
            realization=mc,
            lmin_center=args.atm_lmin_center,
            lmin_sigma=args.atm_lmin_sigma,
            lmax_center=args.atm_lmax_center,
            gain=args.atm_gain,
            lmax_sigma=args.atm_lmax_sigma,
            zatm=args.atm_zatm,
            zmax=args.atm_zmax,
            xstep=args.atm_xstep,
            ystep=args.atm_ystep,
            zstep=args.atm_zstep,
            nelem_sim_max=args.atm_nelem_sim_max,
            verbosity=int(args.debug),
            gangsize=args.atm_gangsize,
            z0_center=args.atm_z0_center,
            z0_sigma=args.atm_z0_sigma,
            apply_flags=False,
            common_flag_mask=args.common_flag_mask,
            cachedir=args.atm_cache,
            flush=args.flush,
            wind_time=args.atm_wind_time,
        )
        atm.exec(data)
        mem_counter.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Atmosphere simulation")
    return


@function_timer
def copy_atmosphere(args, comm, data, mem_counter, totalname, totalname_freq):
    """Copy the atmospheric signal.

    Make a copy of the atmosphere so we can scramble the gains and apply
    frequency-dependent scaling.

    """
    log = Logger.get()
    if totalname != totalname_freq:
        if comm.world_rank == 0:
            log.info(
                "Copying atmosphere from {} to {}".format(totalname, totalname_freq)
            )
        cachecopy = OpCacheCopy(totalname, totalname_freq, force=True)
        cachecopy.exec(data)
        mem_counter.exec(data)
    return


@function_timer
def simulate_noise(args, comm, data, mc, mem_counter, totalname_freq):
    log = Logger.get()
    tmr = Timer()

    if not args.skip_noise:
        tmr.start()
        if comm.world_rank == 0:
            log.info("Simulating noise")
        nse = OpSimNoise(out=totalname_freq, realization=mc)
        nse.exec(data)

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        tmr.stop()
        if comm.world_rank == 0:
            tmr.report("Simulate noise")
        mem_counter.exec(data)
    return


@function_timer
def scramble_gains(args, comm, data, mc, mem_counter, totalname_freq):
    log = Logger.get()
    tmr = Timer()

    if args.gain_sigma:
        tmr.start()
        if comm.world_rank == 0:
            log.info("Scrambling gains")
        scrambler = OpGainScrambler(
            sigma=args.gain_sigma, name=totalname_freq, realization=mc
        )
        scrambler.exec(data)

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        tmr.stop()
        if comm.world_rank == 0:
            tmr.report("Scramble gains")
        mem_counter.exec(data)
    return


@function_timer
def setup_output(args, comm, mc, freq):
    outpath = "{}/{:08}/{:03}".format(args.outdir, mc, int(freq))
    if comm.world_rank == 0:
        if not os.path.isdir(outpath):
            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass
    return outpath


@function_timer
def apply_polyfilter(args, comm, data, mem_counter, totalname_freq):
    log = Logger.get()
    tmr = Timer()
    if args.polyorder:
        tmr.start()
        if comm.world_rank == 0:
            log.info("Polyfiltering signal")
        polyfilter = OpPolyFilter(
            order=args.polyorder,
            name=totalname_freq,
            common_flag_mask=args.common_flag_mask,
        )
        polyfilter.exec(data)
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        tmr.stop()
        if comm.world_rank == 0:
            tmr.report("Polynomial filtering")

        mem_counter.exec(data)
    return


@function_timer
def apply_groundfilter(args, comm, data, mem_counter, totalname_freq):
    log = Logger.get()
    tmr = Timer()
    if args.wbin_ground:
        tmr.start()
        if comm.world_rank == 0:
            log.info("Ground filtering signal")
        groundfilter = OpGroundFilter(
            wbin=args.wbin_ground,
            name=totalname_freq,
            common_flag_mask=args.common_flag_mask,
        )
        groundfilter.exec(data)

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        tmr.stop()
        if comm.world_rank == 0:
            tmr.report("Ground filtering")

        mem_counter.exec(data)
    return


@function_timer
def output_tidas(args, comm, data, totalname):
    if args.tidas is None:
        return
    log = Logger.get()
    tmr = Timer()
    tidas_path = os.path.abspath(args.tidas)

    if comm.world_rank == 0:
        log.info("Exporting data to a TIDAS volume at {}".format(tidas_path))

    tmr.start()
    export = OpTidasExport(
        tidas_path,
        TODTidas,
        backend="hdf5",
        use_intervals=True,
        create_opts={"group_dets": "sim"},
        ctor_opts={"group_dets": "sim"},
        cache_name=totalname,
    )
    export.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("TIDAS export")
    return


@function_timer
def output_spt3g(args, comm, data, totalname):
    if args.spt3g is None:
        return
    log = Logger.get()
    tmr = Timer()

    spt3g_path = os.path.abspath(args.spt3g)

    if comm.world_rank == 0:
        log.info("Exporting data to a SPT3G directory tree at {}".format(spt3g_path))

    tmr.start()
    export = Op3GExport(
        spt3g_path,
        TOD3G,
        use_intervals=True,
        export_opts={"prefix": "sim"},
        cache_name=totalname,
    )
    export.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("spt3g export")
    return


@function_timer
def get_time_communicators(comm, data):
    """ Split the world communicator by time.

    """
    time_comms = [("all", comm.comm_world)]

    # A process will only have data for one season and one day.  If more
    # than one season is observed, we split the communicator to make
    # season maps.

    my_season = data.obs[0]["season"]
    seasons = np.array(comm.comm_world.allgather(my_season))
    do_seasons = np.any(seasons != my_season)
    if do_seasons:
        season_comm = comm.comm_world.Split(my_season, comm.world_rank)
        time_comms.append((str(my_season), season_comm))

    # Split the communicator to make daily maps.  We could easily split
    # by month as well

    my_day = int(data.obs[0]["MJD"])
    my_date = data.obs[0]["date"]
    days = np.array(comm.comm_world.allgather(my_day))
    do_days = np.any(days != my_day)
    if do_days:
        day_comm = comm.comm_world.Split(my_day, comm.world_rank)
        time_comms.append((my_date, day_comm))

    return time_comms


@function_timer
def apply_madam(
    args,
    comm,
    time_comms,
    data,
    telescope_data,
    freq,
    madampars,
    mem_counter,
    mc,
    firstmc,
    outpath,
    detweights,
    totalname_madam,
    first_call=True,
    extra_prefix=None,
):
    """ Use libmadam to bin and optionally destripe data.

    Bin and optionally destripe all conceivable subsets of the data.

    """
    log = Logger.get()
    tmr = Timer()
    tmr.start()
    if comm.world_rank == 0:
        log.info("Making maps")

    pars = copy.deepcopy(madampars)
    pars["path_output"] = outpath
    file_root = pars["file_root"]
    if len(file_root) > 0 and not file_root.endswith("_"):
        file_root += "_"
    if extra_prefix is not None:
        file_root += "{}_".format(extra_prefix)
    file_root += "{:03}".format(int(freq))

    if first_call:
        if mc != firstmc:
            pars["write_matrix"] = False
            pars["write_wcov"] = False
            pars["write_hits"] = False
    else:
        pars["kfirst"] = False
        pars["write_map"] = False
        pars["write_binmap"] = True
        pars["write_matrix"] = False
        pars["write_wcov"] = False
        pars["write_hits"] = False

    outputs = [
        pars["write_map"],
        pars["write_binmap"],
        pars["write_hits"],
        pars["write_wcov"],
        pars["write_matrix"],
    ]
    if not np.any(outputs):
        if comm.world_rank == 0:
            log.info("No Madam outputs requested.  Skipping.")
        return

    if args.madam_noisefilter or not pars["kfirst"]:
        madam_intervals = None
    else:
        madam_intervals = "intervals"

    madam = OpMadam(
        params=pars,
        detweights=detweights,
        name=totalname_madam,
        common_flag_mask=args.common_flag_mask,
        purge_tod=False,
        intervals=madam_intervals,
        conserve_memory=args.conserve_memory,
    )

    if "info" in madam.params:
        info = madam.params["info"]
    else:
        info = 3

    ttmr = Timer()
    for time_name, time_comm in time_comms:
        for tele_name, tele_data in telescope_data:
            if len(time_name.split("-")) == 3:
                # Special rules for daily maps
                if args.skip_daymaps:
                    continue
                if (len(telescope_data) > 1) and (tele_name == "all"):
                    # Skip daily maps over multiple telescopes
                    continue
                if first_call:
                    # Do not destripe daily maps
                    kfirst_save = pars["kfirst"]
                    write_map_save = pars["write_map"]
                    write_binmap_save = pars["write_binmap"]
                    pars["kfirst"] = False
                    pars["write_map"] = False
                    pars["write_binmap"] = True

            ttmr.start()
            madam.params["file_root"] = "{}_telescope_{}_time_{}".format(
                file_root, tele_name, time_name
            )
            if time_comm == comm.comm_world:
                madam.params["info"] = info
            else:
                # Cannot have verbose output from concurrent mapmaking
                madam.params["info"] = 0
            if time_comm.rank == 0:
                log.info("Mapping {}".format(madam.params["file_root"]))
            madam.exec(tele_data, time_comm)

            time_comm.barrier()
            if comm.world_rank == 0:
                ttmr.report_clear("Mapping {}".format(madam.params["file_root"]))

            if len(time_name.split("-")) == 3 and first_call:
                # Restore destriping parameters
                pars["kfirst"] = kfirst_save
                pars["write_map"] = write_map_save
                pars["write_binmap"] = write_binmap_save

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Madam total")

    mem_counter.exec(data)
    return


def main():
    env = Environment.get()
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_ground_sim (total)")

    mpiworld, procs, rank = get_world()
    if rank == 0:
        env.print()
    if mpiworld is None:
        log.info("Running serially with one process at {}".format(str(datetime.now())))
    else:
        if rank == 0:
            log.info(
                "Running with {} processes at {}".format(procs, str(datetime.now()))
            )

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = Comm(world=mpiworld)

    args, comm = parse_arguments(comm)

    # Initialize madam parameters

    madampars = setup_madam(args)

    # Load and broadcast the schedule file

    schedules = load_schedule(args, comm)

    # Load the weather and append to schedules

    load_weather(args, comm, schedules)

    # load or simulate the focalplane

    detweights = load_focalplanes(args, comm, schedules)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    mem_counter = OpMemoryCounter()

    data, telescope_data = create_observations(args, comm, schedules, mem_counter)

    # Split the communicator for day and season mapmaking

    time_comms = get_time_communicators(comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data, mem_counter)

    # Prepare auxiliary information for distributed map objects

    _, localsm, subnpix = get_submaps(args, comm, data)

    if args.input_pysm_model:
        signalname = simulate_sky_signal(
            args, comm, data, mem_counter, schedules, subnpix, localsm
        )
    else:
        signalname = scan_sky_signal(args, comm, data, mem_counter, localsm, subnpix)

    # Set up objects to take copies of the TOD at appropriate times

    totalname, totalname_freq = setup_sigcopy(args)

    # Loop over Monte Carlos

    firstmc = int(args.MC_start)
    nmc = int(args.MC_count)

    freqs = [float(freq) for freq in args.freq.split(",")]
    nfreq = len(freqs)

    for mc in range(firstmc, firstmc + nmc):

        simulate_atmosphere(args, comm, data, mc, mem_counter, totalname)

        # Loop over frequencies with identical focal planes and identical
        # atmospheric noise.

        for ifreq, freq in enumerate(freqs):

            if rank == 0:
                log.info(
                    "Processing frequency {}GHz {} / {}, MC = {}".format(
                        freq, ifreq + 1, nfreq, mc
                    )
                )

            copy_atmosphere(args, comm, data, mem_counter, totalname, totalname_freq)

            scale_atmosphere_by_frequency(args, comm, data, freq, totalname_freq, mc)

            update_atmospheric_noise_weights(args, comm, data, freq, mc)

            add_sky_signal(data, totalname_freq, signalname)

            mcoffset = ifreq * 1000000

            simulate_noise(args, comm, data, mc + mcoffset, mem_counter, totalname_freq)

            scramble_gains(args, comm, data, mc + mcoffset, mem_counter, totalname_freq)

            if (mc == firstmc) and (ifreq == 0):
                # For the first realization and frequency, optionally
                # export the timestream data.
                output_tidas(args, comm, data, totalname)
                output_spt3g(args, comm, data, totalname)

            outpath = setup_output(args, comm, mc, freq)

            # Bin and destripe maps

            apply_madam(
                args,
                comm,
                time_comms,
                data,
                telescope_data,
                freq,
                madampars,
                mem_counter,
                mc + mcoffset,
                firstmc,
                outpath,
                detweights,
                totalname_freq,
                first_call=True,
            )

            if args.polyorder or args.wbin_ground:

                # Filter signal

                apply_polyfilter(args, comm, data, mem_counter, totalname_freq)

                apply_groundfilter(args, comm, data, mem_counter, totalname_freq)

                # Bin maps

                apply_madam(
                    args,
                    comm,
                    time_comms,
                    data,
                    telescope_data,
                    freq,
                    madampars,
                    mem_counter,
                    mc + mcoffset,
                    firstmc,
                    outpath,
                    detweights,
                    totalname_freq,
                    first_call=False,
                    extra_prefix="filtered",
                )

    mem_counter.exec(data)

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()
    tmr = Timer()
    tmr.start()
    alltimers = gather_timers(comm=mpiworld)
    if rank == 0:
        out = os.path.join(args.outdir, "timing")
        dump_timing(alltimers, out)
        tmr.stop()
        tmr.report("Gather and dump timing info")
    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None:
            mpiworld.Abort(6)
