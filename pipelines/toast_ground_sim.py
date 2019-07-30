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
import argparse
from datetime import datetime
import traceback
import pickle

import numpy as np

from toast.mpi import get_world, Comm

from toast.dist import distribute_uniform, Data

from toast.utils import Logger, Environment

from toast.weather import Weather

from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing

from toast.tod import (
    TODGround,
)

from toast.pipeline_tools import (
    add_polyfilter_args,
    apply_polyfilter,
    add_groundfilter_args,
    apply_groundfilter,
    add_atmosphere_args,
    simulate_atmosphere,
    scale_atmosphere_by_frequency,
    update_atmospheric_noise_weights,
    get_focalplane_radius,
    add_noise_args,
    simulate_noise,
    get_analytic_noise,
    add_gainscrambler_args,
    scramble_gains,
    add_pointing_args,
    expand_pointing,
    get_submaps,
    add_madam_args,
    setup_madam,
    apply_madam,
    add_sky_args,
    scan_sky_signal,
    simulate_sky_signal,
    add_sss_args,
    simulate_sss,
    add_signal,
    copy_signal,
    add_tidas_args,
    output_tidas,
    add_spt3g_args,
    output_spt3g,
    add_todground_args,
    load_schedule,
    load_weather,
    add_mc_args,
)

# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

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

    add_todground_args(parser)
    add_pointing_args(parser)
    add_polyfilter_args(parser)
    add_groundfilter_args(parser)
    add_atmosphere_args(parser)
    add_noise_args(parser)
    add_gainscrambler_args(parser)
    add_madam_args(parser)
    add_sky_args(parser)
    add_sss_args(parser)
    add_tidas_args(parser)
    add_spt3g_args(parser)

    parser.add_argument(
        "--day-maps",
        required=False,
        action="store_true",
        help="Bin daily maps",
        dest="do_daymaps",
    )
    parser.add_argument(
        "--no-day-maps",
        required=False,
        action="store_false",
        help="Bin daily maps",
        dest="do_daymaps",
    )
    parser.set_defaults(do_daymaps=False)

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )
    
    add_mc_args(parser)
    parser.add_argument(
        "--focalplane",
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
        "--freq",
        required=True,
        help="Comma-separated list of frequencies with identical focal planes",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

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

    if args.simulate_atmosphere and args.weather is None:
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
def load_focalplanes(args, comm, schedules):
    """ Attach a focalplane to each of the schedules.

    """
    timer = Timer()
    timer.start()

    # Load focalplane information

    focalplanes = []
    if comm.world_rank == 0:
        ftimer = Timer()
        for fpfile in args.focalplane.split(","):
            ftimer.start()
            with open(fpfile, "rb") as picklefile:
                focalplane = pickle.load(picklefile)
                focalplanes.append(focalplane)
                ftimer.report_clear("Load {}".format(fpfile))
        ftimer.stop()
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
            detweight = 1.0 / (args.sample_rate * net * net)
            if detname in detweights and detweights[detname] != detweight:
                raise RuntimeError("Detector weight for {} changes".format(detname))
            detweights[detname] = detweight

    timer.stop()
    if comm.world_rank == 0:
        timer.report("Loading focalplanes")
    return detweights


@function_timer
def get_breaks(comm, all_ces, nces, args):
    """ List operational day limits in the list of CES:s.

    """
    breaks = []
    if not args.do_daymaps:
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

    totsamples = int((CES_stop - CES_start) * args.sample_rate)

    # create the TOD for this observation

    try:
        tod = TODGround(
            comm.comm_group,
            detquats,
            totsamples,
            detranks=comm.comm_group.size,
            firsttime=CES_start,
            rate=args.sample_rate,
            site_lon=site_lon,
            site_lat=site_lat,
            site_alt=site_alt,
            azmin=azmin,
            azmax=azmax,
            el=el,
            scanrate=args.scan_rate,
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
def create_observations(args, comm, schedules):
    """ Create and distribute TOAST observations for every CES in schedules.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

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

    if not args.simulate_atmosphere:
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

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Simulated scans")

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

    data, telescope_data = create_observations(args, comm, schedules)

    # Split the communicator for day and season mapmaking

    time_comms = get_time_communicators(comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data)

    # Purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if (args.tidas is None) and (args.spt3g is None):
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_radec_quats()

    # Prepare auxiliary information for distributed map objects

    _, localsm, subnpix = get_submaps(args, comm, data)

    if args.input_pysm_model:
        signalname = simulate_sky_signal(
            args, comm, data, schedules, subnpix, localsm, "signal"
        )
    else:
        signalname = scan_sky_signal(args, comm, data, localsm, subnpix, "signal")

    # Set up objects to take copies of the TOD at appropriate times

    totalname, totalname_freq = setup_sigcopy(args)

    # Loop over Monte Carlos

    firstmc = int(args.MC_start)
    nmc = int(args.MC_count)

    freqs = [float(freq) for freq in args.freq.split(",")]
    nfreq = len(freqs)

    for mc in range(firstmc, firstmc + nmc):

        simulate_atmosphere(args, comm, data, mc, totalname)

        # Loop over frequencies with identical focal planes and identical
        # atmospheric noise.

        for ifreq, freq in enumerate(freqs):

            if rank == 0:
                log.info(
                    "Processing frequency {}GHz {} / {}, MC = {}".format(
                        freq, ifreq + 1, nfreq, mc
                    )
                )

            # Make a copy of the atmosphere so we can scramble the gains and apply
            # frequency-dependent scaling.
            copy_signal(args, comm, data, totalname, totalname_freq)

            scale_atmosphere_by_frequency(args, comm, data, freq, mc, totalname_freq)

            update_atmospheric_noise_weights(args, comm, data, freq, mc)

            # Add previously simulated sky signal to the atmospheric noise.

            add_signal(data, totalname_freq, signalname)

            mcoffset = ifreq * 1000000

            simulate_noise(args, comm, data, mc + mcoffset, totalname_freq)

            simulate_sss(args, comm, data, mc + mcoffset, totalname_freq)

            scramble_gains(args, comm, data, mc + mcoffset, totalname_freq)

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
                data,
                madampars,
                mc + mcoffset,
                outpath,
                detweights,
                totalname_freq,
                freq=freq,
                time_comms=time_comms,
                telescope_data=telescope_data,
                first_call=(mc == firstmc),
            )

            if args.apply_polyfilter or args.apply_groundfilter:

                # Filter signal

                apply_polyfilter(args, comm, data, totalname_freq)

                apply_groundfilter(args, comm, data, totalname_freq)

                # Bin filtered maps

                apply_madam(
                    args,
                    comm,
                    data,
                    madampars,
                    mc + mcoffset,
                    outpath,
                    detweights,
                    totalname_freq,
                    freq=freq,
                    time_comms=time_comms,
                    telescope_data=telescope_data,
                    first_call=False,
                    extra_prefix="filtered",
                    bin_only=True,
                )

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()
    timer = Timer()
    timer.start()
    alltimers = gather_timers(comm=mpiworld)
    if rank == 0:
        out = os.path.join(args.outdir, "timing")
        dump_timing(alltimers, out)
        timer.stop()
        timer.report("Gather and dump timing info")
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
