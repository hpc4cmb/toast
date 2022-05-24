#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a ground simulation and makes a map.
"""


import os

if "TOAST_STARTUP_DELAY" in os.environ:
    import time

    import numpy as np

    delay = np.float(os.environ["TOAST_STARTUP_DELAY"])
    wait = np.random.rand() * delay
    # print('Sleeping for {} seconds before importing TOAST'.format(wait),
    #      flush=True)
    time.sleep(wait)

import argparse
import copy
import pickle
import sys
import traceback

import numpy as np

from toast import pipeline_tools
from toast.dist import Data, distribute_uniform
from toast.mpi import Comm, get_world
from toast.timing import GlobalTimers, Timer
from toast.timing import dump as dump_timing
from toast.timing import function_timer, gather_timers
from toast.todmap import TODGround
from toast.utils import Environment, Logger, memreport

# import warnings
# warnings.filterwarnings('error')
# warnings.simplefilter('ignore', ImportWarning)
# warnings.simplefilter('ignore', ResourceWarning)
# warnings.simplefilter('ignore', DeprecationWarning)
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def parse_arguments(comm):
    timer = Timer()
    timer.start()
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "atmosphere and make maps for some number of noise Monte Carlos.",
        fromfile_prefix_chars="@",
    )

    pipeline_tools.add_dist_args(parser)
    pipeline_tools.add_debug_args(parser)
    pipeline_tools.add_todground_args(parser)
    pipeline_tools.add_pointing_args(parser)
    pipeline_tools.add_polyfilter_args(parser)
    pipeline_tools.add_polyfilter2D_args(parser)
    pipeline_tools.add_groundfilter_args(parser)
    pipeline_tools.add_atmosphere_args(parser)
    pipeline_tools.add_noise_args(parser)
    pipeline_tools.add_gainscrambler_args(parser)
    pipeline_tools.add_madam_args(parser)
    pipeline_tools.add_mapmaker_args(parser)
    pipeline_tools.add_sky_map_args(parser)
    pipeline_tools.add_pysm_args(parser)
    pipeline_tools.add_sss_args(parser)
    pipeline_tools.add_tidas_args(parser)
    pipeline_tools.add_spt3g_args(parser)
    pipeline_tools.add_mc_args(parser)

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )

    parser.add_argument(
        "--madam",
        required=False,
        action="store_true",
        help="Use libmadam for map-making",
        dest="use_madam",
    )
    parser.add_argument(
        "--no-madam",
        required=False,
        action="store_false",
        help="Do not use libmadam for map-making [default]",
        dest="use_madam",
    )
    parser.set_defaults(use_madam=False)

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
        help="Comma-separated list of frequencies with identical focal planes."
        "  They override the bandpasses in the focalplane for the purpose of"
        " scaling the atmospheric signal but not for simulating the sky signal.",
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

    if args.group_size:
        comm = Comm(groupsize=args.group_size)

    if comm.world_rank == 0:
        os.makedirs(args.outdir, exist_ok=True)

    timer.stop()
    if comm.world_rank == 0:
        timer.report("Parsed parameters")

    return args, comm


@function_timer
def load_focalplanes(args, comm, schedules):
    """Attach a focalplane to each of the schedules.

    Args:
        schedules (list) :  List of Schedule instances.
            Each schedule has two members, telescope
            and ceslist, a list of CES objects.
    Returns:
        detweights (dict) : Inverse variance noise weights for every
            detector across all focal planes. In [K_CMB^-2].
            They can be used to bin the TOD.
    """
    timer = Timer()
    timer.start()

    # Load focalplane information

    focalplanes = []
    if comm.world_rank == 0:
        for fpfile in args.focalplane.split(","):
            focalplanes.append(
                pipeline_tools.Focalplane(
                    fname_pickle=fpfile,
                    sample_rate=args.sample_rate,
                    radius_deg=args.focalplane_radius_deg,
                )
            )
    if comm.comm_world is not None:
        focalplanes = comm.comm_world.bcast(focalplanes)

    if len(focalplanes) == 1 and len(schedules) > 1:
        focalplanes *= len(schedules)
    if len(focalplanes) != len(schedules):
        raise RuntimeError(
            "Number of focalplanes must equal number of schedules or be 1."
        )

    # Append a focal plane and telescope to each entry in the schedules
    # list and assemble a detector weight dictionary that represents all
    # detectors in all focalplanes
    detweights = {}
    for schedule, focalplane in zip(schedules, focalplanes):
        schedule.telescope.focalplane = focalplane
        detweights.update(schedule.telescope.focalplane.detweights)

    timer.stop()
    if comm.world_rank == 0:
        timer.report("Loading focalplanes")
    return detweights


@function_timer
def create_observation(args, comm, telescope, ces, verbose=True):
    """Create a TOAST observation.

    Create an observation for the CES scan

    Args:
        args :  argparse arguments
        comm :  TOAST communicator
        ces (CES) :  One constant elevation scan

    """
    focalplane = telescope.focalplane
    site = telescope.site
    weather = site.weather
    noise = focalplane.noise
    totsamples = int((ces.stop_time - ces.start_time) * args.sample_rate)

    # create the TOD for this observation

    if comm.comm_group is not None:
        ndetrank = comm.comm_group.size
    else:
        ndetrank = 1

    if args.el_nod_deg and (ces.subscan == 0 or args.el_nod_every_scan):
        el_nod = args.el_nod_deg
    else:
        el_nod = None

    try:
        tod = TODGround(
            comm.comm_group,
            focalplane.detquats,
            totsamples,
            detranks=ndetrank,
            boresight_angle=ces.boresight_angle,
            firsttime=ces.start_time,
            rate=args.sample_rate,
            site_lon=site.lon,
            site_lat=site.lat,
            site_alt=site.alt,
            azmin=ces.azmin,
            azmax=ces.azmax,
            el=ces.el,
            el_nod=el_nod,
            start_with_elnod=args.start_with_el_nod,
            end_with_elnod=args.end_with_el_nod,
            el_mod_step=args.el_mod_step_deg,
            el_mod_rate=args.el_mod_rate_hz,
            el_mod_amplitude=args.el_mod_amplitude_deg,
            el_mod_sine=args.el_mod_sine,
            scanrate=args.scan_rate,
            scanrate_el=args.scan_rate_el,
            scan_accel=args.scan_accel,
            scan_accel_el=args.scan_accel_el,
            cosecant_modulation=args.scan_cosecant_modulate,
            CES_start=None,
            CES_stop=None,
            sun_angle_min=args.sun_angle_min,
            coord=args.coord,
            sampsizes=None,
            report_timing=args.debug,
            hwprpm=args.hwp_rpm,
            hwpstep=args.hwp_step_deg,
            hwpsteptime=args.hwp_step_time_s,
        )
    except RuntimeError as e:
        raise RuntimeError(
            'Failed to create TOD for {}-{}-{}: "{}"'
            "".format(ces.name, ces.scan, ces.subscan, e)
        )

    # Create the observation

    obs = {}
    obs["name"] = "CES-{}-{}-{}-{}-{}".format(
        site.name, telescope.name, ces.name, ces.scan, ces.subscan
    )
    obs["tod"] = tod
    obs["baselines"] = None
    obs["noise"] = copy.deepcopy(noise)
    obs["id"] = int(ces.mjdstart * 10000)
    obs["intervals"] = tod.subscans
    obs["site"] = site
    obs["site_name"] = site.name
    obs["site_id"] = site.id
    obs["altitude"] = site.alt
    obs["weather"] = site.weather
    obs["telescope"] = telescope
    obs["telescope_name"] = telescope.name
    obs["telescope_id"] = telescope.id
    obs["focalplane"] = telescope.focalplane.detector_data
    obs["fpradius"] = telescope.focalplane.radius
    obs["start_time"] = ces.start_time
    obs["season"] = ces.season
    obs["date"] = ces.start_date
    obs["MJD"] = ces.mjdstart
    obs["rising"] = ces.rising
    obs["mindist_sun"] = ces.mindist_sun
    obs["mindist_moon"] = ces.mindist_moon
    obs["el_sun"] = ces.el_sun
    return obs


@function_timer
def create_observations(args, comm, schedules):
    """Create and distribute TOAST observations for every CES in
    schedules.

    Args:
        schedules (iterable) :  a list of Schedule objects.
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

        telescope = schedule.telescope
        all_ces = schedule.ceslist
        nces = len(all_ces)

        breaks = pipeline_tools.get_breaks(comm, all_ces, args)

        groupdist = distribute_uniform(nces, comm.ngroups, breaks=breaks)
        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for ices in range(group_firstobs, group_firstobs + group_numobs):
            obs = create_observation(args, comm, telescope, all_ces[ices])
            data.obs.append(obs)

    if comm.comm_world is None or comm.comm_group.rank == 0:
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
        telescope_data = data.split("telescope_name")
        if len(telescope_data) == 1:
            # Only one telescope available
            telescope_data = []
    else:
        telescope_data = []
    telescope_data.insert(0, ("all", data))
    return data, telescope_data


def setup_sigcopy(args):
    """Determine if an extra copy of the atmospheric signal is needed.

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
        os.makedirs(outpath, exist_ok=True)
    return outpath


def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_ground_sim (total)")
    timer0 = Timer()
    timer0.start()

    mpiworld, procs, rank, comm = pipeline_tools.get_comm()

    args, comm = parse_arguments(comm)

    if args.use_madam:
        # Initialize madam parameters
        madampars = pipeline_tools.setup_madam(args)

    # Load and broadcast the schedule file

    schedules = pipeline_tools.load_schedule(args, comm)

    # Load the weather and append to schedules

    pipeline_tools.load_weather(args, comm, schedules)

    # load or simulate the focalplane

    detweights = load_focalplanes(args, comm, schedules)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    data, telescope_data = create_observations(args, comm, schedules)

    # Split the communicator for day and season mapmaking

    time_comms = pipeline_tools.get_time_communicators(args, comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    pipeline_tools.expand_pointing(args, comm, data)

    # Optionally rewrite the noise PSD:s in each observation to include
    # elevation-dependence

    pipeline_tools.get_elevation_noise(args, comm, data)

    # Purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if (args.tidas is None) and (args.spt3g is None):
        for ob in data.obs:
            tod = ob["tod"]
            tod.free_radec_quats()

    # Prepare auxiliary information for distributed map objects

    if args.pysm_model:
        focalplanes = [s.telescope.focalplane.detector_data for s in schedules]
        signalname = pipeline_tools.simulate_sky_signal(
            args, comm, data, focalplanes, "signal"
        )
    else:
        signalname = pipeline_tools.scan_sky_signal(args, comm, data, "signal")

    # Set up objects to take copies of the TOD at appropriate times

    totalname, totalname_freq = setup_sigcopy(args)

    # Loop over Monte Carlos

    firstmc = args.MC_start
    nsimu = args.MC_count

    freqs = [float(freq) for freq in args.freq.split(",")]
    nfreq = len(freqs)

    for mc in range(firstmc, firstmc + nsimu):

        pipeline_tools.simulate_atmosphere(args, comm, data, mc, totalname)

        # Loop over frequencies with identical focal planes and identical
        # atmospheric noise.

        for ifreq, freq in enumerate(freqs):

            if comm.world_rank == 0:
                log.info(
                    "Processing frequency {}GHz {} / {}, MC = {}".format(
                        freq, ifreq + 1, nfreq, mc
                    )
                )

            # Make a copy of the atmosphere so we can scramble the gains and apply
            # frequency-dependent scaling.
            pipeline_tools.copy_signal(args, comm, data, totalname, totalname_freq)

            pipeline_tools.scale_atmosphere_by_frequency(
                args, comm, data, freq=freq, mc=mc, cache_name=totalname_freq
            )

            pipeline_tools.update_atmospheric_noise_weights(args, comm, data, freq, mc)

            # Add previously simulated sky signal to the atmospheric noise.

            pipeline_tools.add_signal(
                args, comm, data, totalname_freq, signalname, purge=(nsimu == 1)
            )

            mcoffset = ifreq * 1000000

            pipeline_tools.simulate_noise(
                args, comm, data, mc + mcoffset, totalname_freq
            )

            pipeline_tools.simulate_sss(args, comm, data, mc + mcoffset, totalname_freq)

            pipeline_tools.scramble_gains(
                args, comm, data, mc + mcoffset, totalname_freq
            )

            if (mc == firstmc) and (ifreq == 0):
                # For the first realization and frequency, optionally
                # export the timestream data.
                pipeline_tools.output_tidas(args, comm, data, totalname)
                pipeline_tools.output_spt3g(args, comm, data, totalname)

            outpath = setup_output(args, comm, mc + mcoffset, freq)

            # Bin and destripe maps

            if args.use_madam:
                pipeline_tools.apply_madam(
                    args,
                    comm,
                    data,
                    madampars,
                    outpath,
                    detweights,
                    totalname_freq,
                    freq=freq,
                    time_comms=time_comms,
                    telescope_data=telescope_data,
                    first_call=(mc == firstmc),
                )
            else:
                pipeline_tools.apply_mapmaker(
                    args,
                    comm,
                    data,
                    outpath,
                    totalname_freq,
                    time_comms=time_comms,
                    telescope_data=telescope_data,
                    first_call=(mc == firstmc),
                )

            if (
                args.apply_polyfilter
                or args.apply_polyfilter2D
                or args.apply_groundfilter
            ):

                # Filter signal

                pipeline_tools.apply_polyfilter(args, comm, data, totalname_freq)

                pipeline_tools.apply_polyfilter2D(args, comm, data, totalname_freq)

                pipeline_tools.apply_groundfilter(args, comm, data, totalname_freq)

                # Bin filtered maps

                if args.use_madam:
                    pipeline_tools.apply_madam(
                        args,
                        comm,
                        data,
                        madampars,
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
                else:
                    pipeline_tools.apply_mapmaker(
                        args,
                        comm,
                        data,
                        outpath,
                        totalname_freq,
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
    if comm.world_rank == 0:
        out = os.path.join(args.outdir, "timing")
        dump_timing(alltimers, out)
        timer.stop()
        timer.report("Gather and dump timing info")
        timer0.report_clear("toast_ground_sim.py")
    return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        if procs == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None:
            mpiworld.Abort(6)
