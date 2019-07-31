#!/usr/bin/env python3

# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
Simpler version of the ground simulation script
"""

import argparse
import dateutil.parser
import os
import pickle
import sys
import traceback

import numpy as np

from toast.mpi import get_world, Comm

from toast.dist import distribute_uniform, Data

from toast.utils import Logger, Environment, memreport

from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing

import toast.qarray as qa
from toast.tod import TODGround, OpCacheCopy, plot_focalplane, OpCacheClear

from toast.pipeline_tools import (
    add_dist_args,
    get_time_communicators,
    get_comm,
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
    add_sky_map_args,
    add_pysm_args,
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
    get_breaks,
    Telescope,
    Site,
    CES,
    load_schedule,
    load_weather,
    add_mc_args,
    add_binner_args,
    init_binner,
    apply_binner,
)


def parse_arguments(comm):
    timer = Timer()
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "and map astrophysical signal.",
        fromfile_prefix_chars="@",
    )

    add_dist_args(parser)
    add_todground_args(parser)
    add_pointing_args(parser)
    add_polyfilter_args(parser)
    add_groundfilter_args(parser)
    add_gainscrambler_args(parser)
    add_noise_args(parser)
    add_sky_map_args(parser)
    add_tidas_args(parser)

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )

    add_madam_args(parser, ground_data=True)
    add_binner_args(parser)

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
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Write diagnostics",
    )
    parser.add_argument(
        "--focalplane",
        required=False,
        default=None,
        help="Pickle file containing a dictionary of detector "
        "properties.  The keys of this dict are the detector "
        "names, and each value is also a dictionary with keys "
        '"quat" (4 element ndarray), "fwhm" (float, arcmin), '
        '"fknee" (float, Hz), "alpha" (float), and '
        '"NET" (float).  For optional plotting, the key "color"'
        " can specify a valid matplotlib color string.",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(0)

    if args.tidas is not None:
        if not tidas_available:
            raise RuntimeError("TIDAS not found- cannot export")

    if comm.comm_world is None or comm.world_rank == 0:
        log.info("All parameters:")
        for ag in vars(args):
            log.info("{} = {}".format(ag, getattr(args, ag)))

    if args.group_size:
        comm = toast.Comm(groupsize=args.group_size)

    if comm.comm_world is None or comm.comm_world.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)

    timer.stop()
    if comm.comm_world is None or comm.world_rank == 0:
        timer.report("Parsed parameters")

    return args, comm


def load_fp(args, comm):
    timer = Timer()
    log = Logger.get()
    timer.start()

    fp = None

    # Load focalplane information

    nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

    if comm.comm_world is None or comm.comm_world.rank == 0:
        if args.focalplane is None:
            XAXIS, YAXIS, ZAXIS = np.eye(3)
            # in this case, create a fake detector at the boresight
            # with a pure white noise spectrum.
            fake = {}
            fake["quat"] = nullquat
            fake["fwhm"] = 30.0
            fake["fknee"] = 0.0
            fake["fmin"] = 1e-9
            fake["alpha"] = 1.0
            fake["NET"] = 1.0
            fake["color"] = "r"
            fp = {}
            # Second detector at 22.5 degree polarization angle
            fp["bore1"] = fake
            fake2 = {}
            zrot = qa.rotation(ZAXIS, np.radians(22.5))
            fake2["quat"] = qa.mult(fake["quat"], zrot)
            fake2["fwhm"] = 30.0
            fake2["fknee"] = 0.0
            fake2["fmin"] = 1e-9
            fake2["alpha"] = 1.0
            fake2["NET"] = 1.0
            fake2["color"] = "r"
            fp["bore2"] = fake2
            # Third detector at 45 degree polarization angle
            fake3 = {}
            zrot = qa.rotation(ZAXIS, np.radians(45))
            fake3["quat"] = qa.mult(fake["quat"], zrot)
            fake3["fwhm"] = 30.0
            fake3["fknee"] = 0.0
            fake3["fmin"] = 1e-9
            fake3["alpha"] = 1.0
            fake3["NET"] = 1.0
            fake3["color"] = "r"
            fp["bore3"] = fake3
            # Fourth detector at 67.5 degree polarization angle
            fake4 = {}
            zrot = qa.rotation(ZAXIS, np.radians(67.5))
            fake4["quat"] = qa.mult(fake["quat"], zrot)
            fake4["fwhm"] = 30.0
            fake4["fknee"] = 0.0
            fake4["fmin"] = 1e-9
            fake4["alpha"] = 1.0
            fake4["NET"] = 1.0
            fake4["color"] = "r"
            fp["bore4"] = fake4
        else:
            with open(args.focalplane, "rb") as p:
                fp = pickle.load(p)
    if comm.comm_world is not None:
        fp = comm.comm_world.bcast(fp, root=0)

    if comm.comm_world is None or comm.comm_world.rank == 0:
        timer.report_clear("Create focalplane")

    if args.debug:
        if comm.comm_world is None or comm.comm_world.rank == 0:
            outfile = "{}/focalplane.png".format(args.outdir)
            plot_focalplane(fp, 6, 6, outfile)

    detectors = sorted(fp.keys())
    detweights = {}
    for d in detectors:
        net = fp[d]["NET"]
        detweights[d] = 1.0 / (args.sample_rate * net * net)

    return fp, detweights


def create_observations(args, comm, fp, all_ces, site):
    """ Simulate constant elevation scans.

    Simulate constant elevation scans at "site" matching entries in
    "all_ces".  Each operational day is assigned to a different
    process group to allow making day maps.

    """
    timer = Timer()
    log = Logger.get()

    data = Data(comm)

    detectors = sorted(fp.keys())
    detquats = {}
    for d in detectors:
        detquats[d] = fp[d]["quat"]

    nces = len(all_ces)

    breaks = get_breaks(comm, all_ces, args)

    nbreak = len(breaks)
    if nbreak != comm.ngroups - 1:
        raise RuntimeError(
            "Number of observing days ({}) does not match number of process "
            "groups ({}).".format(nbreak + 1, comm.ngroups)
        )

    groupdist = distribute_uniform(nces, comm.ngroups, breaks=breaks)
    group_firstobs = groupdist[comm.group][0]
    group_numobs = groupdist[comm.group][1]

    # Create the noise model used by all observations

    noise = get_analytic_noise(args, comm, fp)

    if comm.comm_group is not None:
        ndetrank = comm.comm_group.size
    else:
        ndetrank = 1

    for ices in range(group_firstobs, group_firstobs + group_numobs):
        ces = all_ces[ices]
        totsamples = int((ces.stop_time - ces.start_time) * args.sample_rate)

        # create the single TOD for this observation

        try:
            tod = TODGround(
                comm.comm_group,
                detquats,
                totsamples,
                detranks=ndetrank,
                firsttime=ces.start_time,
                rate=args.sample_rate,
                site_lon=site.lon,
                site_lat=site.lat,
                site_alt=site.alt,
                azmin=ces.azmin,
                azmax=ces.azmax,
                el=ces.el,
                scanrate=args.scan_rate,
                scan_accel=args.scan_accel,
                CES_start=None,
                CES_stop=None,
                sun_angle_min=args.sun_angle_min,
                coord=args.coord,
                sampsizes=None,
                report_timing=args.debug,
            )
        except RuntimeError as e:
            raise RuntimeError("Failed to create the CES scan: {}".format(e))

        # Create the (single) observation

        ob = {}
        ob["name"] = "CES-{}-{}-{}".format(ces.name, ces.scan, ces.subscan)
        ob["tod"] = tod
        if len(tod.subscans) > 0:
            ob["intervals"] = tod.subscans
        else:
            raise RuntimeError("{} has no valid intervals".format(ob["name"]))
        ob["baselines"] = None
        ob["noise"] = noise
        ob["id"] = int(ces.mjdstart * 10000)

        data.obs.append(ob)

    for ob in data.obs:
        tod = ob["tod"]
        tod.free_azel_quats()

    if comm.comm_world is None or comm.comm_group.rank == 0:
        log.info("Group # {:4} has {} observations.".format(comm.group, len(data.obs)))

    if len(data.obs) == 0:
        raise RuntimeError(
            "Too many tasks. Every MPI task must "
            "be assigned to at least one observation."
        )

    if comm.comm_world is None or comm.comm_world.rank == 0:
        timer.report_clear("Simulate scans")

    return data


def setup_sigcopy(args, comm, signalname):
    """ Setup for copying the signal so we can run filter+bin and Madam.

    """
    if args.use_madam:
        signalname_madam = "signal_madam"
        sigcopy_madam = OpCacheCopy(signalname, signalname_madam)
        sigclear = OpCacheClear(signalname)
    else:
        signalname_madam = signalname
        sigcopy_madam = None
        sigclear = None

    return signalname_madam, sigcopy_madam, sigclear


def setup_output(args, comm):
    outpath = "{}".format(args.outdir)
    if comm.comm_world is None or comm.comm_world.rank == 0:
        if not os.path.isdir(outpath):
            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass
    return outpath


def copy_signal_madam(args, comm, data, sigcopy_madam):
    """ Make a copy of the TOD for Madam.

    """
    if sigcopy_madam is not None:
        if comm.comm_world is None or comm.comm_world.rank == 0:
            print("Making a copy of the TOD for Madam", flush=args.flush)
        sigcopy_madam.exec(data)

    return


def clear_signal(args, comm, data, sigclear):
    if sigclear is not None:
        if comm.comm_world is None or comm.comm_world.rank == 0:
            print("Clearing filtered signal")
        sigclear.exec(data)
    return


def main():
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_ground_sim (total)")

    mpiworld, procs, rank, comm = get_comm()

    args, comm = parse_arguments(comm)

    # Initialize madam parameters

    madampars = setup_madam(args)

    # Load and broadcast the schedule file

    site, all_ces = load_schedule(args, comm)[0]

    # load or simulate the focalplane

    fp, detweights = load_fp(args, comm)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    data = create_observations(args, comm, fp, all_ces, site)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data)

    # Prepare auxiliary information for distributed map objects

    localpix, localsm, subnpix = get_submaps(args, comm, data)

    # Scan input map

    signalname = scan_sky_signal(args, comm, data, localsm, subnpix, "signal")

    # Simulate noise

    if signalname is None:
        signalname = "signal"
        mc = 0
        simulate_noise(args, comm, data, mc, signalname)

    # Set up objects to take copies of the TOD at appropriate times

    signalname_madam, sigcopy_madam, sigclear = setup_sigcopy(args, comm, signalname)

    invnpp, zmap = init_binner(
        args, comm, data, detweights, subnpix=subnpix, localsm=localsm
    )

    output_tidas(args, comm, data, signalname)

    outpath = setup_output(args, comm)

    # Make a copy of the signal for Madam

    copy_signal_madam(args, comm, data, sigcopy_madam)

    # Bin unprocessed signal for reference

    apply_binner(args, comm, data, invnpp, zmap, detweights, outpath, signalname)

    if args.apply_polyfilter or args.apply_groundfilter:

        # Filter signal

        apply_polyfilter(args, comm, data, signalname)

        apply_groundfilter(args, comm, data, signalname)

        # Bin the filtered signal

        apply_binner(
            args,
            comm,
            data,
            invnpp,
            zmap,
            detweights,
            outpath,
            signalname,
            prefix="filtered",
        )

    data.obs[0]["tod"].cache.report()

    clear_signal(args, comm, data, sigclear)

    data.obs[0]["tod"].cache.report()

    # Now run Madam on the unprocessed copy of the signal

    if args.use_madam:
        apply_madam(args, comm, data, madampars, outpath, detweights, signalname_madam)

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()
    timer = Timer()
    timer.start()
    alltimers = gather_timers(comm=mpiworld)
    if comm.comm_world is None or comm.comm_world.rank == 0:
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
