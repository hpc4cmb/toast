#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a satellite simulation and makes a map.
"""

import os
import sys
import re
import argparse
import traceback

import pickle

import numpy as np

from astropy.io import fits

from toast.mpi import get_world, Comm

from toast.dist import distribute_uniform, Data

from toast.utils import Logger, Environment

from toast.vis import set_backend

from toast.timing import function_timer, GlobalTimers, Timer, gather_timers
from toast.timing import dump as dump_timing

from toast.tod import regular_intervals, plot_focalplane, OpApplyGain
from toast.todmap import TODSatellite, slew_precession_axis

from toast import pipeline_tools


def parse_arguments(comm, procs):
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Simulate satellite boresight pointing and make a map.",
        fromfile_prefix_chars="@",
    )

    pipeline_tools.add_dist_args(parser)
    pipeline_tools.add_pointing_args(parser)
    pipeline_tools.add_tidas_args(parser)
    pipeline_tools.add_spt3g_args(parser)
    pipeline_tools.add_dipole_args(parser)
    pipeline_tools.add_sky_map_args(parser)
    pipeline_tools.add_pysm_args(parser)
    pipeline_tools.add_mc_args(parser)
    pipeline_tools.add_noise_args(parser)
    pipeline_tools.add_todsatellite_args(parser)
    pipeline_tools.add_conviqt_args(parser)

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )
    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Write diagnostics",
    )

    pipeline_tools.add_madam_args(parser)
    pipeline_tools.add_mapmaker_args(parser)
    pipeline_tools.add_binner_args(parser)

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
        help="Pickle file containing a dictionary of detector properties.  "
        "The keys of this dict are the detector names, and each value is also "
        'a dictionary with keys "quat" (4 element ndarray), "fwhm" '
        '(float, arcmin), "fknee" (float, Hz), "alpha" (float), and "NET" '
        '(float).  For optional plotting, the key "color" can specify a '
        "valid matplotlib color string.",
    )

    parser.add_argument(
        "--gain",
        required=False,
        default=None,
        help="Calibrate the input timelines with a set of gains from a"
        "FITS file containing 3 extensions:"
        "HDU named DETECTORS : table with list of detector names in a column named DETECTORS"
        "HDU named TIME: table with common timestamps column named TIME"
        "HDU named GAINS: 2D image of floats with one row per detector and one column per value.",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit()

    if comm.world_rank == 0:
        log.info("\n")
        log.info("All parameters:")
        for ag in vars(args):
            log.info("{} = {}".format(ag, getattr(args, ag)))
        log.info("\n")

    groupsize = args.group_size
    if groupsize is None or groupsize <= 0:
        groupsize = procs

    # This is the 2-level toast communicator.
    comm = Comm(groupsize=groupsize)

    return args, comm, groupsize


def load_focalplane(args, comm):
    """Load focalplane information"""
    timer = Timer()
    gain = None
    fp = None
    if comm.world_rank == 0:
        if args.focalplane is None:
            # in this case, create a fake detector at the boresight
            # with a pure white noise spectrum.
            fake = {}
            fake["quat"] = np.array([0.0, 0.0, 1.0, 0.0])
            fake["fwhm"] = 30.0
            fake["fknee"] = 0.0
            fake["fmin"] = 1.0e-5
            fake["alpha"] = 1.0
            fake["NET"] = 1.0
            fake["polangle_deg"] = 0
            fake["color"] = "r"
            fp = {}
            fp["bore"] = fake
        else:
            with open(args.focalplane, "rb") as p:
                fp = pickle.load(p)

        if args.gain is not None:
            gain = {}
            with fits.open(args.gain) as f:
                gain["TIME"] = np.array(f["TIME"].data["TIME"])
                for i_det, det_name in f["DETECTORS"].data["DETECTORS"]:
                    gain[det_name] = np.array(f["GAINS"].data[i_det, :])

    if comm.comm_world is not None:
        if args.gain is not None:
            gain = comm.comm_world.bcast(gain, root=0)
        fp = comm.comm_world.bcast(fp, root=0)

    timer.stop()
    if comm.world_rank == 0:
        timer.report("Create focalplane ({} dets)".format(len(fp.keys())))

    if args.debug:
        if comm.world_rank == 0:
            outfile = os.path.join(args.outdir, "focalplane.png")
            set_backend()
            dquats = {x: fp[x]["quat"] for x in fp.keys()}
            dfwhm = {x: fp[x]["fwhm"] for x in fp.keys()}
            plot_focalplane(dquats, 10.0, 10.0, outfile, fwhm=dfwhm)

    # For purposes of this simulation, we use detector noise
    # weights based on the NET (white noise level).  If the destriping
    # baseline is too long, this will not be the best choice.

    detweights = {}
    for d in fp.keys():
        net = fp[d]["NET"]
        detweights[d] = 1.0 / (args.sample_rate * net * net)

    return fp, gain, detweights


def create_observations(args, comm, focalplane, groupsize):
    timer = Timer()
    timer.start()
    log = Logger.get()

    if groupsize > len(focalplane.keys()):
        if comm.world_rank == 0:
            log.error("process group is too large for the number of detectors")
            comm.comm_world.Abort()

    # Detector information from the focalplane

    detectors = sorted(focalplane.keys())
    detquats = {}
    detindx = None
    if "index" in focalplane[detectors[0]]:
        detindx = {}

    for d in detectors:
        detquats[d] = focalplane[d]["quat"]
        if detindx is not None:
            detindx[d] = focalplane[d]["index"]

    # Distribute the observations uniformly

    groupdist = distribute_uniform(args.obs_num, comm.ngroups)

    # Compute global time and sample ranges of all observations

    obsrange = regular_intervals(
        args.obs_num,
        args.start_time,
        0,
        args.sample_rate,
        3600 * args.obs_time_h,
        3600 * args.gap_h,
    )

    noise = pipeline_tools.get_analytic_noise(args, comm, focalplane)

    # The distributed timestream data

    data = Data(comm)

    # Every process group creates its observations

    group_firstobs = groupdist[comm.group][0]
    group_numobs = groupdist[comm.group][1]

    for ob in range(group_firstobs, group_firstobs + group_numobs):
        tod = TODSatellite(
            comm.comm_group,
            detquats,
            obsrange[ob].samples,
            coord=args.coord,
            firstsamp=obsrange[ob].first,
            firsttime=obsrange[ob].start,
            rate=args.sample_rate,
            spinperiod=args.spin_period_min,
            spinangle=args.spin_angle_deg,
            precperiod=args.prec_period_min,
            precangle=args.prec_angle_deg,
            detindx=detindx,
            detranks=comm.group_size,
            hwprpm=args.hwp_rpm,
            hwpstep=args.hwp_step_deg,
            hwpsteptime=args.hwp_step_time_s,
        )

        obs = {}
        obs["name"] = "science_{:05d}".format(ob)
        obs["tod"] = tod
        obs["intervals"] = None
        obs["baselines"] = None
        obs["noise"] = noise
        obs["id"] = ob
        obs["focalplane"] = pipeline_tools.Focalplane(focalplane)

        data.obs.append(obs)

    if comm.world_rank == 0:
        timer.report_clear("Read parameters, compute data distribution")

    # Since we are simulating noise timestreams, we want
    # them to be contiguous and reproducible over the whole
    # observation.  We distribute data by detector within an
    # observation, so ensure that our group size is not larger
    # than the number of detectors we have.

    # we set the precession axis now, which will trigger calculation
    # of the boresight pointing.

    for ob in range(group_numobs):
        curobs = data.obs[ob]
        tod = curobs["tod"]

        # Get the global sample offset from the original distribution of
        # intervals
        obsoffset = obsrange[group_firstobs + ob].first

        # Constantly slewing precession axis
        degday = 360.0 / 365.25
        precquat = np.empty(4 * tod.local_samples[1], dtype=np.float64).reshape((-1, 4))
        slew_precession_axis(
            precquat,
            firstsamp=(obsoffset + tod.local_samples[0]),
            samplerate=args.sample_rate,
            degday=degday,
        )

        tod.set_prec_axis(qprec=precquat)
        del precquat

    if comm.world_rank == 0:
        timer.report_clear("Construct boresight pointing")

    return data


def main():
    env = Environment.get()
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_satellite_sim (total)")
    timer0 = Timer()
    timer0.start()

    mpiworld, procs, rank, comm = pipeline_tools.get_comm()
    args, comm, groupsize = parse_arguments(comm, procs)

    # Parse options

    tmr = Timer()
    tmr.start()

    if comm.world_rank == 0:
        os.makedirs(args.outdir, exist_ok=True)

    focalplane, gain, detweights = load_focalplane(args, comm)
    if comm.world_rank == 0:
        tmr.report_clear("Load focalplane")

    data = create_observations(args, comm, focalplane, groupsize)
    if comm.world_rank == 0:
        tmr.report_clear("Create observations")

    pipeline_tools.expand_pointing(args, comm, data)
    if comm.world_rank == 0:
        tmr.report_clear("Expand pointing")

    signalname = None
    if args.pysm_model:
        skyname = pipeline_tools.simulate_sky_signal(
            args, comm, data, [focalplane], "signal"
        )
    else:
        skyname = pipeline_tools.scan_sky_signal(args, comm, data, "signal")
    if skyname is not None:
        signalname = skyname
    if comm.world_rank == 0:
        tmr.report_clear("Simulate sky signal")

    # NOTE: Conviqt could use different input file names for different
    # Monte Carlo indices, but the operator would need to be invoked within
    # the Monte Carlo loop.
    skyname = pipeline_tools.apply_conviqt(
        args, comm, data, "signal", mc=args.MC_start,
    )
    if skyname is not None:
        signalname = skyname
    if comm.world_rank == 0:
        tmr.report_clear("Apply beam convolution")

    diponame = pipeline_tools.simulate_dipole(args, comm, data, "signal")
    if diponame is not None:
        signalname = diponame
    if comm.world_rank == 0:
        tmr.report_clear("Simulate dipole")

    # in debug mode, print out data distribution information
    if args.debug:
        handle = None
        if comm.world_rank == 0:
            handle = open(os.path.join(args.outdir, "distdata.txt"), "w")
        data.info(handle)
        if comm.world_rank == 0:
            handle.close()
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        if comm.world_rank == 0:
            tmr.report_clear("Dumping data distribution")

    # in debug mode, print out data distribution information
    if args.debug:
        handle = None
        if comm.world_rank == 0:
            handle = open(os.path.join(args.outdir, "distdata.txt"), "w")
        data.info(handle)
        if comm.world_rank == 0:
            handle.close()
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        if comm.world_rank == 0:
            tmr.report_clear("Dumping data distribution")

    # Mapmaking.

    if args.use_madam:
        # Initialize madam parameters
        madampars = pipeline_tools.setup_madam(args)
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        if comm.world_rank == 0:
            tmr.report_clear("Initialize madam map-making")

    # Loop over Monte Carlos

    firstmc = args.MC_start
    nmc = args.MC_count

    for mc in range(firstmc, firstmc + nmc):
        mctmr = Timer()
        mctmr.start()

        # create output directory for this realization
        outpath = os.path.join(args.outdir, "mc_{:03d}".format(mc))

        pipeline_tools.simulate_noise(
            args, comm, data, mc, "tot_signal", overwrite=True
        )
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        if comm.world_rank == 0:
            tmr.report_clear("    Simulate noise {:04d}".format(mc))

        # add sky signal
        pipeline_tools.add_signal(args, comm, data, "tot_signal", signalname)
        if comm.comm_world is not None:
            comm.comm_world.barrier()
        if comm.world_rank == 0:
            tmr.report_clear("    Add sky signal {:04d}".format(mc))

        if gain is not None:
            op_apply_gain = OpApplyGain(gain, name="tot_signal")
            op_apply_gain.exec(data)
            if comm.comm_world is not None:
                comm.comm_world.barrier()
            if comm.world_rank == 0:
                tmr.report_clear("    Apply gains {:04d}".format(mc))

        if mc == firstmc:
            # For the first realization, optionally export the
            # timestream data.  If we had observation intervals defined,
            # we could pass "use_interval=True" to the export operators,
            # which would ensure breaks in the exported data at
            # acceptable places.
            pipeline_tools.output_tidas(args, comm, data, "tot_signal")
            pipeline_tools.output_spt3g(args, comm, data, "tot_signal")
            if comm.comm_world is not None:
                comm.comm_world.barrier()
            if comm.world_rank == 0:
                tmr.report_clear("    Write TOD snapshot {:04d}".format(mc))

        if args.use_madam:
            pipeline_tools.apply_madam(
                args, comm, data, madampars, outpath, detweights, "tot_signal"
            )
        else:
            pipeline_tools.apply_mapmaker(args, comm, data, outpath, "tot_signal")

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        if comm.world_rank == 0:
            tmr.report_clear("  Map-making {:04d}".format(mc))

        if comm.comm_world is not None:
            comm.comm_world.barrier()
        if comm.world_rank == 0:
            mctmr.report_clear("  Monte Carlo loop {:04d}".format(mc))

    gt.stop_all()
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    tmr.clear()
    tmr.start()
    alltimers = gather_timers(comm=comm.comm_world)
    if comm.world_rank == 0:
        out = os.path.join(args.outdir, "timing")
        dump_timing(alltimers, out)
        tmr.stop()
        tmr.report("Gather and dump timing info")
        timer0.report_clear("toast_satellite_sim.py")
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
