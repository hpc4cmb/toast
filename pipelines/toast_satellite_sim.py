#!/usr/bin/env python3

# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
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

from toast.map import (
    OpMadam,
    OpAccumDiag,
    OpLocalPixels,
    covariance_apply,
    covariance_invert,
    DistPixels,
)

from toast.tod import (
    regular_intervals,
    AnalyticNoise,
    plot_focalplane,
    OpApplyGain,
    OpSimNoise,
    OpSimDipole,
    OpPointingHpix,
    slew_precession_axis,
    OpSimPySM,
    OpMemoryCounter,
    TODSatellite,
)

# FIXME: put these back into the import statement above after porting.
tidas_available = False
spt3g_available = False

if tidas_available:
    from toast.tod.tidas import OpTidasExport, TODTidas

if spt3g_available:
    from toast.tod.spt3g import Op3GExport, TOD3G


@function_timer
def add_sky_signal(args, comm, data, totalname, signalname):
    """ Add signalname to totalname in the obs tod

    """
    if signalname is not None:
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                cachename_in = "{}_{}".format(signalname, det)
                cachename_out = "{}_{}".format(totalname, det)
                ref_in = tod.cache.reference(cachename_in)
                if comm.comm_world.rank == 0 and args.debug:
                    print(
                        "add_sky_signal", signalname, "to", totalname, flush=args.flush
                    )
                    print(signalname, "min max", ref_in.min(), ref_in.max())
                if tod.cache.exists(cachename_out):
                    ref_out = tod.cache.reference(cachename_out)
                    if comm.comm_world.rank == 0 and args.debug:
                        print(totalname, "min max", ref_out.min(), ref_out.max())
                    ref_out += ref_in
                else:
                    ref_out = tod.cache.put(cachename_out, ref_in)
                if comm.comm_world.rank == 0 and args.debug:
                    print("final", "min max", ref_out.min(), ref_out.max())
                del ref_in, ref_out
    return


@function_timer
def get_submaps(args, comm, data):
    """ Get a list of locally hit pixels and submaps on every process.

    """
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
            "detectors than processes in the group?".format(comm.comm_world.rank)
        )

    # find the locally hit submaps.
    localsm = np.unique(np.floor_divide(localpix, subnpix))

    return localpix, localsm, subnpix


@function_timer
def simulate_sky_signal(
    args, comm, data, mem_counter, focalplanes, subnpix, localsm, signalname
):
    """ Use PySM to simulate smoothed sky signal.

    """
    # Convolve a signal TOD from PySM
    op_sim_pysm = OpSimPySM(
        comm=comm.comm_rank,
        out=signalname,
        pysm_model=args.input_pysm_model,
        pysm_precomputed_cmb_K_CMB=args.input_pysm_precomputed_cmb_K_CMB,
        focalplanes=focalplanes,
        nside=args.nside,
        subnpix=subnpix,
        localsm=localsm,
        apply_beam=args.apply_beam,
        debug=args.debug,
        coord=args.coord,
    )
    op_sim_pysm.exec(data)
    if comm.comm_world.rank == 0:
        tod = data.obs[0]["tod"]
        for det in tod.local_dets:
            ref = tod.cache.reference(signalname + "_" + det)
            print("PySM signal first observation min max", det, ref.min(), ref.max())
            del ref
    del op_sim_pysm
    mem_counter.exec(data)
    return


def main():
    env = Environment.get()
    log = Logger.get()
    gt = GlobalTimers.get()
    gt.start("toast_satellite_sim (total)")

    mpiworld, procs, rank = get_world()
    if rank == 0:
        env.print()
    if mpiworld is None:
        log.info("Running serially with one process")
    else:
        if rank == 0:
            log.info("Running with {} processes".format(procs))

    parser = argparse.ArgumentParser(
        description="Simulate satellite boresight pointing and make a map.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--groupsize",
        required=False,
        type=int,
        default=0,
        help="size of processor groups used to distribute observations",
    )

    parser.add_argument(
        "--samplerate",
        required=False,
        type=float,
        default=40.0,
        help="Detector sample rate (Hz)",
    )

    parser.add_argument(
        "--starttime",
        required=False,
        type=float,
        default=0.0,
        help="The overall start time of the simulation",
    )

    parser.add_argument(
        "--spinperiod",
        required=False,
        type=float,
        default=10.0,
        help="The period (in minutes) of the rotation about the " "spin axis",
    )
    parser.add_argument(
        "--spinangle",
        required=False,
        type=float,
        default=30.0,
        help="The opening angle (in degrees) of the boresight " "from the spin axis",
    )

    parser.add_argument(
        "--precperiod",
        required=False,
        type=float,
        default=50.0,
        help="The period (in minutes) of the rotation about the " "precession axis",
    )
    parser.add_argument(
        "--precangle",
        required=False,
        type=float,
        default=65.0,
        help="The opening angle (in degrees) of the spin axis "
        "from the precession axis",
    )

    parser.add_argument(
        "--hwprpm",
        required=False,
        type=float,
        default=0.0,
        help="The rate (in RPM) of the HWP rotation",
    )
    parser.add_argument(
        "--hwpstep",
        required=False,
        default=None,
        help="For stepped HWP, the angle in degrees of each step",
    )
    parser.add_argument(
        "--hwpsteptime",
        required=False,
        type=float,
        default=0.0,
        help="For stepped HWP, the the time in seconds between " "steps",
    )

    parser.add_argument(
        "--obs",
        required=False,
        type=float,
        default=1.0,
        help="Number of hours in one science observation",
    )
    parser.add_argument(
        "--gap",
        required=False,
        type=float,
        default=0.0,
        help="Cooler cycle time in hours between science obs",
    )
    parser.add_argument(
        "--numobs",
        required=False,
        type=int,
        default=1,
        help="Number of complete observations",
    )

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

    parser.add_argument(
        "--nside", required=False, type=int, default=64, help="Healpix NSIDE"
    )
    parser.add_argument(
        "--subnside",
        required=False,
        type=int,
        default=4,
        help="Distributed pixel sub-map NSIDE",
    )

    parser.add_argument(
        "--coord", required=False, default="E", help="Sky coordinate system [C,E,G]"
    )

    parser.add_argument(
        "--baseline",
        required=False,
        type=float,
        default=60.0,
        help="Destriping baseline length (seconds)",
    )
    parser.add_argument(
        "--noisefilter",
        required=False,
        default=False,
        action="store_true",
        help="Destripe with the noise filter enabled",
    )

    parser.add_argument(
        "--madam",
        required=False,
        default=False,
        action="store_true",
        help="If specified, use libmadam for map-making",
    )
    parser.add_argument(
        "--madampar", required=False, default=None, help="Madam parameter file"
    )

    parser.add_argument(
        "--flush",
        required=False,
        default=False,
        action="store_true",
        help="Flush every print statement.",
    )

    parser.add_argument(
        "--MC_start",
        required=False,
        type=int,
        default=0,
        help="First Monte Carlo noise realization",
    )
    parser.add_argument(
        "--MC_count",
        required=False,
        type=int,
        default=1,
        help="Number of Monte Carlo noise realizations",
    )

    parser.add_argument(
        "--fp",
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

    parser.add_argument(
        "--tidas", required=False, default=None, help="Output TIDAS export path"
    )

    parser.add_argument(
        "--spt3g", required=False, default=None, help="Output SPT3G export path"
    )

    parser.add_argument("--input_map", required=False, help="Input map for signal")
    parser.add_argument(
        "--input_pysm_model",
        required=False,
        help="Comma separated models for on-the-fly PySM "
        'simulation, e.g. s3,d6,f1,a2"',
    )
    parser.add_argument(
        "--input_pysm_precomputed_cmb_K_CMB",
        required=False,
        help="Precomputed CMB map for PySM in K_CMB"
        'it overrides any model defined in input_pysm_model"',
    )
    parser.add_argument(
        "--apply_beam",
        required=False,
        action="store_true",
        help="Apply beam convolution to input map with gaussian "
        "beam parameters defined in focalplane",
    )

    parser.add_argument(
        "--input_dipole",
        required=False,
        help="Simulate dipole, possible values are " "total, orbital, solar",
    )
    parser.add_argument(
        "--input_dipole_solar_speed_kms",
        required=False,
        help="Solar system speed [km/s]",
        type=float,
        default=369.0,
    )
    parser.add_argument(
        "--input_dipole_solar_gal_lat_deg",
        required=False,
        help="Solar system speed galactic latitude [degrees]",
        type=float,
        default=48.26,
    )
    parser.add_argument(
        "--input_dipole_solar_gal_lon_deg",
        required=False,
        help="Solar system speed galactic longitude[degrees]",
        type=float,
        default=263.99,
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

    groupsize = args.groupsize
    if groupsize <= 0:
        groupsize = procs

    # This is the 2-level toast communicator.
    comm = Comm(world=mpiworld, groupsize=groupsize)

    # Parse options
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)

    npix = 12 * args.nside * args.nside

    subnside = args.subnside
    if subnside > args.nside:
        subnside = args.nside
    subnpix = 12 * subnside * subnside

    tmr = Timer()
    tmr.start()

    if rank == 0:
        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

    fp = None
    gain = None

    # Load focalplane information

    if rank == 0:
        if args.fp is None:
            # in this case, create a fake detector at the boresight
            # with a pure white noise spectrum.
            fake = {}
            fake["quat"] = np.array([0.0, 0.0, 1.0, 0.0])
            fake["fwhm"] = 30.0
            fake["fknee"] = 0.0
            fake["fmin"] = 1.0e-5
            fake["alpha"] = 1.0
            fake["NET"] = 1.0
            fake["color"] = "r"
            fp = {}
            fp["bore"] = fake
        else:
            with open(args.fp, "rb") as p:
                fp = pickle.load(p)

        if args.gain is not None:
            gain = {}
            with fits.open(args.gain) as f:
                gain["TIME"] = np.array(f["TIME"].data["TIME"])
                for i_det, det_name in f["DETECTORS"].data["DETECTORS"]:
                    gain[det_name] = np.array(f["GAINS"].data[i_det, :])

    if mpiworld is not None:
        if args.gain is not None:
            gain = mpiworld.bcast(gain, root=0)
        fp = mpiworld.bcast(fp, root=0)

    if rank == 0:
        tmr.report_clear("Create focalplane ({} dets)".format(len(fp.keys())))

    if args.debug:
        if rank == 0:
            outfile = os.path.join(args.outdir, "focalplane.png")
            set_backend()
            dquats = {x: fp[x]["quat"] for x in fp.keys()}
            dfwhm = {x: fp[x]["fwhm"] for x in fp.keys()}
            plot_focalplane(dquats, 10.0, 10.0, outfile, fwhm=dfwhm)

    # Since we are simulating noise timestreams, we want
    # them to be contiguous and reproducible over the whole
    # observation.  We distribute data by detector within an
    # observation, so ensure that our group size is not larger
    # than the number of detectors we have.

    if groupsize > len(fp.keys()):
        if rank == 0:
            log.error("process group is too large for the number of detectors")
            mpiworld.Abort()

    # Detector information from the focalplane

    detectors = sorted(fp.keys())
    detquats = {}
    detindx = None
    if "index" in fp[detectors[0]]:
        detindx = {}

    for d in detectors:
        detquats[d] = fp[d]["quat"]
        if detindx is not None:
            detindx[d] = fp[d]["index"]

    # Distribute the observations uniformly

    groupdist = distribute_uniform(args.numobs, comm.ngroups)

    # Compute global time and sample ranges of all observations

    obsrange = regular_intervals(
        args.numobs,
        args.starttime,
        0,
        args.samplerate,
        3600 * args.obs,
        3600 * args.gap,
    )

    # Create the noise model used for all observations

    fmin = {}
    fknee = {}
    alpha = {}
    NET = {}
    rates = {}
    for d in detectors:
        rates[d] = args.samplerate
        fmin[d] = fp[d]["fmin"]
        fknee[d] = fp[d]["fknee"]
        alpha[d] = fp[d]["alpha"]
        NET[d] = fp[d]["NET"]

    noise = AnalyticNoise(
        rate=rates, fmin=fmin, detectors=detectors, fknee=fknee, alpha=alpha, NET=NET
    )

    mem_counter = OpMemoryCounter()

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
            rate=args.samplerate,
            spinperiod=args.spinperiod,
            spinangle=args.spinangle,
            precperiod=args.precperiod,
            precangle=args.precangle,
            detindx=detindx,
            detranks=comm.group_size,
        )

        obs = {}
        obs["name"] = "science_{:05d}".format(ob)
        obs["tod"] = tod
        obs["intervals"] = None
        obs["baselines"] = None
        obs["noise"] = noise
        obs["id"] = ob

        data.obs.append(obs)

    if rank == 0:
        tmr.report_clear("Read parameters, compute data distribution")

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
            samplerate=args.samplerate,
            degday=degday,
        )

        tod.set_prec_axis(qprec=precquat)
        del precquat

    if rank == 0:
        tmr.report_clear("Construct boresight pointing")

    # make a Healpix pointing matrix.

    pointing = OpPointingHpix(
        nside=args.nside,
        nest=True,
        mode="IQU",
        hwprpm=args.hwprpm,
        hwpstep=hwpstep,
        hwpsteptime=args.hwpsteptime,
    )
    pointing.exec(data)

    if mpiworld is not None:
        mpiworld.barrier()

    if rank == 0:
        tmr.report_clear("Pointing generation")

    localpix, localsm, subnpix = get_submaps(args, comm, data)

    if rank == 0:
        tmr.report_clear("Compute locally hit pixels")

    signalname = "signal"
    has_signal = False
    if args.input_pysm_model:
        has_signal = True
        simulate_sky_signal(
            args, comm, data, mem_counter, [fp], subnpix, localsm, signalname=signalname
        )
        if rank == 0:
            tmr.report_clear("Simulate sky signal")

    if args.input_dipole:
        print("Simulating dipole")
        has_signal = True
        op_sim_dipole = OpSimDipole(
            mode=args.input_dipole,
            solar_speed=args.input_dipole_solar_speed_kms,
            solar_gal_lat=args.input_dipole_solar_gal_lat_deg,
            solar_gal_lon=args.input_dipole_solar_gal_lon_deg,
            out=signalname,
            keep_quats=False,
            keep_vel=False,
            subtract=False,
            coord=args.coord,
            freq=0,  # we could use frequency for quadrupole correction
            flag_mask=255,
            common_flag_mask=255,
        )
        op_sim_dipole.exec(data)
        del op_sim_dipole
        if rank == 0:
            tmr.report_clear("Simulate dipole")

    # Mapmaking.  For purposes of this simulation, we use detector noise
    # weights based on the NET (white noise level).  If the destriping
    # baseline is too long, this will not be the best choice.

    detweights = {}
    for d in detectors:
        net = fp[d]["NET"]
        detweights[d] = 1.0 / (args.samplerate * net * net)

    if not args.madam:
        if rank == 0:
            log.info("Not using Madam, will only make a binned map")

        # find the locally hit submaps.
        localsm = np.unique(np.floor_divide(localpix, subnpix))
        if rank == 0:
            tmr.report_clear("Compute local submaps")

        # construct distributed maps to store the covariance,
        # noise weighted map, and hits

        invnpp = DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=6,
            dtype=np.float64,
            submap=subnpix,
            local=localsm,
        )
        hits = DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=1,
            dtype=np.int64,
            submap=subnpix,
            local=localsm,
        )
        zmap = DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=3,
            dtype=np.float64,
            submap=subnpix,
            local=localsm,
        )

        # compute the hits and covariance once, since the pointing and noise
        # weights are fixed.

        if invnpp.data is not None:
            invnpp.data.fill(0.0)

        if hits.data is not None:
            hits.data.fill(0)

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(data)

        if rank == 0:
            tmr.report_clear("Accumulate N_pp'^1")

        invnpp.allreduce()
        hits.allreduce()

        if rank == 0:
            tmr.report_clear("All reduce N_pp'^1")

        hits.write_healpix_fits(os.path.join(args.outdir, "hits.fits"))
        invnpp.write_healpix_fits(os.path.join(args.outdir, "invnpp.fits"))

        if mpiworld is not None:
            mpiworld.barrier()

        if rank == 0:
            tmr.report_clear("Writing hits and N_pp'^1")

        # invert it
        covariance_invert(invnpp, 1.0e-3)

        if mpiworld is not None:
            mpiworld.barrier()
        if rank == 0:
            tmr.report_clear("Invert N_pp'^1")

        invnpp.write_healpix_fits(os.path.join(args.outdir, "npp.fits"))

        if mpiworld is not None:
            mpiworld.barrier()
        if rank == 0:
            tmr.report_clear("Write N_pp'")

        # in debug mode, print out data distribution information
        if args.debug:
            handle = None
            if rank == 0:
                handle = open(os.path.join(args.outdir, "distdata.txt"), "w")
            data.info(handle)
            if rank == 0:
                handle.close()

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("Dumping data distribution")

        # Loop over Monte Carlos

        firstmc = int(args.MC_start)
        nmc = int(args.MC_count)

        for mc in range(firstmc, firstmc + nmc):
            mctmr = Timer()
            mctmr.start()

            # create output directory for this realization
            outpath = os.path.join(args.outdir, "mc_{:03d}".format(mc))
            if rank == 0:
                if not os.path.isdir(outpath):
                    os.makedirs(outpath)

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("Creating output dir {:04d}".format(mc))

            # clear all signal data from the cache, so that we can generate
            # new noise timestreams.
            for obs in data.obs:
                tod = obs["tod"]
                tod.cache.clear("tot_signal_.*")

            # simulate noise

            nse = OpSimNoise(out="tot_signal", realization=mc)
            nse.exec(data)

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Noise simulation {:04d}".format(mc))

            # add sky signal
            if has_signal:
                add_sky_signal(
                    args, comm, data, totalname="tot_signal", signalname=signalname
                )

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Add sky signal {:04d}".format(mc))

            if gain is not None:
                op_apply_gain = OpApplyGain(gain, name="tot_signal")
                op_apply_gain.exec(data)

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Apply gains {:04d}".format(mc))

            if mc == firstmc:
                # For the first realization, optionally export the
                # timestream data.  If we had observation intervals defined,
                # we could pass "use_interval=True" to the export operators,
                # which would ensure breaks in the exported data at
                # acceptable places.
                if args.tidas is not None:
                    tidas_path = os.path.abspath(args.tidas)
                    export = OpTidasExport(
                        tidas_path,
                        TODTidas,
                        backend="hdf5",
                        use_todchunks=True,
                        create_opts={"group_dets": "sim"},
                        ctor_opts={"group_dets": "sim"},
                        cache_name="tot_signal",
                    )
                    export.exec(data)

                    if mpiworld is not None:
                        mpiworld.barrier()
                    if rank == 0:
                        tmr.report_clear("  TIDAS export")

                if args.spt3g is not None:
                    spt3g_path = os.path.abspath(args.spt3g)
                    export = Op3GExport(
                        spt3g_path,
                        TOD3G,
                        use_todchunks=True,
                        export_opts={"prefix": "sim"},
                        cache_name="tot_signal",
                    )
                    export.exec(data)

                    if mpiworld is not None:
                        mpiworld.barrier()
                    if rank == 0:
                        tmr.report_clear("  SPT3G export")

            if zmap.data is not None:
                zmap.data.fill(0.0)
            build_zmap = OpAccumDiag(
                zmap=zmap, name="tot_signal", detweights=detweights
            )
            build_zmap.exec(data)
            zmap.allreduce()

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Building noise weighted map {:04d}".format(mc))

            covariance_apply(invnpp, zmap)

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Computing binned map {:04d}".format(mc))

            zmap.write_healpix_fits(os.path.join(outpath, "binned.fits"))

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Writing binned map {:04d}".format(mc))
                mctmr.report_clear("  Map-making {:04d}".format(mc))
    else:

        # Set up MADAM map making.

        pars = {}

        cross = args.nside // 2

        pars["temperature_only"] = "F"
        pars["force_pol"] = "T"
        pars["kfirst"] = "T"
        pars["concatenate_messages"] = "T"
        pars["write_map"] = "T"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "T"
        pars["write_wcov"] = "T"
        pars["write_hits"] = "T"
        pars["nside_cross"] = cross
        pars["nside_submap"] = subnside

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

        pars["base_first"] = args.baseline
        pars["nside_map"] = args.nside
        if args.noisefilter:
            pars["kfilter"] = "T"
        else:
            pars["kfilter"] = "F"
        pars["fsample"] = args.samplerate

        # in debug mode, print out data distribution information
        if args.debug:
            handle = None
            if rank == 0:
                handle = open(os.path.join(args.outdir, "distdata.txt"), "w")
            data.info(handle)
            if rank == 0:
                handle.close()
            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("Dumping data distribution")

        # Loop over Monte Carlos

        firstmc = int(args.MC_start)
        nmc = int(args.MC_count)

        for mc in range(firstmc, firstmc + nmc):
            mctmr = Timer()
            mctmr.start()

            # create output directory for this realization
            pars["path_output"] = os.path.join(args.outdir, "mc_{:03d}".format(mc))
            if rank == 0:
                if not os.path.isdir(pars["path_output"]):
                    os.makedirs(pars["path_output"])

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("Creating output dir {:04d}".format(mc))

            # clear all total signal data from the cache, so that we can generate
            # new noise timestreams.
            for obs in data.obs:
                tod = obs["tod"]
                tod.cache.clear("tot_signal_.*")

            # simulate noise

            nse = OpSimNoise(out="tot_signal", realization=mc)
            nse.exec(data)

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Noise simulation {:04d}".format(mc))

            # add sky signal
            if has_signal:
                add_sky_signal(
                    args, comm, data, totalname="tot_signal", signalname=signalname
                )

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Add sky signal {:04d}".format(mc))

            if gain is not None:
                op_apply_gain = OpApplyGain(gain, name="tot_signal")
                op_apply_gain.exec(data)

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                tmr.report_clear("  Apply gains {:04d}".format(mc))

            madam = OpMadam(params=pars, detweights=detweights, name="tot_signal")
            madam.exec(data)

            if mpiworld is not None:
                mpiworld.barrier()
            if rank == 0:
                mctmr.report_clear("  Map-making {:04d}".format(mc))

    gt.stop_all()
    if mpiworld is not None:
        mpiworld.barrier()
    tmr.stop()
    tmr.clear()
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
