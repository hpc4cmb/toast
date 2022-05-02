#!/usr/bin/env python3

# FIXME:  Port this pipeline to 2.3.


# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os
import pickle
import re
import sys
import time
import traceback

import numpy as np

import toast
import toast.map as tm
import toast.qarray as qa
import toast.timing as timing
import toast.tod as tt
import toast.todmap as ttm
from toast.mpi import MPI, finalize
from toast.vis import set_backend

if tt.tidas_available:
    from toast.tod import tidas as tds

if tt.spt3g_available:
    from toast.tod import spt3g as s3g


def elapsed(mcomm, start, msg):
    mcomm.barrier()
    stop = MPI.Wtime()
    dur = stop - start
    if mcomm.rank == 0:
        print("{}: {:.3f} s".format(msg, dur), flush=True)
    return stop


def main():

    if MPI.COMM_WORLD.rank == 0:
        print("Running with {} processes".format(MPI.COMM_WORLD.size), flush=True)

    global_start = MPI.Wtime()

    parser = argparse.ArgumentParser(
        description="Read existing data and make a simple map.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--groupsize",
        required=False,
        type=int,
        default=0,
        help="size of processor groups used to distribute " "observations",
    )

    parser.add_argument(
        "--hwprpm",
        required=False,
        type=float,
        default=0.0,
        help="The rate (in RPM) of the HWP rotation",
    )

    parser.add_argument(
        "--samplerate",
        required=False,
        default=100.0,
        type=np.float,
        help="Detector sample rate (Hz)",
    )

    parser.add_argument(
        "--outdir", required=False, default="out", help="Output directory"
    )

    parser.add_argument(
        "--nside", required=False, type=int, default=64, help="Healpix NSIDE"
    )

    parser.add_argument(
        "--subnside",
        required=False,
        type=int,
        default=8,
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
        "--polyorder",
        required=False,
        type=int,
        help="Polynomial order for the polyfilter",
    )

    parser.add_argument(
        "--wbin_ground",
        required=False,
        type=float,
        help="Ground template bin width [degrees]",
    )

    parser.add_argument(
        "--flush",
        required=False,
        default=False,
        action="store_true",
        help="Flush every print statement.",
    )

    parser.add_argument(
        "--tidas", required=False, default=None, help="Input TIDAS volume"
    )

    parser.add_argument(
        "--tidas_detgroup", required=False, default=None, help="TIDAS detector group"
    )

    parser.add_argument(
        "--spt3g", required=False, default=None, help="Input SPT3G data directory"
    )

    parser.add_argument(
        "--spt3g_prefix",
        required=False,
        default=None,
        help="SPT3G data frame file prefix",
    )

    parser.add_argument(
        "--common_flag_mask",
        required=False,
        default=0,
        type=np.uint8,
        help="Common flag mask",
    )

    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Write data distribution info and focalplane plot",
    )

    args = timing.add_arguments_and_parse(parser, timing.FILE(noquotes=True))
    # args = parser.parse_args(sys.argv)

    autotimer = timing.auto_timer("@{}".format(timing.FILE()))

    if (args.tidas is not None) and (args.spt3g is not None):
        raise RuntimeError("Cannot read two datasets!")

    if (args.tidas is None) and (args.spt3g is None):
        raise RuntimeError("No dataset specified!")

    if args.tidas is not None:
        if not tt.tidas_available:
            raise RuntimeError("TIDAS not found- cannot load")

    if args.spt3g is not None:
        if not tt.spt3g_available:
            raise RuntimeError("SPT3G not found- cannot load")

    groupsize = args.groupsize
    if groupsize == 0:
        groupsize = MPI.COMM_WORLD.size

    # Pixelization

    nside = args.nside
    npix = 12 * args.nside * args.nside
    subnside = args.subnside
    if subnside > nside:
        subnside = nside
    subnpix = 12 * subnside * subnside

    # This is the 2-level toast communicator.

    if MPI.COMM_WORLD.size % groupsize != 0:
        if MPI.COMM_WORLD.rank == 0:
            print(
                "WARNING:  process groupsize does not evenly divide into "
                "total number of processes",
                flush=True,
            )
    comm = toast.Comm(world=MPI.COMM_WORLD, groupsize=groupsize)

    # Create output directory

    mtime = MPI.Wtime()

    if comm.comm_world.rank == 0:
        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)

    mtime = elapsed(comm.comm_world, mtime, "Creating output directory")

    # The distributed timestream data

    data = None

    if args.tidas is not None:
        if args.tidas_detgroup is None:
            raise RuntimeError("you must specify the detector group")
        data = tds.load_tidas(
            comm,
            comm.group_size,
            args.tidas,
            "r",
            args.tidas_detgroup,
            tds.TODTidas,
            group_dets=args.tidas_detgroup,
            distintervals="chunks",
        )

    if args.spt3g is not None:
        if args.spt3g_prefix is None:
            raise RuntimeError("you must specify the frame file prefix")
        data = s3g.load_spt3g(
            comm,
            comm.group_size,
            args.spt3g,
            args.spt3g_prefix,
            s3g.obsweight_spt3g,
            s3g.TOD3G,
        )

    mtime = elapsed(comm.comm_world, mtime, "Distribute data")

    # In debug mode, print out data distribution information

    if args.debug:
        handle = None
        if comm.comm_world.rank == 0:
            handle = open("{}_distdata.txt".format(args.outdir), "w")
        data.info(handle)
        if comm.comm_world.rank == 0:
            handle.close()
        mtime = elapsed(comm.comm_world, mtime, "Dumping debug data distribution")
        if comm.comm_world.rank == 0:
            outfile = "{}_focalplane.png".format(args.outdir)
            set_backend()
            # Just plot the dets from the first TOD
            temptod = data.obs[0]["tod"]
            # FIXME: change this once we store det info in the metadata.
            dfwhm = {x: 10.0 for x in temptod.detectors}
            tt.plot_focalplane(temptod.detoffset(), 10.0, 10.0, outfile, fwhm=dfwhm)
        comm.comm_world.barrier()
        mtime = elapsed(comm.comm_world, mtime, "Plotting debug focalplane")

    # Compute pointing matrix

    pointing = tt.OpPointingHpix(
        nside=args.nside, nest=True, mode="IQU", hwprpm=args.hwprpm
    )
    pointing.exec(data)

    mtime = elapsed(comm.comm_world, mtime, "Expand pointing")

    # Mapmaking.

    # FIXME:  We potentially have a different noise model for every
    # observation.  We need to have both spt3g and tidas format Noise
    # classes which read the information from disk.  Then the mapmaking
    # operators need to get these noise weights from each observation.
    detweights = {d: 1.0 for d in data.obs[0]["tod"].detectors}

    if not args.madam:
        if comm.comm_world.rank == 0:
            print("Not using Madam, will only make a binned map!", flush=True)

        # Filter data if desired

        if args.polyorder:
            polyfilter = tt.OpPolyFilter(
                order=args.polyorder, common_flag_mask=args.common_flag_mask
            )
            polyfilter.exec(data)
            mtime = elapsed(comm.comm_world, mtime, "Polynomial filtering")

        if args.wbin_ground:
            groundfilter = tt.OpGroundFilter(
                wbin=args.wbin_ground, common_flag_mask=args.common_flag_mask
            )
            groundfilter.exec(data)
            mtime = elapsed(comm.comm_world, mtime, "Ground template filtering")

        # Compute pixel space distribution

        lc = tm.OpLocalPixels()
        localpix = lc.exec(data)
        if localpix is None:
            raise RuntimeError(
                "Process {} has no hit pixels. Perhaps there are fewer "
                "detectors than processes in the group?".format(comm.comm_world.rank)
            )
        localsm = np.unique(np.floor_divide(localpix, subnpix))
        mtime = elapsed(comm.comm_world, mtime, "Compute local submaps")

        # construct distributed maps to store the covariance,
        # noise weighted map, and hits

        mtime = MPI.Wtime()
        invnpp = tm.DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=6,
            dtype=np.float64,
            submap=subnpix,
            local=localsm,
        )
        hits = tm.DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=1,
            dtype=np.int64,
            submap=subnpix,
            local=localsm,
        )
        zmap = tm.DistPixels(
            comm=comm.comm_world,
            size=npix,
            nnz=3,
            dtype=np.float64,
            submap=subnpix,
            local=localsm,
        )

        # compute the hits and covariance.

        invnpp.data.fill(0.0)
        hits.data.fill(0)

        build_invnpp = tm.OpAccumDiag(
            detweights=detweights,
            invnpp=invnpp,
            hits=hits,
            common_flag_mask=args.common_flag_mask,
        )
        build_invnpp.exec(data)

        invnpp.allreduce()
        hits.allreduce()
        mtime = elapsed(comm.comm_world, mtime, "Building hits and N_pp^-1")

        hits.write_healpix_fits("{}_hits.fits".format(args.outdir))
        invnpp.write_healpix_fits("{}_invnpp.fits".format(args.outdir))
        mtime = elapsed(comm.comm_world, mtime, "Writing hits and N_pp^-1")

        # invert it
        tm.covariance_invert(invnpp, 1.0e-3)
        mtime = elapsed(comm.comm_world, mtime, "Inverting N_pp^-1")

        invnpp.write_healpix_fits("{}_npp.fits".format(args.outdir))
        mtime = elapsed(comm.comm_world, mtime, "Writing N_pp")

        zmap.data.fill(0.0)
        build_zmap = tm.OpAccumDiag(
            zmap=zmap, detweights=detweights, common_flag_mask=args.common_flag_mask
        )
        build_zmap.exec(data)
        zmap.allreduce()
        mtime = elapsed(comm.comm_world, mtime, "Building noise weighted map")

        tm.covariance_apply(invnpp, zmap)
        mtime = elapsed(comm.comm_world, mtime, "Computing binned map")

        zmap.write_healpix_fits(os.path.join(args.outdir, "binned.fits"))
        mtime = elapsed(comm.comm_world, mtime, "Writing binned map")

    else:
        # Set up MADAM map making.

        pars = {}
        pars["temperature_only"] = "F"
        pars["force_pol"] = "T"
        pars["kfirst"] = "T"
        pars["concatenate_messages"] = "T"
        pars["write_map"] = "T"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "T"
        pars["write_wcov"] = "T"
        pars["write_hits"] = "T"
        pars["nside_cross"] = nside // 2
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
        pars["nside_map"] = nside
        if args.noisefilter:
            pars["kfilter"] = "T"
        else:
            pars["kfilter"] = "F"
        pars["fsample"] = args.samplerate

        madam = tm.OpMadam(
            params=pars, detweights=detweights, common_flag_mask=args.common_flag_mask
        )
        madam.exec(data)
        mtime = elapsed(comm.comm_world, mtime, "Madam mapmaking")

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    dur = stop - global_start
    if comm.comm_world.rank == 0:
        print("Total Time:  {:.2f} seconds".format(dur), flush=True)
    return


if __name__ == "__main__":
    try:
        main()
        tman = timing.timing_manager()
        tman.report()
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(MPI.COMM_WORLD.rank, x) for x in lines]
        print("".join(lines), flush=True)
        toast.raise_error(6)  # typical error code for SIGABRT
        MPI.COMM_WORLD.Abort(6)
    finalize()
