# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..map import DistPixels
from ..todmap import OpSimPySM, OpSimScan, OpMadam, OpSimConviqt, OpSimWeightedConviqt


def add_sky_signal_args(parser):
    # The sky signal flag may be already added
    try:
        parser.add_argument(
            "--sky",
            required=False,
            action="store_true",
            help="Add simulated sky signal",
            dest="simulate_sky",
        )
        parser.add_argument(
            "--simulate-sky",
            required=False,
            action="store_true",
            help="Add simulated sky",
            dest="simulate_sky",
        )
        parser.add_argument(
            "--no-sky",
            required=False,
            action="store_false",
            help="Do not add simulated sky",
            dest="simulate_sky",
        )
        parser.add_argument(
            "--no-simulate-sky",
            required=False,
            action="store_false",
            help="Do not add simulated sky",
            dest="simulate_sky",
        )
        parser.set_defaults(simulate_sky=True)
    except argparse.ArgumentError:
        pass
    return


def add_sky_map_args(parser):
    """Add the sky arguments"""

    parser.add_argument(
        "--input-map",
        required=False,
        help="Input map for signal.  You can use Python formatting for Monte "
        "Carlo realization, {mc:04i}, and detector name, {detector}",
    )

    # The nside may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int64, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # The coordinate system may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass
    add_sky_signal_args(parser)

    return


def add_pysm_args(parser):
    """Add the sky arguments"""

    parser.add_argument(
        "--pysm-model",
        required=False,
        help="Comma separated models for on-the-fly PySM "
        'simulation, e.g. "s1,d6,f1,a2"',
    )

    parser.add_argument(
        "--pysm-apply-beam",
        required=False,
        action="store_true",
        help="Convolve sky with detector beam",
        dest="pysm_apply_beam",
    )
    parser.add_argument(
        "--no-pysm-apply-beam",
        required=False,
        action="store_false",
        help="Do not convolve sky with detector beam.",
        dest="pysm_apply_beam",
    )
    parser.set_defaults(pysm_apply_beam=True)

    parser.add_argument(
        "--pysm-precomputed-cmb-K_CMB",
        required=False,
        help="Precomputed CMB map for PySM in K_CMB"
        'it overrides any model defined in pysm_model"',
    )

    parser.add_argument(
        "--pysm-mpi-comm",
        required=False,
        help="MPI communicator used by the PySM operator, either 'rank' or 'group'",
        dest="pysm_mpi_comm",
    )
    parser.set_defaults(pysm_mpi_comm="group")

    # The nside may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int64, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # The coordinate system may already be added
    try:
        parser.add_argument(
            "--coord", required=False, default="C", help="Sky coordinate system [C,E,G]"
        )
    except argparse.ArgumentError:
        pass

    add_sky_signal_args(parser)

    return


def add_conviqt_args(parser):
    """Add arguments for synthesizing signal with libConviqt"""
    parser.add_argument(
        "--conviqt-sky-file",
        required=False,
        help="Path to sky alm files. Tag {detector} will be "
        "replaced with detector name.",
    )
    parser.add_argument(
        "--conviqt-lmax",
        required=False,
        type=np.int64,
        help="Simulation lmax.  May not exceed the expansion order in conviqt-sky-file",
    )
    parser.add_argument(
        "--conviqt-fwhm",
        required=False,
        type=np.float,
        help="Sky fwhm [arcmin] to deconvolve",
    )
    parser.add_argument(
        "--conviqt-beam-file",
        required=False,
        help="Path to beam alm files. Tag {detector} will be "
        "replaced with detector name.",
    )
    parser.add_argument(
        "--conviqt-beam-mmax",
        required=False,
        type=np.int64,
        help="Beam mmax.  May not exceed the expansion order in conviqt-beam-file",
    )
    parser.add_argument(
        "--conviqt-pxx",
        required=False,
        action="store_false",
        help="Beams are in Pxx frame, not Dxx",
        dest="conviqt_dxx",
    )
    parser.add_argument(
        "--conviqt-dxx",
        required=False,
        action="store_true",
        help="Beams are in Dxx frame, not Pxx",
        dest="conviqt_dxx",
    )
    parser.set_defaults(conviqt_dxx=True)
    parser.add_argument(
        "--conviqt-order", default=11, type=np.int64, help="Iteration order"
    )
    parser.add_argument(
        "--conviqt-normalize-beam",
        required=False,
        action="store_true",
        help="Normalize the beams",
        dest="conviqt_normalize_beam",
    )
    parser.add_argument(
        "--no-conviqt-normalize-beam",
        required=False,
        action="store_false",
        help="Do not normalize the beams",
        dest="conviqt_normalize_beam",
    )
    parser.set_defaults(conviqt_normalize_beam=False)

    parser.add_argument(
        "--conviqt-calibrate",
        required=False,
        action="store_true",
        help="Normalize the beams",
        dest="conviqt_calibrate",
    )
    parser.add_argument(
        "--no-conviqt-calibrate",
        required=False,
        action="store_false",
        help="Do not normalize the beams",
        dest="conviqt_calibrate",
    )
    parser.set_defaults(conviqt_calibrate=True)
    parser.add_argument(
        "--conviqt-remove-monopole",
        required=False,
        action="store_true",
        help="Remove the sky monopole before convolution",
        dest="conviqt_remove_monopole",
    )
    parser.add_argument(
        "--no-conviqt-remove-monopole",
        required=False,
        action="store_false",
        help="Do not remove the sky monopole before convolution",
        dest="conviqt_remove_monopole",
    )
    parser.set_defaults(conviqt_remove_monopole=False)
    parser.add_argument(
        "--conviqt-remove-dipole",
        required=False,
        action="store_true",
        help="Remove the sky dipole before convolution",
    )
    parser.add_argument(
        "--no-conviqt-remove-dipole",
        required=False,
        action="store_false",
        help="Do not remove the sky dipole before convolution",
    )
    parser.set_defaults(conviqt_remove_dipole=False)
    parser.add_argument(
        "--conviqt-mpi-comm",
        required=False,
        help="MPI communicator used by the OpSimConviqt operator, "
        "either 'rank' or 'group'",
        dest="conviqt_mpi_comm",
    )
    parser.set_defaults(conviqt_mpi_comm="rank")

    add_sky_signal_args(parser)

    return


@function_timer
def apply_conviqt(args, comm, data, cache_prefix="signal", verbose=True, mc=0):
    if (
        args.conviqt_sky_file is None
        or args.conviqt_beam_file is None
        or not args.simulate_sky
    ):
        return None

    log = Logger.get()
    timer = Timer()
    timer.start()

    if comm.world_rank == 0 and verbose:
        log.info("Running Conviqt")

    verbosity = 0
    if verbose:
        verbosity = 1
    if args.debug:
        verbosity = 10

    conviqt = OpSimConviqt(
        getattr(comm, "comm_" + args.conviqt_mpi_comm),
        args.conviqt_sky_file,
        args.conviqt_beam_file,
        lmax=args.conviqt_lmax,
        beammmax=args.conviqt_beam_mmax,
        pol=True,
        fwhm=args.conviqt_fwhm,
        order=args.conviqt_order,
        calibrate=args.conviqt_calibrate,
        dxx=args.conviqt_dxx,
        out=cache_prefix,
        remove_monopole=args.conviqt_remove_monopole,
        remove_dipole=args.conviqt_remove_dipole,
        normalize_beam=args.conviqt_normalize_beam,
        verbosity=verbosity,
        mc=mc,
    )
    conviqt.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Read and sample map")

    return cache_prefix


@function_timer
def apply_weighted_conviqt(args, comm, data, cache_prefix="signal", verbose=True):
    if (
        args.conviqt_sky_file is None
        or args.conviqt_beam_file is None
        or not args.simulate_sky
    ):
        return None

    log = Logger.get()
    timer = Timer()
    timer.start()

    if comm.world_rank == 0 and verbose:
        log.info("Running Weighted Conviqt")

    verbosity = 0
    if verbose:
        verbosity = 1
    if args.debug:
        verbosity = 10

    conviqt = OpSimWeightedConviqt(
        getattr(comm, "comm_" + args.conviqt_mpi_comm),
        args.conviqt_sky_file,
        args.conviqt_beam_file,
        lmax=args.conviqt_lmax,
        beammmax=args.conviqt_beam_mmax,
        pol=True,
        fwhm=args.conviqt_fwhm,
        order=args.conviqt_order,
        calibrate=args.conviqt_calibrate,
        dxx=args.conviqt_dxx,
        out=cache_prefix,
        remove_monopole=args.conviqt_remove_monopole,
        remove_dipole=args.conviqt_remove_dipole,
        normalize_beam=args.conviqt_normalize_beam,
        verbosity=verbosity,
    )
    conviqt.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Read and sample map")

    return cache_prefix


@function_timer
def scan_sky_signal(
    args, comm, data, cache_prefix="signal", verbose=True, pixels="pixels", nnz=3, mc=0
):
    """Scan sky signal from a map."""
    if not args.input_map or not args.simulate_sky:
        return None

    log = Logger.get()
    timer = Timer()
    timer.start()

    # Scan the sky signal

    scansim = OpSimScan(input_map=args.input_map, out=cache_prefix, nnz=nnz, mc=mc)
    scansim.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Read and sample map")

    return cache_prefix


@function_timer
def simulate_sky_signal(
    args, comm, data, focalplanes, cache_prefix, verbose=False, pixels="pixels", mc=0
):
    """Use PySM to simulate smoothed sky signal."""
    if not args.pysm_model or not args.simulate_sky:
        return None
    timer = Timer()
    timer.start()
    fn_cmb = args.pysm_precomputed_cmb_K_CMB
    if fn_cmb is not None:
        fn_cmb = fn_cmb.format(mc)
    # Convolve a signal TOD from PySM
    op_sim_pysm = OpSimPySM(
        data,
        comm=getattr(comm, "comm_" + args.pysm_mpi_comm),
        out=cache_prefix,
        pysm_model=args.pysm_model.split(","),
        pysm_precomputed_cmb_K_CMB=fn_cmb,
        focalplanes=focalplanes,
        apply_beam=args.pysm_apply_beam,
        coord=args.coord,
        pixels=pixels,
    )
    op_sim_pysm.exec(data)
    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("PySM")

    return cache_prefix
