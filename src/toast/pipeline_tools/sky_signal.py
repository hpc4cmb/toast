# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..map import DistPixels
from ..todmap import OpSimPySM, OpSimScan, OpMadam, OpSimConviqt


def add_sky_map_args(parser):
    """ Add the sky arguments
    """

    parser.add_argument("--input-map", required=False, help="Input map for signal")

    # The nside may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
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

    return


def add_pysm_args(parser):
    """ Add the sky arguments
    """

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
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
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

    return


def add_conviqt_args(parser):
    """ Add arguments for synthesizing signal with libConviqt

    """
    parser.add_argument(
        "--conviqt-sky-file",
        required=True,
        help="Path to sky alm files. Tag DETECTOR will be "
        "replaced with detector name.",
    )
    parser.add_argument(
        "--conviqt-lmax",
        required=False,
        type=np.int,
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
        required=True,
        help="Path to beam alm files. Tag DETECTOR will be "
        "replaced with detector name.",
    )
    parser.add_argument(
        "--conviqt-beam-mmax",
        required=False,
        type=np.int,
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
        "--conviqt-order", default=11, type=np.int, help="Iteration order",
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

    return


@function_timer
def apply_conviqt(args, comm, data, cache_prefix="signal", verbose=True):
    log = Logger.get()
    timer = Timer()
    timer.start()

    # Assemble detector data (name, sky file, beam file, epsilon, psipol)

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
        calibrate=True,
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
    args, comm, data, cache_prefix="signal", verbose=True, pixels="pixels",
):
    """ Scan sky signal from a map.

    """
    if not args.input_map:
        return None

    log = Logger.get()
    timer = Timer()
    timer.start()

    if comm.world_rank == 0 and verbose:
        log.info("Scanning input map")

    npix = 12 * args.nside ** 2

    # Scan the sky signal
    if comm.world_rank == 0 and not os.path.isfile(args.input_map):
        raise RuntimeError("Input map does not exist: {}".format(args.input_map))
    distmap = DistPixels(data, nnz=3, dtype=np.float32, pixels=pixels,)
    distmap.read_healpix_fits(args.input_map)
    scansim = OpSimScan(distmap=distmap, out=cache_prefix)
    scansim.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    if comm.world_rank == 0 and verbose:
        timer.report_clear("Read and sample map")

    return cache_prefix


@function_timer
def simulate_sky_signal(
    args, comm, data, focalplanes, cache_prefix, verbose=False, pixels="pixels",
):
    """ Use PySM to simulate smoothed sky signal.

    """
    if not args.pysm_model:
        return None
    timer = Timer()
    timer.start()
    # Convolve a signal TOD from PySM
    op_sim_pysm = OpSimPySM(
        data,
        comm=getattr(comm, "comm_" + args.pysm_mpi_comm),
        out=cache_prefix,
        pysm_model=args.pysm_model.split(","),
        pysm_precomputed_cmb_K_CMB=args.pysm_precomputed_cmb_K_CMB,
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
