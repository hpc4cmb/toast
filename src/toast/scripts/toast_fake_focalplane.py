#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Generate a focalplane file compatible with the example workflows.
"""

import argparse
import sys

import numpy as np
from astropy import units as u

import toast
from toast.instrument_sim import fake_hexagon_focalplane, plot_focalplane
from toast.io import H5File
from toast.mpi import get_world
from toast.utils import Logger


def main():
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Simulate a fake hexagonal focalplane."
    )

    parser.add_argument(
        "--min_pix",
        required=False,
        type=int,
        default=7,
        help="Choose a hexagonal packing with at least this many pixels",
    )

    parser.add_argument(
        "--out",
        required=False,
        default="focalplane.h5",
        help="Name of output HDF5 file",
    )

    parser.add_argument(
        "--fwhm_arcmin",
        required=False,
        type=float,
        default=10.0,
        help="beam FWHM in arcmin",
    )

    parser.add_argument(
        "--fwhm_sigma",
        required=False,
        type=float,
        default=0,
        help="Relative beam FWHM distribution width",
    )

    parser.add_argument(
        "--fov_deg",
        required=False,
        type=float,
        default=5.0,
        help="Field of View in degrees",
    )

    parser.add_argument(
        "--sample_rate",
        required=False,
        type=float,
        default=50.0,
        help="Detector sample rate in Hz",
    )

    parser.add_argument(
        "--pol_leakage",
        required=False,
        type=float,
        default=0.0,
        help="Detector polarization leakage (0 = perfect polarizer)",
    )

    parser.add_argument(
        "--psd_fknee",
        required=False,
        type=float,
        default=0.05,
        help="Detector noise model f_knee in Hz",
    )

    parser.add_argument(
        "--psd_net",
        required=False,
        type=float,
        default=50.0e-6,
        help="Detector noise model NET in K*sqrt(sec)",
    )

    parser.add_argument(
        "--psd_alpha",
        required=False,
        type=float,
        default=1.0,
        help="Detector noise model slope",
    )

    parser.add_argument(
        "--psd_fmin",
        required=False,
        type=float,
        default=1.0e-5,
        help="Detector noise model f_min in Hz",
    )

    parser.add_argument(
        "--bandcenter_ghz",
        required=False,
        type=float,
        default=150.0,
        help="Band center frequency in GHz",
    )

    parser.add_argument(
        "--bandcenter_sigma",
        required=False,
        type=float,
        default=0,
        help="Band center distribution width in GHz",
    )

    parser.add_argument(
        "--bandwidth_ghz",
        required=False,
        type=float,
        default=20.0,
        help="Bandwidth in GHz",
    )

    parser.add_argument(
        "--bandwidth_sigma",
        required=False,
        type=float,
        default=0,
        help="Bandwidth distribution width in GHz",
    )

    parser.add_argument(
        "--random_seed",
        required=False,
        type=int,
        default=123456,
        help="Random number generator seed for randomized detector parameters",
    )

    args = parser.parse_args()

    log = Logger.get()

    npix = 1
    ring = 1
    while npix < args.min_pix:
        npix += 6 * ring
        ring += 1

    msg = "Using {} hexagon-packed pixels, which is >= requested number ({})".format(
        npix, args.min_pix
    )
    log.info(msg)

    fp = fake_hexagon_focalplane(
        n_pix=npix,
        width=args.fov_deg * u.degree,
        sample_rate=args.sample_rate * u.Hz,
        epsilon=args.pol_leakage,
        fwhm=args.fwhm_arcmin * u.arcmin,
        bandcenter=args.bandcenter_ghz * u.GHz,
        bandwidth=args.bandwidth_ghz * u.GHz,
        psd_net=args.psd_net * u.K * np.sqrt(1 * u.second),
        psd_fmin=args.psd_fmin * u.Hz,
        psd_alpha=args.psd_alpha,
        psd_fknee=args.psd_fknee * u.Hz,
        fwhm_sigma=args.fwhm_sigma * u.arcmin,
        bandcenter_sigma=args.bandcenter_sigma * u.GHz,
        bandwidth_sigma=args.bandwidth_sigma * u.GHz,
        random_seed=args.random_seed,
    )

    # Guard against being called with multiple processes
    mpiworld, procs, rank = get_world()
    if rank == 0:
        with H5File(args.out, "w", comm=mpiworld, force_serial=True) as f:
            fp.save_hdf5(f.handle)
        plotfile = "{}.pdf".format(args.out)
        plot_focalplane(
            fp,
            args.fov_deg * u.degree,
            args.fov_deg * u.degree,
            plotfile,
            show_labels=True,
        )


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
