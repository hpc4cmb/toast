#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Generate a focalplane pickle file compatible with the example scripts.
"""

import sys
import argparse
import traceback

import pickle

import numpy as np

from scipy.constants import degree

from toast.mpi import get_world

from toast.utils import Logger

from toast.tod import hex_pol_angles_qu, hex_layout, plot_focalplane


def main():
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Simulate fake hexagonal focalplane.", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--minpix",
        required=False,
        type=int,
        default=100,
        help="minimum number of pixels to use",
    )

    parser.add_argument(
        "--out",
        required=False,
        default="fp_fake",
        help="Root name of output pickle file",
    )

    parser.add_argument(
        "--fwhm", required=False, type=float, default=5.0, help="beam FWHM in arcmin"
    )

    parser.add_argument(
        "--fwhm_sigma",
        required=False,
        type=float,
        default=0,
        help="Relative beam FWHM distribution width",
    )

    parser.add_argument(
        "--fov",
        required=False,
        type=float,
        default=5.0,
        help="Field of View in degrees",
    )

    parser.add_argument(
        "--psd_fknee",
        required=False,
        type=float,
        default=0.05,
        help="Detector noise model f_knee in Hz",
    )

    parser.add_argument(
        "--psd_NET",
        required=False,
        type=float,
        default=60.0e-6,
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
        help="Band center frequency [GHz]",
    )

    parser.add_argument(
        "--bandcenter_sigma",
        required=False,
        type=float,
        default=0,
        help="Relative band center distribution width",
    )

    parser.add_argument(
        "--bandwidth_ghz", required=False, type=float, help="Bandwidth [GHz]"
    )

    parser.add_argument(
        "--bandwidth_sigma",
        required=False,
        type=float,
        default=0,
        help="Relative bandwidth distribution width",
    )

    parser.add_argument(
        "--random_seed",
        required=False,
        type=np.int,
        default=123456,
        help="Random number generator seed for randomized " "detector parameters",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    # Get the default communicator
    mpiworld, procs, rank = get_world()

    # Guard against being called with multiple processes
    if rank == 0:
        # Make one big hexagon layout at the center of the focalplane.
        # Compute the number of pixels that is at least the number requested.

        test = args.minpix - 1
        nrings = 0
        while (test - 6 * nrings) > 0:
            test -= 6 * nrings
            nrings += 1

        npix = 1
        for r in range(1, nrings + 1):
            npix += 6 * r

        log.info("using {} pixels ({} detectors)".format(npix, npix * 2))

        # Translate the field-of-view into distance between flag sides
        angwidth = args.fov * np.cos(30 * degree)

        Apol = hex_pol_angles_qu(npix, offset=0.0)
        Bpol = hex_pol_angles_qu(npix, offset=90.0)

        Adets = hex_layout(npix, angwidth, "fake_", "A", Apol)
        Bdets = hex_layout(npix, angwidth, "fake_", "B", Bpol)

        dets = Adets.copy()
        dets.update(Bdets)

        np.random.seed(args.random_seed)

        for indx, d in enumerate(sorted(dets.keys())):
            dets[d]["fknee"] = args.psd_fknee
            dets[d]["fmin"] = args.psd_fmin
            dets[d]["alpha"] = args.psd_alpha
            dets[d]["NET"] = args.psd_NET
            # This is in degrees, but the input is in arcmin.
            dets[d]["fwhm_deg"] = (args.fwhm / 60.0) * (
                1 + np.random.randn() * args.fwhm_sigma
            )
            # This is a fixed value, in arcmin.
            dets[d]["fwhm"] = args.fwhm
            if args.bandcenter_ghz:
                dets[d]["bandcenter_ghz"] = args.bandcenter_ghz * (
                    1 + np.random.randn() * args.bandcenter_sigma
                )
            if args.bandwidth_ghz:
                dets[d]["bandwidth_ghz"] = args.bandwidth_ghz * (
                    1 + np.random.randn() * args.bandwidth_sigma
                )
            dets[d]["index"] = indx

        outfile = "{}_{}".format(args.out, npix)
        qdets = {x: y["quat"] for x, y in dets.items()}
        beams = {x: (60.0 * y["fwhm_deg"]) for x, y in dets.items()}
        plot_focalplane(qdets, args.fov, args.fov, "{}.png".format(outfile), fwhm=beams)

        with open("{}.pkl".format(outfile), "wb") as p:
            pickle.dump(dets, p)

    return


if __name__ == "__main__":
    try:
        main()
    except:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None:
            mpiworld.Abort(6)
