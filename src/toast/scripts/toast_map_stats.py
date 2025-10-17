#!/usr/bin/env python3

# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script loads toast mapmaker outputs and computes various quantities that
are useful for quality assurance, data cuts, and weighted coadds.
"""

import argparse
import os

import numpy as np
import healpy as hp
import astropy.io
from astropy import units as u
from astropy.table import Column, QTable
from matplotlib import pyplot as plt

import toast
from toast.utils import stdouterr_redirected
from toast.pixels_io_healpix import read_healpix
from toast.pixels_io_wcs import read_wcs
from toast.scripts.toast_plot_healpix import main as healpix_main
from toast.scripts.toast_plot_wcs import main as wcs_main


def plot_maps(args, remaining, out_dir):
    # Build up the arguments to pass to the plotting routine
    plot_opts = [
        "--out_dir",
        out_dir,
        "--map_file",
        args.map_file,
        "--hit_file",
        args.hits_file,
        "--cmap",
        "bwr",
    ]
    plot_opts.extend(remaining)
    try:
        healpix_main(opts=plot_opts)
    except Exception as hpe:
        try:
            wcs_main(opts=plot_opts)
        except Exception as wpe:
            msg = "Failed to plot maps as either healpix or WCS"
            raise RuntimeError(msg)


def compute_spectra(args, remaining, out_dir):
    try:
        hits = read_healpix(args.hits_file, nest=False)
        use_healpix = True
    except Exception as ehpx:
        try:
            hits = read_wcs(args.hits_file)
            use_healpix = False
        except Exception as ewcs:
            msg = "Cannot read hits file as either healpix or WCS format"
            raise RuntimeError(msg)

    if not use_healpix:
        raise NotImplementedError("WCS processing not yet implemented")

    good = hits > 3
    bad = np.logical_not(good)
    nside = hp.get_nside(hits)
    data_map, data_header = read_healpix(args.map_file, field=None, nest=False, h=True)

    if "UNITS" in data_header:
        map_units = u.Unit(data_header["UNITS"])
    else:
        map_units = u.K
    spec_units = map_units**2

    # We just use the intensity covariance for weighting
    invcov = read_healpix(args.invcov_file, field=(0,), nest=False)
    invcov /= np.amax(invcov)
    invcov[bad] = 0.0

    fsky = np.mean(invcov**2)
    weighted_map = data_map * invcov
    weighted_map[:, bad] = 0.0
    mono = np.mean(weighted_map[0, good])
    weighted_map[0, good] -= mono

    lmax = 3 * nside
    cl = hp.anafast(weighted_map, lmax=lmax, iter=3) / fsky
    cl_file_fits = os.path.join(out_dir, "pseudo_cl.fits")
    hp.write_cl(cl_file_fits, cl, dtype=np.float64, overwrite=True)

    cl_file = os.path.join(out_dir, "pseudo_cl.ecsv")
    cl_table = QTable(
        [
            Column(name="cl_TT", data=cl[0], unit=spec_units),
            Column(name="cl_EE", data=cl[1], unit=spec_units),
            Column(name="cl_BB", data=cl[2], unit=spec_units),
            Column(name="cl_TE", data=cl[3], unit=spec_units),
            Column(name="cl_EB", data=cl[4], unit=spec_units),
            Column(name="cl_TB", data=cl[5], unit=spec_units),
        ]
    )
    cl_table.meta["toast_version"] = toast.__version__
    cl_table.write(cl_file, format="ascii.ecsv", overwrite=True)

    # Plot in uK
    scale = 1.0 * spec_units
    cl_uK = cl * scale.to_value(u.uK**2)

    img_file = os.path.join(out_dir, "pseudo_cl.pdf")
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ell = np.arange(cl[0].size)
    for icomp, comp in enumerate(["TT", "EE", "BB"]):
        ax = fig.add_subplot(3, 1, icomp + 1)
        ax.loglog(ell[2:], cl_uK[icomp][2:], color="black")
        ax.set_xlabel(r"Multipole, $\ell$")
        ax.set_ylabel(r"C$_\ell^{" + comp + r"}$ [$\mu$K$^2$]")
    fig.tight_layout()
    plt.savefig(img_file)
    plt.close()


def main(opts=None, comm=None):
    log = toast.utils.Logger.get()

    # Get optional MPI parameters
    procs = 1
    rank = 0
    if comm is not None:
        procs = comm.size
        rank = comm.rank

    if "OMP_NUM_THREADS" not in os.environ:
        msg = "OMP_NUM_THREADS not set in the environment. "
        msg += "Job may try to use all cores for threads."
        log.warning_rank(msg, comm=comm)

    # Argument parsing
    parser = argparse.ArgumentParser(description="Toast Map Statistics")

    parser.add_argument(
        "--hits_file",
        required=True,
        type=str,
        default=None,
        help="The hits map",
    )
    parser.add_argument(
        "--map_file",
        required=True,
        type=str,
        default=None,
        help="The map",
    )
    parser.add_argument(
        "--invcov_file",
        required=True,
        type=str,
        default=None,
        help="The inverse pixel covariance",
    )

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default=None,
        help="Write outputs to this directory",
    )
    parser.add_argument(
        "--out_log_name",
        required=False,
        type=str,
        default=None,
        help="Redirect stdout / stderr to this file within `out_dir`",
    )

    args, remaining = parser.parse_known_args(args=opts)

    # Determine the output directory
    if args.out_dir is None:
        # Write to the same directory as the map file
        out_dir = os.path.dirname(args.map_file)
        if out_dir == "":
            out_dir = "."
    else:
        out_dir = args.out_dir

    # One process makes output directory
    if comm is None or comm.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if comm is not None:
        comm.barrier()

    # Redirect stdout / stderr during the run
    if args.out_log_name is None:
        # Do not redirect
        plot_maps(args, remaining, out_dir)
        compute_spectra(args, remaining, out_dir)
    else:
        out_log = os.path.join(out_dir, args.out_log_name)
        with stdouterr_redirected(to=out_log, comm=comm, overwrite=True):
            # Redirect
            plot_maps(args, remaining, out_dir)
            compute_spectra(args, remaining, out_dir)


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(comm=world)
