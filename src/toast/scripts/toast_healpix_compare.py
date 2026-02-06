#!/usr/bin/env python3

# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script loads two HEALPix maps and performs a simple comparison"""

import argparse
import os
import sys
import traceback

import h5py
import healpy as hp
import numpy as np
from toast._libtoast import cov_apply_diag, cov_eigendecompose_diag
from toast.covariance import covariance_apply, covariance_invert
from toast.mpi import MPI, Comm, get_world
from toast.pixels_io_healpix import read_healpix, write_healpix
from toast.timing import Timer
from toast.utils import Environment, Logger

import toast
from toast import PixelData, PixelDistribution


def load_map(fname):
    if fname is None:
        return None
    log = Logger.get()
    timer = Timer()
    timer.start()
    try:
        m = read_healpix(fname, None, nest=True, dtype=float)
    except Exception as e:
        msg = f"Failed to load HEALPix map: {e}"
        raise RuntimeError(msg)
    log.info_rank(f"Loaded {fname} in", timer=timer, comm=None)
    m = np.atleast_2d(m)
    return m


def parse_args(opts):
    parser = argparse.ArgumentParser(description="Compare HEALPix maps")

    parser.add_argument(
        "map1",
        help="First input map",
    )

    parser.add_argument(
        "map2",
        help="Second input map",
        nargs="?",
    )

    args = parser.parse_args(args=opts)

    return args


def main(opts=None, comm=None):
    env = Environment.get()
    log = Logger.get()

    timer0 = Timer()
    timer1 = Timer()
    timer = Timer()
    timer0.start()
    timer1.start()
    timer.start()

    args = parse_args(opts)

    m = load_map(args.map1)
    m2 = load_map(args.map2)

    nside = hp.get_nside(m)
    if m2 is not None:
        nside2 = hp.get_nside(m2)
        if nside != nside2:
            msg = f"NSides do not agree: {nside} != {nside2}"
            raise RuntimeError(msg)
    log.info(f"Nside = {nside}")
    npix = hp.nside2npix(nside)

    nmap = len(m)
    if m2 is not None:
        nmap2 = len(m2)
        if nmap != nmap2:
            msg = f"Number of components do not agree: {nmap} != {nmap2}"
            raise RuntimeError(msg)
    log.info(f"Nmap = {nmap}")

    for imap in range(nmap):
        log.info(f"Component # {imap}")
        good = m[imap] != 0
        nnz = np.sum(good)
        mm = m[imap][good]
        log.info(f"    nonzero : {nnz} ({nnz / npix * 100:.3f}%)")
        log.info(f"    mean = {np.mean(mm)}, rms = {np.std(mm)}")
        if m2 is not None:
            good2 = m2[imap] != 0
            mm2 = m2[imap][good2]
            if np.any(good != good2):
                nnz2 = np.sum(good2)
                ndiff = np.sum(good != good2)
                log.info(f"    nonzeros differ in {ndiff} pixels")
                log.info(f"    Map 2 : {nnz2} " f"({nnz2 / npix * 100:.3f}%) nonzeros")
            log.info(f"    Map 2 mean = {np.mean(mm2)}, rms = {np.std(mm2)}")
            good12 = np.logical_and(good, good2)
            dm = (m - m2)[imap][good12]
            log.info(f"    Diff mean = {np.mean(dm)}, rms = {np.std(dm)}")

    log.info_rank("Comparison done in", timer=timer0, comm=None)

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(comm=world)
