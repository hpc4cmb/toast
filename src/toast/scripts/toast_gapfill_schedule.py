#!/usr/bin/env python3

# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script fills gaps in one TOAST schedule with entries from
another schedule
"""

import argparse
import datetime
import os
import sys
import traceback

import astropy.units as u
import ephem
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from toast.coordinates import to_DJD, to_UTC
from toast.mpi import MPI, Comm, get_world
from toast.pixels_io_healpix import read_healpix, write_healpix
from toast.timing import Timer
from toast.utils import Environment, Logger

import toast


def parse_arguments():
    """Parse the command line arguments"""

    parser = argparse.ArgumentParser(
        description="Gapfill observing schedule with entries from another schedule"
    )

    parser.add_argument(
        "fname_primary",
        help="Primary TOAST observing schedule",
    )

    parser.add_argument(
        "fname_supplement",
        help="Supplemental TOAST observing schedule",
    )

    parser.add_argument(
        "fname_out",
        help="Output TOAST observing schedule",
    )

    parser.add_argument(
        "--supplement_only",
        required=False,
        default=False,
        action="store_true",
        help="Only write the supplemental entries to output schedule",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # load the input schedules
    header = []
    primary = []
    t_primary = 0
    with open(args.fname_primary, "r") as f:
        for iline, line in enumerate(f):
            if iline < 3:
                header.append(line)
                continue
            parts = line.split()
            sstart = f"{parts[0]} {parts[1]}"
            sstop = f"{parts[2]} {parts[3]}"
            start = datetime.datetime.fromisoformat(sstart).timestamp()
            stop = datetime.datetime.fromisoformat(sstop).timestamp()
            t_primary += stop - start
            primary.append([start, stop, line])
        print(
            f"Loaded {len(primary)} entries ({t_primary / 86400:.3f} days) "
            f"from the primary schedule ({args.fname_primary})"
        )

    supplement = []
    t_supplement = 0
    with open(args.fname_supplement, "r") as f:
        for iline, line in enumerate(f):
            if iline < 3:
                continue
            parts = line.split()
            sstart = f"{parts[0]} {parts[1]}"
            sstop = f"{parts[2]} {parts[3]}"
            start = datetime.datetime.fromisoformat(sstart).timestamp()
            stop = datetime.datetime.fromisoformat(sstop).timestamp()
            t_supplement += stop - start
            supplement.append([start, stop, line])
        print(
            f"Loaded {len(supplement)} entries ({t_supplement / 86400:.3f} days) "
            f"from the supplemental "
            f"schedule ({args.fname_supplement})"
        )

    with open(args.fname_out, "w") as f:
        for line in header:
            f.write(line)
        nline = len(primary)
        nfill = 0
        tfill = 0
        for iline in range(nline - 1):
            if not args.supplement_only:
                f.write(primary[iline][2])
            gap_start = primary[iline][1]
            gap_stop = primary[iline + 1][0]
            if gap_stop - gap_start > 3600:
                # see if we can fill the gap from the other schedule
                for start, stop, line in supplement:
                    if stop < gap_start:
                        continue
                    if start > gap_stop:
                        break
                    if gap_start < start and stop < gap_stop:
                        f.write(line)
                        nfill += 1
                        tfill += stop - start
        f.write(primary[-1][2])
        print(
            f"Filled in {nfill} entries ({tfill / 86400:3f} days) "
            f"from the supplemental schedule"
        )

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
