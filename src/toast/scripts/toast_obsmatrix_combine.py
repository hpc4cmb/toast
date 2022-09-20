#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script converts HEALPiX maps between FITS and HDF5
"""

import argparse
import os
import sys
import traceback

import h5py
import healpy as hp
import numpy as np

import toast
from toast.mpi import Comm, get_world
from toast.ops import combine_observation_matrix
from toast.utils import Environment, Logger, Timer


def main():
    env = Environment.get()
    log = Logger.get()
    comm, procs, rank = get_world()
    timer0 = Timer()
    timer = Timer()
    timer0.start()
    timer.start()

    parser = argparse.ArgumentParser(
        description="Convert HEALPiX maps between FITS and HDF5"
    )

    parser.add_argument(
        "rootname",
        help="Root name of the observation matrix slices",
    )

    args = parser.parse_args()

    if rank == 0:
        fname_matrix = combine_observation_matrix(args.rootname)

    log.info_rank(
        f"Wrote combined matrix to {fname_matrix} in", timer=timer0, comm=comm
    )

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
