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
from toast.pixels_io_healpix import (
    filename_is_fits,
    filename_is_hdf5,
    read_healpix,
    write_healpix,
)
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
        "inmap",
        nargs="+",
        help="One or more input maps",
    )

    parser.add_argument(
        "--outmap",
        required=False,
        help="Name of output file",
    )

    parser.add_argument(
        "--outdir",
        required=False,
        help="Name of output directory",
    )

    parser.add_argument(
        "--nside_submap",
        default=16,
        help="Submap size is 12 * nside_submap ** 2.  "
        "Number of submaps is (nside / nside_submap) ** 2",
    )

    args = parser.parse_args()

    if len(args.inmap) != 1 and args.outmap is not None:
        raise RuntimeError(
            "Cannot specify output file with multiple inputs. Use --outdir instead"
        )

    def get_outfile(infile, suffix):
        if args.outmap is not None:
            outfile = args.outmap
        else:
            indir = os.path.dirname(infile)
            inroot = os.path.basename(os.path.splitext(infile)[0])
            if args.outdir is None:
                outfile = os.path.join(indir, inroot + suffix)
            else:
                outfile = os.path.join(args.outdir, inroot + suffix)
        return outfile

    for infile in args.inmap:
        if filename_is_fits(infile):
            # Convert a FITS map to HDF5

            outfile = get_outfile(infile, ".h5")
            log.info(f"Converting {infile} to {outfile}")

            mapdata, header = read_healpix(infile, None, h=True, nest=True)
            log.info_rank(f"Loaded {infile} in", timer=timer, comm=comm)

            write_healpix(
                outfile, mapdata, extra_header=header, nest=True, overwrite=True
            )
            log.info_rank(f"Wrote {outfile} in", timer=timer, comm=comm)

        elif filename_is_hdf5(infile):
            # Convert an HDF5 map to FITS

            outfile = get_outfile(infile, ".fits")
            log.info(f"Converting {infile} to {outfile}")

            mapdata, header = read_healpix(infile, h=True, nest=True)
            log.info_rank(f"Loaded {infile} in", timer=timer, comm=comm)

            write_healpix(
                outfile,
                mapdata,
                nside_submap=args.nside_submap,
                extra_header=header,
                nest=True,
                overwrite=True,
            )
            log.info_rank(f"Wrote {outfile} in", timer=timer, comm=comm)

        else:
            msg = f"Cannot guess input map file type from {args.inmap}"
            raise RuntimeError(msg)

    log.info_rank(f"Conversion done in", timer=timer0, comm=comm)

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
