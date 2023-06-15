#!/usr/bin/env python3

# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script co-adds noise-weighted observation matrices and
de-weights the result
"""

import argparse
import os
import sys
import traceback

import toast
from toast.ops import coadd_observation_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Co-add noise-weighted observation matrices and "
        "de-weight the result"
    )

    parser.add_argument(
        "inmatrix",
        nargs="+",
        help="One or more noise-weighted observation matrices",
    )

    parser.add_argument(
        "--outmatrix",
        required=False,
        help="Name of output file",
    )

    parser.add_argument(
        "--invcov",
        required=False,
        help="Name of output inverse covariance file",
    )

    parser.add_argument(
        "--cov",
        required=False,
        help="Name of output covariance file",
    )

    parser.add_argument(
        "--nside_submap",
        default=16,
        type=int,
        help="Submap size is 12 * nside_submap ** 2.  "
        "Number of submaps is (nside / nside_submap) ** 2",
    )

    parser.add_argument(
        "--rcond_limit",
        default=1e-3,
        type=float,
        help="Reciprocal condition number limit",
    )

    parser.add_argument(
        "--double_precision",
        default=False,
        action="store_true",
        help="Output in double precision",
    )

    args = parser.parse_args()

    coadd_observation_matrix(
        args.inmatrix,
        outmatrix=args.outmatrix,
        file_invcov=args.invcov,
        file_cov=args.cov,
        nside_submap=args.nside_submap,
        rcond_limit=args.rcond_limit,
        double_precision=args.double_precision,
    )

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
