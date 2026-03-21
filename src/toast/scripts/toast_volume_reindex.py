#!/usr/bin/env python3

# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Script to re-index an existing volume."""

import os
import argparse

from toast.mpi import exception_guard, get_world, Comm
from toast.io import VolumeIndex


def main(opts=None):
    world, procs, rank = get_world()

    parser = argparse.ArgumentParser(
        description="This program creates a new index for a volume.",
        usage="toast_volume_reindex <options>",
    )

    parser.add_argument(
        "--volume",
        required=True,
        type=str,
        default=None,
        help="The path to the Volume",
    )

    parser.add_argument(
        "--volume_index",
        required=False,
        type=str,
        default=None,
        help="Index file to create (default is standard location inside volume)",
    )

    parser.add_argument(
        "--index_fields",
        required=False,
        type=str,
        default=None,
        help="String representation of the dictionary of extra fields to index",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        default=False,
        action="store_true",
        help="Overwrite any existing index",
    )

    args = parser.parse_args(args=opts)

    indxfields = None
    if args.index_fields is not None:
        indxfields = eval(args.index_fields)

    if args.volume_index is None:
        index_file = os.path.join(args.volume, VolumeIndex.default_name)
    else:
        index_file = os.path.join(args.volume, args.volume_index)

    if rank == 0 and os.path.isfile(index_file):
        if args.overwrite:
            os.remove(index_file)
        else:
            msg = f"Index path {index_file} exists, but overwrite is not enabled"
            raise RuntimeError(msg)
    if world is not None:
        world.barrier()

    toast_comm = Comm(world=world)
    
    vindx = VolumeIndex(index_file)

    vindx.reindex(args.volume, indexfields=indxfields, toastcomm=toast_comm)


if __name__ == "__main__":
    world, procs, rank = get_world()
    with exception_guard(comm=world):
        main()
