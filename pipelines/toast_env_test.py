#!/usr/bin/env python3

# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This does some simple tests of the TOAST runtime environment.
"""

import sys
import argparse
import traceback

from toast.mpi import get_world, Comm

from toast.utils import Logger, Environment


def main():
    env = Environment.get()
    log = Logger.get()

    parser = argparse.ArgumentParser(
        description="Test the TOAST runtime environment.", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--groupsize",
        required=False,
        type=int,
        default=0,
        help="size of processor groups used to distribute observations",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        return

    mpiworld, procs, rank = get_world()
    if rank == 0:
        env.print()
    if mpiworld is None:
        log.info("Running serially with one process")
    else:
        if rank == 0:
            log.info("Running with {} processes".format(procs))

    groupsize = args.groupsize
    if groupsize <= 0:
        groupsize = procs

    if rank == 0:
        log.info("Using group size of {} processes".format(groupsize))

    comm = Comm(world=mpiworld, groupsize=groupsize)

    log.info(
        "Process {}:  world rank {}, group {} of {}, group rank {}".format(
            rank, comm.world_rank, comm.group + 1, comm.ngroups, comm.group_rank
        )
    )

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
