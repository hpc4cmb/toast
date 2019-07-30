# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
from datetime import datetime
import os

import numpy as np

from ..mpi import get_world, Comm
from ..timing import function_timer, Timer
from ..utils import Logger, Environment


def add_dist_args(parser):
    parser.add_argument(
        "--group-size",
        required=False,
        type=np.int,
        help="Size of a process group assigned to an observation",
    )
    return


def get_comm():
    log = Logger.get()
    env = Environment.get()
    mpiworld, procs, rank = get_world()
    if rank == 0:
        env.print()
    if mpiworld is None:
        log.info("Running serially with one process at {}".format(str(datetime.now())))
    else:
        if rank == 0:
            log.info(
                "Running with {} processes at {}".format(procs, str(datetime.now()))
            )

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = Comm(world=mpiworld)
    return mpiworld, procs, rank, comm
