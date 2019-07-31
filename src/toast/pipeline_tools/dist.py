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


@function_timer
def get_time_communicators(comm, data):
    """ Split the world communicator by time.

    """
    time_comms = [("all", comm.comm_world)]
    if comm.comm_world is None:
        return time_comms

    # A process will only have data for one season and one day.  If more
    # than one season is observed, we split the communicator to make
    # season maps.

    my_season = data.obs[0]["season"]
    seasons = np.array(comm.comm_world.allgather(my_season))
    do_seasons = np.any(seasons != my_season)
    if do_seasons:
        season_comm = comm.comm_world.Split(my_season, comm.world_rank)
        time_comms.append((str(my_season), season_comm))

    # Split the communicator to make daily maps.  We could easily split
    # by month as well

    my_day = int(data.obs[0]["MJD"])
    my_date = data.obs[0]["date"]
    days = np.array(comm.comm_world.allgather(my_day))
    do_days = np.any(days != my_day)
    if do_days:
        day_comm = comm.comm_world.Split(my_day, comm.world_rank)
        time_comms.append((my_date, day_comm))

    return time_comms
