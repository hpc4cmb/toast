#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This does some simple tests of the TOAST runtime environment.
"""

import argparse
import sys
import traceback

import toast
from toast.mpi import Comm, get_world
from toast.utils import Environment, Logger, numba_threading_layer


def main():
    env = Environment.get()
    log = Logger.get()

    mpiworld, procs, rank = get_world()
    if rank == 0:
        print(env)
        log.info("Numba threading layer set to '{}'".format(numba_threading_layer))
        n_acc, proc_per_acc, my_acc = env.get_acc()
        if n_acc <= 0:
            log.info("Accelerators unavailable")
        else:
            msg = f"{n_acc} accelerator device(s), using up to "
            msg += f"{proc_per_acc} processes per device"
            log.info(msg)
    if mpiworld is None:
        log.info("Running with one process")
    else:
        if rank == 0:
            log.info("Running with {} MPI processes".format(procs))

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
