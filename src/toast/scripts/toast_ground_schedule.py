#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script creates a CES schedule file that can be used as input
to toast_ground_sim.py
"""

import sys
import traceback

import toast
from toast.mpi import get_world
from toast.schedule_sim_ground import run_scheduler
from toast.timing import GlobalTimers


def main():
    gt = GlobalTimers.get()
    gt.start("toast_ground_schedule")

    run_scheduler()

    gt.stop_all()
    gt.report()
    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
