#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script creates a CES schedule file that can be used as input
to toast_ground_sim.py
"""

from toast.timing import GlobalTimers
from toast.schedule import run_scheduler


def main():
    gt = GlobalTimers.get()
    gt.start("toast_ground_schedule")

    run_scheduler()

    gt.stop_all()
    gt.report()
    return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # We have an unhandled exception on at least one process.  Print a stack
        # trace for this process and then abort so that all processes terminate.
        mpiworld, procs, rank = get_world()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = ["Proc {}: {}".format(rank, x) for x in lines]
        print("".join(lines), flush=True)
        if mpiworld is not None and procs > 1:
            mpiworld.Abort(6)
        else:
            raise e
