#!/usr/bin/env python3

# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script projects observations listed in a TOAST schedule
onto a hitmap.
"""

import argparse
import os
import sys
import traceback

import astropy.units as u
import ephem
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from toast.coordinates import to_DJD, to_UTC
from toast.mpi import MPI, Comm, get_world
from toast.pixels_io_healpix import read_healpix, write_healpix
from toast.timing import Timer
from toast.utils import Environment, Logger

import toast


def parse_arguments():
    """Parse the command line arguments"""

    parser = argparse.ArgumentParser(description="Analyze observing schedule")

    parser.add_argument(
        "schedule",
        help="TOAST observing schedule",
    )

    args = parser.parse_args()
    return args


class ScanTarget:
    def __init__(self):
        self.rising_count = 0
        self.rising_time = 0
        self.transit_count = 0
        self.transit_time = 0
        self.setting_count = 0
        self.setting_time = 0

    @property
    def total_time(self):
        return self.rising_time + self.transit_time + self.setting_time

    @property
    def total_count(self):
        return self.rising_count + self.transit_count + self.setting_count


def main():
    args = parse_arguments()

    # Load the observing schedule

    schedule = toast.schedule.GroundSchedule()
    schedule.read(args.schedule)

    t_start = schedule.scans[0].start
    t_stop = schedule.scans[-1].stop
    n_tot = len(schedule.scans)

    t_tot = 0
    el_min = 90
    el_max = 0
    el_sum = 0

    scan_targets = {}

    for scan in schedule.scans:
        doy = scan.start.timetuple().tm_yday
        az_min = scan.az_min.to_value(u.deg)
        az_max = scan.az_max.to_value(u.deg)
        az_mean = 0.5 * (az_min + az_max)
        t_delta = (scan.stop - scan.start).total_seconds()
        t_tot += t_delta

        az_delta = scan.az_max - scan.az_min
        el = scan.el.to_value(u.deg)
        el_min = min(el_min, el)
        el_max = max(el_max, el)
        el_sum += el * t_delta

        if scan.name not in scan_targets:
            scan_targets[scan.name] = ScanTarget()
        if az_min < 180 and az_max < 180:
            scan_targets[scan.name].rising_count += 1
            scan_targets[scan.name].rising_time += t_delta
        elif az_min > 180 and az_max > 180:
            scan_targets[scan.name].setting_count += 1
            scan_targets[scan.name].setting_time += t_delta
        else:
            scan_targets[scan.name].transit_count += 1
            scan_targets[scan.name].transit_time += t_delta

    def neat_time(t):
        if t == 0:
            return "-"
        elif t < 1:
            return f"{t:.1f} s"
        elif t < 60:
            return f"{t:.1f} s"
        elif t < 3600:
            return f"{t / 60:.1f} min"
        elif t < 86400:
            return f"{t / 3600:.1f} h"
        else:
            return f"{t / 86400:.1f} d"

    print(
        f"{'':>30} | "
        f"{'rising':<22} | "
        f"{'setting':<22} | "
        f"{'transit':<22} | "
        f"{'total':<22}"
    )
    print(
        f"{'name':>30} | "
        f"{'count':>6} {'time':>8} {'frac':>6} | "
        f"{'count':>6} {'time':>8} {'frac':>6} | "
        f"{'count':>6} {'time':>8} {'frac':>6} | "
        f"{'count':>6} {'time':>8} {'frac':>6}"
    )
    print(31 * "-" + "+" + 24 * "-" + "+" + 24 * "-" + "+" + 24 * "-" + "+" + 24 * "-")
    for name in sorted(scan_targets):
        print(f"{name[:30]:>30}", end="")
        for direction in "rising", "setting", "transit", "total":
            totals = scan_targets[name]
            n = getattr(totals, f"{direction}_count")
            t = getattr(totals, f"{direction}_time")
            print(f" | {n:>6} {neat_time(t):>8} {t / t_tot * 100:4.1f} %", end="")
        print()

    t_delta_tot = (t_stop - t_start).total_seconds()
    print(
        f"Scheduled observing time: "
        f"{t_tot / 86400:.3f} / {t_delta_tot / 86400:.3f} "
        f"days = {t_tot / t_delta_tot:.3f} efficiency"
    )

    el_mean = el_sum / t_tot
    print(
        f"el_min = {el_min:.1f} deg, "
        f"el_max = {el_max:.1f} deg, "
        f"el_mean = {el_mean:.1f} deg"
    )

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
