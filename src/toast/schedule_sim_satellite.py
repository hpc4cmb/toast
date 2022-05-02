#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import datetime

import numpy as np
from astropy import units as u

from .schedule import SatelliteScan, SatelliteSchedule
from .utils import Logger


def create_satellite_schedule(
    prefix="",
    mission_start=None,
    observation_time=10 * u.minute,
    gap_time=0 * u.minute,
    num_observations=1,
    prec_period=10 * u.minute,
    spin_period=2 * u.minute,
    site_name="space",
    telescope_name="satellite",
):
    """Generate a satellite observing schedule.

    This creates a series of scans with identical lengths and rotation rates, as well
    as optional gaps between.

    Args:
        prefix (str):  The prefix for the name of each scan.
        mission_start (datetime):  The overall start time of the schedule.
        observation_time (Quantity):  The length of each observation.
        gap_time (Quantity):  The time between observations.
        num_observations (int):  The number of observations.
        prec_period (Quantity):  The time for one revolution about the precession axis.
        spin_period (Quantity):  The time for one revolution about the spin axis.
        site_name (str):  The name of the site to include in the schedule.
        telescope_name (str):  The name of the telescope to include in the schedule.

    Returns:
        (SatelliteSchedule):  The resulting schedule.

    """
    log = Logger.get()
    if mission_start is None:
        raise RuntimeError("You must specify the mission start")

    if mission_start.tzinfo is None:
        msg = f"Mission start time '{mission_start}' is not timezone-aware.  Assuming UTC."
        log.warning(msg)
        mission_start = mission_start.replace(tzinfo=datetime.timezone.utc)

    obs = datetime.timedelta(seconds=observation_time.to_value(u.second))
    gap = datetime.timedelta(seconds=gap_time.to_value(u.second))
    epsilon = datetime.timedelta(seconds=0)
    if gap_time.to_value(u.second) == 0:
        # If there is no gap, we add a tiny break (much less than one sample for any
        # reasonable experiment) so that the start time of one observation is never
        # identical to the stop time of the previous one.
        epsilon = datetime.timedelta(microseconds=2)

    total = obs + gap

    scans = list()
    for sc in range(num_observations):
        start = sc * total + mission_start
        stop = start + obs - epsilon
        name = "{}{:06d}_{}".format(prefix, sc, start.isoformat(timespec="minutes"))
        scans.append(
            SatelliteScan(
                name=name,
                start=start,
                stop=stop,
                prec_period=prec_period,
                spin_period=spin_period,
            )
        )
    return SatelliteSchedule(
        scans=scans, site_name=site_name, telescope_name=telescope_name
    )
