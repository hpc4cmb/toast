# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..timing import function_timer, Timer


@function_timer
def simulate_hwp_angle(
    obs, obs_key, hwp_start_s, hwp_rpm, hwp_step_deg, hwp_step_time_m
):
    """Simulate and store the HWP angle for one observation.

    Args:
        obs (Observation): The observation to populate.
        obs_key (str): The observation key for the HWP angle.
        hwp_start_s (float): The mission starting time in seconds of the HWP rotation.
        hwp_rpm (float): The HWP rotation rate in Revolutions Per Minute.
        hwp_step_deg (float): The HWP step size in degrees.
        hwp_step_time_m (float): The time in minutes between steps.

    Returns:
        None

    """
    if hwp_rpm is None and hwp_step_deg is None:
        # Nothing to do!
        return

    if (hwp_rpm is not None) and (hwp_step_deg is not None):
        raise RuntimeError("choose either continuously rotating or stepped HWP")

    if hwp_step_deg is not None and hwp_step_time_m is None:
        raise RuntimeError("for a stepped HWP, you must specify the time between steps")

    # compute effective sample rate
    times = obs.times
    dt = np.mean(times[1:-1] - times[0:-2])
    rate = 1.0 / dt

    hwp_rate = None
    hwp_step = None
    hwp_step_time = None

    if hwp_rpm is not None:
        # convert to radians / second
        hwp_rate = hwp_rpm * 2.0 * np.pi / 60.0

    if hwp_step_deg is not None:
        # convert to radians and seconds
        hwp_step = hwp_step_deg * np.pi / 180.0
        hwp_step_time = hwp_step_time_m * 60.0

    first_samp, n_samp = obs.local_samples

    obs.shared.create(
        obs_key, shape=(n_samp,), dtype=np.float64, comm=obs.grid_comm_col
    )

    # Only the first process in each grid column simulates the common HWP angle

    start_sample = int(hwp_start_s * rate)
    hwp_angle = None

    if obs.grid_comm_col is None or obs.grid_comm_col.rank == 0:
        if hwp_rate is not None:
            # continuous HWP
            # HWP increment per sample is:
            # (hwprate / samplerate)
            hwpincr = hwp_rate / rate
            startang = np.fmod((start_sample + first_samp) * hwpincr, 2 * np.pi)
            hwp_angle = hwpincr * np.arange(n_samp, dtype=np.float64)
            hwp_angle += startang
        elif hwp_step is not None:
            # stepped HWP
            hwp_angle = np.ones(n_samp, dtype=np.float64)
            stepsamples = int(hwp_step_time * rate)
            wholesteps = int((start_sample + first_samp) / stepsamples)
            remsamples = (start_sample + first_samp) - wholesteps * stepsamples
            curang = np.fmod(wholesteps * hwp_step, 2 * np.pi)
            curoff = 0
            fill = remsamples
            while curoff < n_samp:
                if curoff + fill > n_samp:
                    fill = n_samp - curoff
                hwp_angle[curoff:fill] *= curang
                curang += hwp_step
                curoff += fill
                fill = stepsamples
        if hwp_angle is not None:
            # Choose the HWP angle between [0, 2*pi)
            hwp_angle %= 2 * np.pi

    obs.shared[obs_key].set(hwp_angle, offset=(0,), fromrank=0)

    return
