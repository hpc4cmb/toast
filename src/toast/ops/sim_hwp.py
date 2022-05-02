# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from astropy import units as u

from ..timing import Timer, function_timer


@function_timer
def simulate_hwp_response(
    ob,
    ob_time_key=None,
    ob_angle_key=None,
    ob_mueller_key=None,
    hwp_start=None,
    hwp_rpm=None,
    hwp_step=None,
    hwp_step_time=None,
):
    """Simulate and store the HWP angle for one observation.

    Args:
        ob (Observation):  The observation to populate.
        ob_time_key (str):  The observation shared key for timestamps.
        ob_angle_key (str):  (optional) The output observation key for the HWP angle.
        ob_mueller_key (str):  (optional) The output observation key for the full
            Mueller matrix.
        hwp_start (Quantity): The mission starting time of the HWP rotation.
        hwp_rpm (float): The HWP rotation rate in Revolutions Per Minute.
        hwp_step (Quantity): The HWP step size.
        hwp_step_time (Quantity): The time between steps.

    Returns:
        None

    """
    if ob_mueller_key is not None:
        raise NotImplementedError("Mueller matrix not yet implemented")

    if hwp_rpm is None and hwp_step is None:
        # Nothing to do!
        return

    if (hwp_rpm is not None) and (hwp_step is not None):
        raise RuntimeError("choose either continuously rotating or stepped HWP")

    if hwp_step is not None and hwp_step_time is None:
        raise RuntimeError("for a stepped HWP, you must specify the time between steps")

    hwp_start_s = hwp_start.to_value(u.second)

    # compute effective sample rate
    times = ob.shared[ob_time_key]
    dt = np.mean(times[1:-1] - times[0:-2])
    rate = 1.0 / dt

    hwp_rate = None
    hwp_step_rad = None
    hwp_step_time_s = None

    if hwp_rpm is not None:
        # convert to radians / second
        hwp_rate = hwp_rpm * 2.0 * np.pi / 60.0

    if hwp_step is not None:
        # convert to radians and seconds
        hwp_step_rad = hwp_step.to_value(u.radian)
        hwp_step_time_s = hwp_step_time.to_value(u.second)

    # Only the first process in each grid column simulates the common HWP angle

    start_sample = int(hwp_start_s * rate)
    first_sample = ob.local_index_offset
    n_sample = ob.n_local_samples

    hwp_angle = None
    hwp_mueller = None

    if ob.comm_col_rank == 0:
        if hwp_rate is not None:
            # continuous HWP
            # HWP increment per sample is:
            # (hwprate / samplerate)
            hwpincr = hwp_rate / rate
            startang = np.fmod((start_sample + first_sample) * hwpincr, 2 * np.pi)
            hwp_angle = hwpincr * np.arange(n_sample, dtype=np.float64)
            hwp_angle += startang
        elif hwp_step is not None:
            # stepped HWP
            hwp_angle = np.ones(n_sample, dtype=np.float64)
            stepsamples = int(hwp_step_time_s * rate)
            wholesteps = int((start_sample + first_sample) / stepsamples)
            remsamples = (start_sample + first_sample) - wholesteps * stepsamples
            curang = np.fmod(wholesteps * hwp_step_rad, 2 * np.pi)
            curoff = 0
            fill = remsamples
            while curoff < n_sample:
                if curoff + fill > n_sample:
                    fill = n_sample - curoff
                hwp_angle[curoff:fill] *= curang
                curang += hwp_step
                curoff += fill
                fill = stepsamples
        # Choose the HWP angle between [0, 2*pi)
        hwp_angle %= 2 * np.pi

        # Create a Mueller matrix if we will be writing that...

    # Store the angle and / or the Mueller matrix
    if ob_angle_key is not None:
        ob.shared.create_column(ob_angle_key, shape=(n_sample,), dtype=np.float64)
        ob.shared[ob_angle_key].set(hwp_angle, offset=(0,), fromrank=0)

    return
