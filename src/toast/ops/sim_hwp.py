# Copyright (c) 2015-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
from astropy import units as u

from .. import rng
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Int, Quantity, Unicode, trait_docs
from ..utils import GlobalTimers, Logger, Timer, dtype_to_aligned, name_UID
from .operator import Operator


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


@trait_docs
class PerturbHWP(Operator):
    """Operator that adds irregularities to HWP rotation"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    hwp_angle = Unicode(
        defaults.hwp_angle,
        allow_none=True,
        help="Observation shared key for HWP angle",
    )

    drift_sigma = Quantity(
        None,
        allow_none=True,
        help="1-sigma relative change in spin rate, such as 0.01 / hour",
    )

    time_sigma = Quantity(
        None,
        allow_none=True,
        help="1-sigma difference between real and nominal time stamps",
    )

    realization = Int(0, allow_none=False, help="Realization index")

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        t0 = time()
        log = Logger.get()

        for trait in ("times", "hwp_angle"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        for iobs, obs in enumerate(data.obs):
            offset = obs.local_index_offset
            nlocal = obs.n_local_samples
            ntotal = obs.n_all_samples

            # Get an RNG seed
            key1 = self.realization * 1543343 + obs.telescope.uid
            key2 = obs.session.uid
            counter1 = 0

            # The times and hwp_angle are shared among columns of the process
            # grid.  Only the first process row needs to modify the data.
            if (
                obs.shared.comm_type(self.times) != "column"
                or obs.shared.comm_type(self.hwp_angle) != "column"
            ):
                msg = f"obs {obs.name}: expected shared fields {self.times} and "
                msg += f"{self.hwp_angle} to be on the column communicator."
                raise RuntimeError(msg)

            if obs.comm_col_rank == 0:
                times = obs.shared[self.times].data
                hwp_angle = obs.shared[self.hwp_angle].data

                # We are in the first process row.  In our RNG generation,
                # "counter2" corresponds to the sample index.  If there are
                # multiple processes in the grid row, start our RNG stream
                # at the first sample on this process.
                counter2 = obs.local_index_offset

                time_delta = times[-1] - times[0]

                # Simulate timing error (jitter)
                if self.time_sigma is None:
                    time_error = 0
                else:
                    component = 0
                    rngdata = rng.random(
                        times.size,
                        sampler="gaussian",
                        key=(key1, key2 + component),
                        counter=(counter1, counter2),
                    )
                    time_error = np.array(rngdata) * self.time_sigma.to_value(u.s)
                new_times = times + time_error
                if np.any(np.diff(new_times) <= 0):
                    raise RuntimeError("Simulated timing error causes time travel")

                # Simulate rate drift
                nominal_rate = (hwp_angle[-1] - hwp_angle[0]) / time_delta
                if self.drift_sigma is None:
                    begin_rate = nominal_rate
                    accel = 0
                else:
                    # This random number is for the uniform drift across the whole
                    # observation.  All processes along the row of the grid should
                    # use the same value here.
                    counter2 = 0
                    component = 1
                    rngdata = rng.random(
                        1,
                        sampler="gaussian",
                        key=(key1, key2 + component),
                        counter=(counter1, counter2),
                    )
                    sigma = self.drift_sigma.to_value(1 / u.s) * time_delta
                    drift = rngdata[0] * sigma
                    begin_rate = nominal_rate * (1 - drift)
                    end_rate = nominal_rate * (1 + drift)
                    accel = (end_rate - begin_rate) / time_delta

                # Now calculcate the HWP angle subject to jitter and drift
                t = new_times - new_times[0]
                new_angle = 0.5 * accel * t**2 + begin_rate * t + hwp_angle[0]
            else:
                new_angle = None

            # Set the new HWP angle values
            obs.shared[self.hwp_angle].set(new_angle, offset=(0,), fromrank=0)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return {
            "shared": [
                self.times,
                self.hwp_angle,
            ]
        }

    def _provides(self):
        return dict()
