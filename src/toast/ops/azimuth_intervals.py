# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
from astropy import units as u
from scipy.ndimage import uniform_filter1d

from .. import qarray as qa
from .._libtoast import add_templates, bin_invcov, bin_proj, legendre
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Int, Unicode, trait_docs
from ..utils import Environment, Logger, rate_from_times
from ..vis import set_matplotlib_backend
from .flag_intervals import FlagIntervals
from .operator import Operator


@trait_docs
class AzimuthIntervals(Operator):
    """Build intervals that describe the scanning motion in azimuth.

    This operator passes through the azimuth angle and builds the list of
    intervals for standard types of scanning / turnaround motion.  Note
    that it only makes sense to use this operator for ground-based
    telescopes that primarily scan in azimuth rather than more complicated (e.g.
    lissajous) patterns.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    azimuth = Unicode(defaults.azimuth, help="Observation shared key for Azimuth")

    cut_short = Bool(True, help="If True, remove very short scanning intervals")

    scanning_interval = Unicode(
        defaults.scanning_interval, help="Interval name for scanning"
    )

    turnaround_interval = Unicode(
        defaults.turnaround_interval, help="Interval name for turnarounds"
    )

    throw_leftright_interval = Unicode(
        defaults.throw_leftright_interval,
        help="Interval name for left to right scans + turnarounds",
    )

    throw_rightleft_interval = Unicode(
        defaults.throw_rightleft_interval,
        help="Interval name for right to left scans + turnarounds",
    )

    throw_interval = Unicode(
        defaults.throw_interval, help="Interval name for scan + turnaround intervals"
    )

    scan_leftright_interval = Unicode(
        defaults.scan_leftright_interval, help="Interval name for left to right scans"
    )

    turn_leftright_interval = Unicode(
        defaults.turn_leftright_interval,
        help="Interval name for turnarounds after left to right scans",
    )

    scan_rightleft_interval = Unicode(
        defaults.scan_rightleft_interval, help="Interval name for right to left scans"
    )

    turn_rightleft_interval = Unicode(
        defaults.turn_rightleft_interval,
        help="Interval name for turnarounds after right to left scans",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for bad azimuth pointing",
    )

    window_seconds = Float(0.25, help="Smoothing window in seconds")

    debug_root = Unicode(
        None,
        allow_none=True,
        help="If not None, dump debug plots to this root file name",
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        for obs in data.obs:
            # For now, we just have the first process row do the calculation.  It
            # is relatively fast.

            throw_times = None
            throw_leftright_times = None
            throw_rightleft_times = None
            stable_times = None
            stable_leftright_times = None
            stable_rightleft_times = None

            # Sample rate
            stamps = obs.shared[self.times].data
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(stamps)

            if obs.comm_col_rank == 0:
                # Smoothing window in samples
                window = int(rate * self.window_seconds)

                # The scan velocity
                scan_vel = np.gradient(obs.shared[self.azimuth].data)

                # Smooth with moving window
                wscan_vel = uniform_filter1d(scan_vel, size=window, mode="nearest")

                # When the velocity changes sign, we have a turnaround
                vel_switch = np.where(wscan_vel[:-1] * wscan_vel[1:] < 0)[0] + 1
                throw_times = [
                    (stamps[x[0]], stamps[x[1]])
                    for x in zip(vel_switch[:-1], vel_switch[1:])
                ]

                # The peak to peak range of the scan velocity
                vel_range = np.amax(wscan_vel) - np.amin(wscan_vel)

                # The smoothed scan acceleration
                scan_accel = uniform_filter1d(
                    np.gradient(wscan_vel),
                    size=window,
                    mode="nearest",
                )

                accel_range = np.amax(scan_accel) - np.amin(scan_accel)

                # When the acceleration is zero to some tolerance, we are
                # scanning.
                stable = (np.absolute(scan_accel) < 0.1 * accel_range) * np.ones(
                    len(scan_accel), dtype=np.int8
                )
                begin_stable = np.where(stable[1:] - stable[:-1] == 1)[0]
                end_stable = np.where(stable[:-1] - stable[1:] == 1)[0]
                if begin_stable[0] > end_stable[0]:
                    # We start in the middle of a scan
                    begin_stable = np.concatenate(([0], begin_stable))
                if begin_stable[-1] > end_stable[-1]:
                    # We end in the middle of a scan
                    end_stable = np.concatenate((end_stable, [obs.n_local_samples]))
                stable_times = [
                    (stamps[x[0]], stamps[x[1] - 1])
                    for x in zip(begin_stable, end_stable)
                ]

                # In some situations there are very short stable scans detected at the
                # beginning and end of observations.  Here we cut any short throw and
                # stable periods.
                if self.cut_short:
                    stable_spans = np.array([(x[1] - x[0]) for x in stable_times])
                    throw_spans = np.array([(x[1] - x[0]) for x in throw_times])
                    mean_stable = np.mean(stable_spans)
                    std_stable = np.std(stable_spans)
                    mean_throw = np.mean(throw_spans)
                    std_throw = np.std(throw_spans)
                    stable_bad = stable_spans < mean_stable - 5 * std_stable
                    begin_stable = np.array(
                        [x for (x, y) in zip(begin_stable, stable_bad) if not y]
                    )
                    end_stable = np.array(
                        [x for (x, y) in zip(end_stable, stable_bad) if not y]
                    )
                    stable_times = [
                        x for (x, y) in zip(stable_times, stable_bad) if not y
                    ]
                    throw_bad = throw_spans < mean_throw - 5 * std_throw
                    throw_times = [x for (x, y) in zip(throw_times, throw_bad) if not y]

                # Split scans into left and right-going intervals
                stable_leftright_times = []
                stable_rightleft_times = []
                for start, stop in stable_times:
                    # Check the velocity at the middle of the scan
                    i = np.argwhere(np.abs(stamps - 0.5 * (start + stop)))[0]
                    if wscan_vel[i] > 0:
                        stable_leftright_times.append((start, stop))
                    elif wscan_vel[i] < 0:
                        stable_rightleft_times.append((start, stop))
                throw_leftright_times = []
                throw_rightleft_times = []
                for start, stop in throw_times:
                    # Check the velocity at the middle of the throw
                    i = np.argwhere(np.abs(stamps - 0.5 * (start + stop)))[0]
                    if wscan_vel[i] > 0:
                        throw_leftright_times.append((start, stop))
                    elif wscan_vel[i] < 0:
                        throw_rightleft_times.append((start, stop))

                if self.debug_root is not None:
                    set_matplotlib_backend()

                    import matplotlib.pyplot as plt

                    # Dump some plots
                    out_file = f"{self.debug_root}_{obs.comm_row_rank}.pdf"
                    if len(vel_switch) >= 5:
                        # Plot a few scans
                        n_plot = vel_switch[4]
                    else:
                        # Plot it all
                        n_plot = obs.n_local_samples

                    swplot = vel_switch[vel_switch <= n_plot]
                    bstable = begin_stable[begin_stable <= n_plot]
                    estable = end_stable[end_stable <= n_plot]

                    fig = plt.figure(dpi=100, figsize=(8, 16))

                    ax = fig.add_subplot(4, 1, 1)
                    ax.plot(
                        np.arange(n_plot),
                        obs.shared[self.azimuth].data[:n_plot],
                        "-",
                    )

                    ax = fig.add_subplot(4, 1, 2)
                    ax.plot(np.arange(n_plot), stable[:n_plot], "-")
                    ax.vlines(bstable, ymin=-1, ymax=2, color="green")
                    ax.vlines(estable, ymin=-1, ymax=2, color="red")

                    ax = fig.add_subplot(4, 1, 3)
                    ax.plot(np.arange(n_plot), scan_vel[:n_plot], "-")
                    ax.plot(np.arange(n_plot), wscan_vel[:n_plot], "-")
                    ax.vlines(
                        swplot,
                        ymin=np.amin(scan_vel),
                        ymax=np.amax(scan_vel),
                    )

                    ax = fig.add_subplot(4, 1, 4)
                    ax.plot(np.arange(n_plot), scan_accel[:n_plot], "-")

                    plt.savefig(out_file)
                    plt.close()

            # Now create the intervals across each process column

            # The throw intervals are between turnarounds
            obs.intervals.create_col(
                self.throw_interval, throw_times, stamps, fromrank=0
            )
            obs.intervals.create_col(
                self.throw_leftright_interval, throw_leftright_times, stamps, fromrank=0
            )
            obs.intervals.create_col(
                self.throw_rightleft_interval, throw_rightleft_times, stamps, fromrank=0
            )

            # Stable scanning intervals
            obs.intervals.create_col(
                self.scanning_interval, stable_times, stamps, fromrank=0
            )
            obs.intervals.create_col(
                self.scan_leftright_interval, stable_leftright_times, stamps, fromrank=0
            )
            obs.intervals.create_col(
                self.scan_rightleft_interval, stable_rightleft_times, stamps, fromrank=0
            )

            # Turnarounds are the inverse of stable scanning
            obs.intervals[self.turnaround_interval] = ~obs.intervals[
                self.scanning_interval
            ]

        # Additionally flag turnarounds as unstable pointing
        flag_intervals = FlagIntervals(
            shared_flags=self.shared_flags,
            shared_flag_bytes=1,
            view_mask=[
                (self.turnaround_interval, defaults.shared_mask_unstable_scanrate),
            ],
        )
        flag_intervals.apply(data, detectors=None)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [self.times, self.azimuth],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        return req

    def _provides(self):
        return {
            "intervals": [
                self.scanning_interval,
                self.turnaround_interval,
                self.scan_leftright_interval,
                self.scan_rightleft_interval,
                self.turn_leftright_interval,
                self.turn_rightleft_interval,
                self.throw_interval,
                self.throw_leftright_interval,
                self.throw_rightleft_interval,
            ]
        }
