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
from ..traits import Bool, Float, Int, Quantity, Unicode, trait_docs
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

    cut_long = Bool(True, help="If True, remove very long scanning intervals")

    short_limit = Quantity(
        0.25 * u.dimensionless_unscaled,
        help="Minimum length of a scan.  Either the minimum length in time or a "
        "fraction of median scan length",
    )

    long_limit = Quantity(
        1.25 * u.dimensionless_unscaled,
        help="Maximum length of a scan.  Either the maximum length in time or a "
        "fraction of median scan length",
    )

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

    window_seconds = Float(0.5, help="Smoothing window in seconds")

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
            have_scanning = True

            # Sample rate
            stamps = obs.shared[self.times].data
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(stamps)

            if obs.comm_col_rank == 0:
                # Smoothing window in samples
                window = int(rate * self.window_seconds)

                # The azimuth, with flagged samples interpolated
                azimuth = np.array(obs.shared[self.azimuth].data)
                self._interpolate_azimuth(obs, azimuth)

                # The scan velocity
                scan_vel = np.gradient(azimuth)

                # Smooth with moving window
                wscan_vel = uniform_filter1d(scan_vel, size=window, mode="nearest")

                # The peak to peak range of the scan velocity
                vel_range = np.amax(wscan_vel) - np.amin(wscan_vel)

                # The smoothed scan acceleration
                scan_accel = uniform_filter1d(
                    np.gradient(wscan_vel),
                    size=window,
                    mode="nearest",
                )

                # Peak to peak acceleration range
                accel_range = np.amax(scan_accel) - np.amin(scan_accel)

                # When the acceleration is zero to some tolerance, we are
                # scanning.  However, we also need to only consider times where
                # the velocity is non-zero.
                stable = (np.absolute(scan_accel) < 0.1 * accel_range) * np.ones(
                    len(scan_accel), dtype=np.int8
                )
                stable *= np.absolute(wscan_vel) > 0.1 * vel_range

                # The first estimate of the samples where stable pointing
                # begins and ends.
                begin_stable = np.where(stable[1:] - stable[:-1] == 1)[0]
                end_stable = np.where(stable[:-1] - stable[1:] == 1)[0]

                if len(begin_stable) == 0 or len(end_stable) == 0:
                    msg = f"Observation {obs.name} has no stable scanning"
                    msg += f" periods.  You should cut this observation or"
                    msg += f" change the filter window.  Flagging all samples"
                    msg += f" as unstable pointing."
                    log.warning(msg)
                    have_scanning = False

                if have_scanning:
                    # Refine our list of stable periods
                    if begin_stable[0] > end_stable[0]:
                        # We start in the middle of a scan
                        begin_stable = np.concatenate(([0], begin_stable))
                    if begin_stable[-1] > end_stable[-1]:
                        # We end in the middle of a scan
                        end_stable = np.concatenate((end_stable, [obs.n_local_samples]))

                    # In some situations there are very short stable scans detected at
                    # the beginning and end of observations.  Here we cut any short
                    # throw and stable periods.
                    cut_threshold = 4
                    if (self.cut_short or self.cut_long) and (
                        len(begin_stable) >= cut_threshold
                    ):
                        if self.cut_short:
                            stable_timespans = np.array(
                                [stamps[y-1] - stamps[x] for x, y in zip(begin_stable, end_stable)]
                            )
                            try:
                                # First try short limit as time
                                stable_bad = stable_timespans < self.short_limit.to_value(
                                    u.s
                                )
                            except:
                                # Try short limit as fraction
                                median_stable = np.median(stable_timespans)
                                stable_bad = (
                                    stable_timespans < self.short_limit * median_stable
                                )
                            begin_stable = np.array(
                                [x for (x, y) in zip(begin_stable, stable_bad) if not y]
                            )
                            end_stable = np.array(
                                [x for (x, y) in zip(end_stable, stable_bad) if not y]
                            )
                        if self.cut_long:
                            stable_timespans = np.array(
                                [stamps[y-1] - stamps[x] for x, y in zip(begin_stable, end_stable)]
                            )
                            try:
                                # First try long limit as time
                                stable_bad = stable_timespans > self.long_limit.to_value(
                                    u.s
                                )
                            except:
                                # Try long limit as fraction
                                median_stable = np.median(stable_timespans)
                                stable_bad = (
                                    stable_timespans > self.long_limit * median_stable
                                )
                            begin_stable = np.array(
                                [x for (x, y) in zip(begin_stable, stable_bad) if not y]
                            )
                            end_stable = np.array(
                                [x for (x, y) in zip(end_stable, stable_bad) if not y]
                            )
                    if len(begin_stable) == 0:
                        have_scanning = False

                # The "throw" intervals extend from one turnaround to the next.
                # We start the first throw at the beginning of the first stable scan
                # and then find the sample between stable scans where the turnaround
                # happens.  This reduces false detections of turnarounds before or
                # after the stable scanning within the observation.
                #
                # If no turnaround is found between stable scans, we log a warning
                # and choose the sample midway between stable scans to be the throw
                # boundary.
                if have_scanning:
                    begin_throw = [begin_stable[0]]
                    end_throw = list()
                    for start_turn, end_turn in zip(end_stable[:-1], begin_stable[1:]):
                        vel_switch = np.where(
                            wscan_vel[start_turn : end_turn - 1]
                            * wscan_vel[start_turn + 1 : end_turn]
                            < 0
                        )[0]
                        if len(vel_switch) != 1:
                            msg = f"{obs.name}: Single turnaround not found between"
                            msg += " end of stable scan at"
                            msg += f" sample {start_turn} and next start at"
                            msg += f" {end_turn}. Selecting midpoint as turnaround."
                            log.warning(msg)
                            half_gap = (end_turn - start_turn) // 2
                            end_throw.append(start_turn + half_gap)
                        else:
                            end_throw.append(start_turn + vel_switch[0])
                        begin_throw.append(end_throw[-1] + 1)
                    end_throw.append(end_stable[-1])
                    begin_throw = np.array(begin_throw)
                    end_throw = np.array(end_throw)

                    stable_times = [
                        (stamps[x[0]], stamps[x[1] - 1])
                        for x in zip(begin_stable, end_stable)
                    ]
                    throw_times = [
                        (stamps[x[0]], stamps[x[1] - 1])
                        for x in zip(begin_throw, end_throw)
                    ]

                    throw_leftright_times = list()
                    throw_rightleft_times = list()
                    stable_leftright_times = list()
                    stable_rightleft_times = list()

                    # Split scans into left and right-going intervals
                    for iscan, (first, last) in enumerate(
                        zip(begin_stable, end_stable)
                    ):
                        # Check the velocity at the middle of the scan
                        mid = first + (last - first) // 2
                        if wscan_vel[mid] >= 0:
                            stable_leftright_times.append(stable_times[iscan])
                            throw_leftright_times.append(throw_times[iscan])
                        else:
                            stable_rightleft_times.append(stable_times[iscan])
                            throw_rightleft_times.append(throw_times[iscan])

                if self.debug_root is not None:
                    set_matplotlib_backend()

                    import matplotlib.pyplot as plt

                    # Dump some plots
                    out_file = f"{self.debug_root}_{obs.name}_{obs.comm_row_rank}.pdf"
                    if have_scanning:
                        if len(end_throw) >= 10:
                            # Plot a few scans
                            n_plot = end_throw[10]
                        else:
                            # Plot it all
                            n_plot = obs.n_local_samples

                        swplot = vel_switch[vel_switch <= n_plot]
                        bstable = begin_stable[begin_stable <= n_plot]
                        estable = end_stable[end_stable <= n_plot]
                        bthrow = begin_throw[begin_throw <= n_plot]
                        ethrow = end_throw[end_throw <= n_plot]

                        fig = plt.figure(dpi=100, figsize=(8, 16))

                        ax = fig.add_subplot(4, 1, 1)
                        ax.plot(
                            np.arange(n_plot),
                            azimuth[:n_plot],
                            "-",
                        )
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Azimuth")

                        ax = fig.add_subplot(4, 1, 2)
                        ax.plot(np.arange(n_plot), stable[:n_plot], "-")
                        ax.vlines(bstable, ymin=-1, ymax=2, color="green")
                        ax.vlines(estable, ymin=-1, ymax=2, color="red")
                        ax.vlines(bthrow, ymin=-2, ymax=1, color="cyan")
                        ax.vlines(ethrow, ymin=-2, ymax=1, color="purple")
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Stable Scan / Throw")

                        ax = fig.add_subplot(4, 1, 3)
                        ax.plot(np.arange(n_plot), scan_vel[:n_plot], "-")
                        ax.plot(np.arange(n_plot), wscan_vel[:n_plot], "-")
                        ax.vlines(
                            swplot,
                            ymin=np.amin(scan_vel),
                            ymax=np.amax(scan_vel),
                        )
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Scan Velocity")

                        ax = fig.add_subplot(4, 1, 4)
                        ax.plot(np.arange(n_plot), scan_accel[:n_plot], "-")
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Scan Acceleration")
                    else:
                        n_plot = obs.n_local_samples
                        fig = plt.figure(dpi=100, figsize=(8, 12))

                        ax = fig.add_subplot(3, 1, 1)
                        ax.plot(
                            np.arange(n_plot),
                            azimuth[:n_plot],
                            "-",
                        )
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Azimuth")

                        ax = fig.add_subplot(3, 1, 2)
                        ax.plot(np.arange(n_plot), scan_vel[:n_plot], "-")
                        ax.plot(np.arange(n_plot), wscan_vel[:n_plot], "-")
                        ax.vlines(
                            swplot,
                            ymin=np.amin(scan_vel),
                            ymax=np.amax(scan_vel),
                        )
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Scan Velocity")

                        ax = fig.add_subplot(3, 1, 3)
                        ax.plot(np.arange(n_plot), scan_accel[:n_plot], "-")
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Scan Acceleration")
                    plt.savefig(out_file)
                    plt.close()

            # Now create the intervals across each process column
            if obs.comm_col is not None:
                have_scanning = obs.comm_col.bcast(have_scanning, root=0)

            if have_scanning:
                # The throw intervals are between turnarounds
                obs.intervals.create_col(
                    self.throw_interval, throw_times, stamps, fromrank=0
                )
                obs.intervals.create_col(
                    self.throw_leftright_interval,
                    throw_leftright_times,
                    stamps,
                    fromrank=0,
                )
                obs.intervals.create_col(
                    self.throw_rightleft_interval,
                    throw_rightleft_times,
                    stamps,
                    fromrank=0,
                )

                # Stable scanning intervals
                obs.intervals.create_col(
                    self.scanning_interval, stable_times, stamps, fromrank=0
                )
                obs.intervals.create_col(
                    self.scan_leftright_interval,
                    stable_leftright_times,
                    stamps,
                    fromrank=0,
                )
                obs.intervals.create_col(
                    self.scan_rightleft_interval,
                    stable_rightleft_times,
                    stamps,
                    fromrank=0,
                )

                # Turnarounds are the inverse of stable scanning
                obs.intervals[self.turnaround_interval] = ~obs.intervals[
                    self.scanning_interval
                ]
            else:
                # Flag all samples as unstable
                if self.shared_flags not in obs.shared:
                    obs.shared.create_column(
                        self.shared_flags,
                        shape=(obs.n_local_samples,),
                        dtype=np.uint8,
                    )
                if obs.comm_col_rank == 0:
                    obs.shared[self.shared_flags].set(
                        np.zeros_like(obs.shared[self.shared_flags].data),
                        offset=(0,),
                        fromrank=0,
                    )
                else:
                    obs.shared[self.shared_flags].set(None, offset=(0,), fromrank=0)

        # Additionally flag turnarounds as unstable pointing
        flag_intervals = FlagIntervals(
            shared_flags=self.shared_flags,
            shared_flag_bytes=1,
            view_mask=[
                (self.turnaround_interval, defaults.shared_mask_unstable_scanrate),
            ],
        )
        flag_intervals.apply(data, detectors=None)

    def _interpolate_azimuth(self, obs, az):
        flags = np.array(obs.shared[self.shared_flags].data)
        flags &= self.shared_flag_mask
        flag_indx = np.arange(len(flags), dtype=np.int64)[np.nonzero(flags)]
        flag_groups = np.split(flag_indx, np.where(np.diff(flag_indx) != 1)[0] + 1)
        for grp in flag_groups:
            if len(grp) == 0:
                continue
            bad_first = grp[0]
            bad_last = grp[-1]
            if bad_first == 0:
                # Starting bad samples
                az[: bad_last + 1] = az[bad_last + 1]
            elif bad_last == len(flags) - 1:
                # Ending bad samples
                az[bad_first:] = az[bad_first - 1]
            else:
                int_first = bad_first - 1
                int_last = bad_last + 1
                az[bad_first : bad_last + 1] = np.interp(
                    np.arange(bad_first, bad_last + 1, 1, dtype=np.int32),
                    [int_first, int_last],
                    [az[int_first], az[int_last]],
                )

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
