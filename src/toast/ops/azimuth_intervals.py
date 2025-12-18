# Copyright (c) 2023-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
from astropy import units as u
from scipy.ndimage import uniform_filter1d

from .. import qarray as qa
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, flagged_noise_fill, rate_from_times
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

            # Smoothing window in samples
            window = int(rate * self.window_seconds)

            if obs.comm_col_rank == 0:
                # The azimuth angle
                azimuth = np.array(obs.shared[self.azimuth].data)

                # The azimuth flags
                flags = np.array(obs.shared[self.shared_flags].data)
                flags &= self.shared_flag_mask

                # Scan velocity
                scan_vel = self._gradient(azimuth, window, flags=flags)

                # The peak to peak range of the scan velocity
                vel_range = np.amax(scan_vel) - np.amin(scan_vel)

                # Scan acceleration
                scan_accel = self._gradient(scan_vel, window)

                # Peak to peak acceleration range
                accel_range = np.amax(scan_accel) - np.amin(scan_accel)

                # When the acceleration is zero to some tolerance, we are
                # scanning.  However, we also need to only consider times where
                # the velocity is non-zero.
                stable = (np.absolute(scan_accel) < 0.1 * accel_range) * np.ones(
                    len(scan_accel), dtype=np.int8
                )
                stable *= np.absolute(scan_vel) > 0.1 * vel_range

                # The first estimate of the samples where stable pointing
                # begins and ends.
                begin_stable = np.where(stable[1:] - stable[:-1] == 1)[0]
                end_stable = np.where(stable[:-1] - stable[1:] == 1)[0]

                if len(begin_stable) == 0 or len(end_stable) == 0:
                    msg = f"Observation {obs.name} has no stable scanning"
                    msg += " periods.  You should cut this observation or"
                    msg += " change the filter window.  Flagging all samples"
                    msg += " as unstable pointing."
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
                                [
                                    stamps[y - 1] - stamps[x]
                                    for x, y in zip(begin_stable, end_stable)
                                ]
                            )
                            try:
                                # First try short limit as time
                                stable_bad = (
                                    stable_timespans < self.short_limit.to_value(u.s)
                                )
                            except Exception:
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
                                [
                                    stamps[y - 1] - stamps[x]
                                    for x, y in zip(begin_stable, end_stable)
                                ]
                            )
                            try:
                                # First try long limit as time
                                stable_bad = (
                                    stable_timespans > self.long_limit.to_value(u.s)
                                )
                            except Exception:
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
                    vel_switch = list()
                    for start_turn, end_turn in zip(end_stable[:-1], begin_stable[1:]):
                        # Fit a quadratic polynomial and find the velocity change sample
                        vel_turn = self._find_turnaround(scan_vel[start_turn:end_turn])
                        if vel_turn is None:
                            msg = f"{obs.name}: Turnaround not found between"
                            msg += " end of stable scan at"
                            msg += f" sample {start_turn} and next start at"
                            msg += f" {end_turn}. Selecting midpoint as turnaround."
                            log.warning(msg)
                            half_gap = (end_turn - start_turn) // 2
                            end_throw.append(start_turn + half_gap)
                        else:
                            end_throw.append(start_turn + vel_turn)
                        vel_switch.append(end_throw[-1])
                        begin_throw.append(end_throw[-1] + 1)
                    end_throw.append(end_stable[-1])
                    begin_throw = np.array(begin_throw)
                    end_throw = np.array(end_throw)
                    vel_switch = np.array(vel_switch)

                    stable_times = [
                        (stamps[x[0]], stamps[x[1]])
                        for x in zip(begin_stable, end_stable)
                    ]
                    throw_times = [
                        (stamps[x[0]], stamps[x[1]])
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
                        if scan_vel[mid] >= 0:
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
                        if len(end_throw) >= 5:
                            # Plot a few scans
                            plot_start = 0
                            n_plot = end_throw[4]
                        else:
                            # Plot it all
                            plot_start = 0
                            n_plot = obs.n_local_samples
                        pslc = slice(plot_start, plot_start + n_plot, 1)
                        px = np.arange(plot_start, plot_start + n_plot, 1)

                        swplot = vel_switch[
                            np.logical_and(
                                vel_switch <= plot_start + n_plot,
                                vel_switch >= plot_start,
                            )
                        ]
                        bstable = begin_stable[
                            np.logical_and(
                                begin_stable <= plot_start + n_plot,
                                begin_stable >= plot_start,
                            )
                        ]
                        estable = end_stable[
                            np.logical_and(
                                end_stable <= plot_start + n_plot,
                                end_stable >= plot_start,
                            )
                        ]
                        bthrow = begin_throw[
                            np.logical_and(
                                begin_throw <= plot_start + n_plot,
                                begin_throw >= plot_start,
                            )
                        ]
                        ethrow = end_throw[
                            np.logical_and(
                                end_throw <= plot_start + n_plot,
                                end_throw >= plot_start,
                            )
                        ]

                        fig = plt.figure(dpi=100, figsize=(8, 16))

                        ax = fig.add_subplot(4, 1, 1)
                        ax.plot(px, azimuth[pslc], "-", label="Azimuth")
                        ax.legend(loc="best")
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Azimuth (Radians)")

                        ax = fig.add_subplot(4, 1, 2)
                        ax.plot(px, stable[pslc], "-", label="Stable Pointing")
                        ax.plot(px, flags[pslc], color="black", label="Flags")
                        ax.vlines(
                            bstable,
                            ymin=-1,
                            ymax=2,
                            color="green",
                            label="Begin Stable",
                        )
                        ax.vlines(
                            estable, ymin=-1, ymax=2, color="red", label="End Stable"
                        )
                        ax.vlines(
                            bthrow, ymin=-2, ymax=1, color="cyan", label="Begin Throw"
                        )
                        ax.vlines(
                            ethrow, ymin=-2, ymax=1, color="purple", label="End Throw"
                        )
                        ax.legend(loc="best")
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Stable Scan / Throw")

                        ax = fig.add_subplot(4, 1, 3)
                        ax.plot(px, scan_vel[pslc], "-", label="Velocity")
                        ax.vlines(
                            swplot,
                            ymin=np.amin(scan_vel),
                            ymax=np.amax(scan_vel),
                            color="red",
                            label="Velocity Switch",
                        )
                        ax.legend(loc="best")
                        ax.set_xlabel("Samples")
                        ax.set_ylabel("Scan Velocity (Radians / s)")

                        ax = fig.add_subplot(4, 1, 4)
                        ax.plot(px, scan_accel[pslc], "-", label="Acceleration")
                        ax.legend(loc="best")
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

        # Add azimuth ranges to the observations
        azimuth_ranges = AzimuthRanges(
            azimuth=self.azimuth,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
        )
        azimuth_ranges.apply(data, detectors=None)

        # Additionally flag turnarounds as unstable pointing
        flag_intervals = FlagIntervals(
            shared_flags=self.shared_flags,
            shared_flag_bytes=1,
            view_mask=[
                (self.turnaround_interval, defaults.shared_mask_unstable_scanrate),
            ],
        )
        flag_intervals.apply(data, detectors=None)

    def _find_turnaround(self, vel):
        """Fit a polynomial and find the turnaround sample."""
        x = np.arange(len(vel))
        fit_poly = np.polynomial.polynomial.Polynomial.fit(x, vel, 5)
        fit_vel = fit_poly(x)
        vel_switch = np.where(fit_vel[:-1] * fit_vel[1:] < 0)[0]
        if len(vel_switch) != 1:
            return None
        else:
            return vel_switch[0]

    def _gradient(self, data, window, flags=None):
        """Compute the numerical derivative with smoothing.

        Args:
            data (array):  The local data buffer to process.
            window (int):  The number of samples in the smoothing window.
            flags (array):  The optional array of sample flags.

        Returns:
            (array):  The result.

        """
        if flags is not None:
            # Fill flags with noise
            flagged_noise_fill(data, flags, window // 4, poly_order=5)
        # Smooth the data
        smoothed = uniform_filter1d(
            data,
            size=window,
            mode="nearest",
        )
        # Derivative
        result = np.gradient(smoothed)
        return result

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


@trait_docs
class AzimuthRanges(Operator):
    """Measure and record the azimuth ranges in each observation"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    azimuth = Unicode(defaults.azimuth, help="Observation shared key for Azimuth")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for bad azimuth pointing",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
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
        for obs in data.obs:
            az_min_rad = None
            az_max_rad = None
            if obs.comm_col_rank == 0:
                # Compute the good azimuth data along the top process row

                # The azimuth angle
                azimuth = np.array(obs.shared[self.azimuth].data)

                # The azimuth flags
                flags = np.array(obs.shared[self.shared_flags].data)
                flags &= self.shared_flag_mask

                # The min / max Az range for this time chunk
                good = flags == 0

                az = []
                for view in obs.intervals[self.view]:
                    ind = slice(view.first, view.last)
                    az.append(azimuth[ind][good[ind]])
                az = np.hstack(az)

                # Gather the data to the first process and compute the range
                # there.
                if obs.comm_row is not None:
                    az = np.hstack(obs.comm_row.gather(az, root=0))

                if obs.comm_row_rank == 0:
                    # Find the global min / max on one process
                    az = np.unwrap(az)
                    az_min_rad = np.amin(az)
                    az_max_rad = np.amax(az)
                    # Find the right branch
                    while az_min_rad < 0:
                        az_min_rad += 2 * np.pi
                        az_max_rad += 2 * np.pi
                    while az_min_rad > 2 * np.pi:
                        az_min_rad -= 2 * np.pi
                        az_max_rad -= 2 * np.pi
                    # Check if we wrap around
                    if az_max_rad - az_min_rad > 2 * np.pi:
                        az_min_rad = 0
                        az_max_rad = 2 * np.pi

            # Broadcast the result to the whole group
            if obs.comm.comm_group is not None:
                az_min_rad = obs.comm.comm_group.bcast(az_min_rad, root=0)
                az_max_rad = obs.comm.comm_group.bcast(az_max_rad, root=0)

            # Set the metadata
            obs["scan_min_az"] = az_min_rad * u.radian
            obs["scan_max_az"] = az_max_rad * u.radian

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [self.azimuth],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        return req

    def _provides(self):
        return {}
