# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column

from .. import ops as ops
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..vis import set_matplotlib_backend
from .helpers import (
    close_data,
    create_fake_healpix_scanned_tod,
    create_ground_data,
    create_outdir,
    fake_flags,
    fake_hwpss,
)
from .mpi import MPITestCase


class HWPModelTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        self.shared_flag_mask = 1
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True
        # Extra debug plots?
        self.debug_plots = False

    def create_test_data(self, testdir):
        # Slightly slower than 1 Hz
        hwp_rpm = 59.0
        hwp_rate = 2 * np.pi * hwp_rpm / 60.0  # rad/s

        sample_rate = 60 * u.Hz
        ang_per_sample = hwp_rate / sample_rate.to_value(u.Hz)

        # Create a fake ground observations set for testing
        data = create_ground_data(self.comm, sample_rate=sample_rate, hwp_rpm=hwp_rpm)

        # Modify the HWP angle to be stopped initially and then accelerate to the
        # target velocity.

        n_ramp = 50
        for ob in data.obs:
            end_rampup = int(0.1 * ob.n_local_samples)
            begin_rampup = end_rampup - n_ramp
            hwp_data = ob.shared[defaults.hwp_angle].data
            ang_end = hwp_data[end_rampup - 1]
            half_ramp = n_ramp // 2
            max_accel = ang_per_sample * 2 / n_ramp
            ang_accel = np.concatenate(
                [
                    (max_accel / half_ramp) * np.arange(0, half_ramp, 1),
                    (max_accel / half_ramp) * np.arange(half_ramp, 0, -1),
                ],
                axis=None,
            )
            ang_vel = np.cumsum(ang_accel)
            ang_pos = np.cumsum(ang_vel)
            off = ang_pos[-1] - ang_end
            ramp = ang_pos - off

            if ob.comm.group_rank == 0:
                hwp_data[:end_rampup] = ramp[0]
                hwp_data[begin_rampup:end_rampup] = ramp

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Generic pointing matrix
        nside = 128
        detpointing = ops.PointingDetectorSimple(shared_flag_mask=0)
        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        # Create some fake sky tod
        skyfile = os.path.join(testdir, "input_sky.fits")
        map_key = "input_sky"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights,
            skyfile,
            "input_sky_dist",
            map_key=map_key,
            fwhm=30.0 * u.arcmin,
            lmax=3 * pixels.nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Simulate noise from this model and save the result for comparison
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)
        ops.Copy(detdata=[(defaults.det_data, "input")]).apply(data)

        # Create HWPSS
        hwpss_scale = 20.0
        tod_rms = np.std(data.obs[0].detdata["input"][0])
        coeff = fake_hwpss(data, defaults.det_data, hwpss_scale * tod_rms)
        return data, tod_rms, coeff

    def plot_compare(self, dir, obs, det_mask):
        if not self.make_plots:
            return
        import matplotlib.pyplot as plt

        # Every process will plot its first detector.
        selected_dets = obs.select_local_detectors(flagmask=det_mask)

        # Recall that we have flagged the first half of each detector above.
        # Plot the whole TOD and also a slice near the transition.
        n_all_samp = obs.n_all_samples
        plot_ranges = list()
        plot_files = list()
        n_plot = 4
        fig_height = 6 * n_plot
        axes = dict()
        pltsamp = 400
        for first, last in [
            (0, n_all_samp),
            (n_all_samp // 2 - pltsamp, n_all_samp // 2 + pltsamp),
            (n_all_samp - 2 * pltsamp, n_all_samp),
            (int(0.1 * n_all_samp) - 50, int(0.1 * n_all_samp) + 50),
        ]:
            rangestr = f"{first}-{last}"
            axes[rangestr] = dict()
            axes[rangestr]["range"] = (first, last)
            axes[rangestr]["file"] = os.path.join(
                dir,
                f"compare_{obs.name}_{selected_dets[0]}_{first}-{last}.pdf",
            )
            axes[rangestr]["fig"] = plt.figure(figsize=(12, fig_height), dpi=72)
            axes[rangestr]["ax"] = list()
            for pl in range(n_plot):
                axes[rangestr]["ax"].append(
                    axes[rangestr]["fig"].add_subplot(n_plot, 1, pl + 1, aspect="auto")
                )

        for idet, det in enumerate(selected_dets):
            if idet != 0:
                continue
            # original_cal = 1.0 / ob["input_cal"][det]
            # rel_cal = ob[hwpss_model.relcal_fixed][det]
            # print(f"{ob.name}:{det}:  input = {original_cal}, solved = {rel_cal}")
            flags = obs.shared[defaults.shared_flags].data & self.shared_flag_mask
            flags |= obs.detdata[defaults.det_flags][det]
            good = flags == 0
            original = obs.detdata["original"][det]
            input = obs.detdata["input"][det]
            filtered = obs.detdata["filtered"][det]
            alt = obs.detdata["alt_filtered"][det]
            calibrated = obs.detdata[defaults.det_data][det]
            residual = calibrated - input
            alt_resid = alt - input

            for rangestr, props in axes.items():
                rg = props["range"]
                file = props["file"]
                fig = props["fig"]
                ax = props["ax"]
                plot_slc = slice(rg[0], rg[1], 1)
                # Plot input signal and starting data with hwpss + calibration
                ax[0].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    original[plot_slc],
                    color="blue",
                    label=f"{det} Input + HWPSS",
                )
                ax[0].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    input[plot_slc],
                    color="red",
                    label=f"{det} Input",
                )
                # Plot the filtered data
                ax[1].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    input[plot_slc],
                    color="red",
                    label=f"{det} Input",
                )
                ax[1].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    alt[plot_slc],
                    color="black",
                    label=f"{det} Alternate HWPFilter Result",
                )
                ax[1].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    filtered[plot_slc],
                    color="green",
                    linestyle="dotted",
                    label=f"{det} Filtered, Uncalibrated",
                )
                ax[1].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    calibrated[plot_slc],
                    color="green",
                    label=f"{det} Filtered, Calibrated",
                )
                # Plot residual
                ax[2].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    input[plot_slc],
                    color="red",
                    label=f"{det} Input",
                )
                ax[2].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    alt_resid[plot_slc],
                    color="black",
                    label=f"{det} Alternate HWPFilter Residual",
                )
                ax[2].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    residual[plot_slc],
                    color="green",
                    label=f"{det} Residual",
                )
                # Plot flags
                ax[3].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    obs.shared[defaults.hwp_angle].data[plot_slc],
                    color="black",
                    label=f"HWP Angle",
                )
                ax[3].plot(
                    obs.shared[defaults.times].data[plot_slc],
                    flags[plot_slc],
                    color="cyan",
                    label=f"{det} Flags",
                )
        for rangestr, props in axes.items():
            rg = props["range"]
            file = props["file"]
            fig = props["fig"]
            ax = props["ax"]
            for a in ax:
                a.legend(loc="best")
            fig.suptitle(f"Obs {obs.name}:{rangestr}")
            fig.savefig(file)
            plt.close(fig)

    def test_hwpss_basic(self):
        testdir = os.path.join(self.outdir, "basic")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        data, tod_rms, coeff = self.create_test_data(testdir)
        n_harmonics = len(coeff[data.obs[0].name]) // 4

        # Add random DC level
        for ob in data.obs:
            for det in ob.local_detectors:
                dc = np.random.uniform(
                    low=-5.0 * tod_rms,
                    high=5.0 * tod_rms,
                    size=1,
                )[0]
                ob.detdata[defaults.det_data][det] += dc
        ops.Copy(detdata=[(defaults.det_data, "original")]).apply(data)

        # Skip flags for this basic test, so we can clearly see the performance around
        # the HWP acceleration at the start.
        # fake_flags(data)

        # Filter
        ops.Copy(detdata=[(defaults.det_data, "alt_filtered")]).apply(data)
        hwp_filter = ops.HWPFilter(
            filter_order=n_harmonics,
            detrend=True,
            det_data="alt_filtered",
        )
        hwp_filter.apply(data)

        debug = None
        if self.debug_plots:
            debug = os.path.join(testdir, "debug")
        hwpss_model = ops.HWPSynchronousModel(
            harmonics=n_harmonics,
            subtract_model=True,
            fill_gaps=True,
            debug=debug,
        )
        hwpss_model.apply(data)

        ops.Copy(detdata=[(defaults.det_data, "filtered")]).apply(data)

        for ob in data.obs:
            self.plot_compare(testdir, ob, defaults.det_mask_invalid)
            # Check that filtered and calibrated signal has smaller rms than
            # the original.
            for det in ob.select_local_detectors(flagmask=255):
                good = ob.detdata[defaults.det_flags][det] == 0
                original_rms = np.std(ob.detdata["original"][det][good])
                filtered_rms = np.std(ob.detdata[defaults.det_data][det][good])
                # print(f"{ob.name}[{det}]:  {filtered_rms} <? {original_rms}")
                self.assertTrue(filtered_rms < original_rms)
        close_data(data)

    def test_hwpss_relcal_fixed(self):
        testdir = os.path.join(self.outdir, "fixed")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        data, tod_rms, coeff = self.create_test_data(testdir)
        n_harmonics = len(coeff[data.obs[0].name]) // 4

        # Apply a random inverse relative calibration
        np.random.seed(123456)
        for ob in data.obs:
            fake_relcal = dict()
            for det in ob.local_detectors:
                fake_relcal[det] = 1.0 / np.random.uniform(low=0.5, high=1.5, size=1)[0]
            ob["input_cal"] = fake_relcal
        ops.CalibrateDetectors(cal_name="input_cal").apply(data)

        # Add random DC level
        for ob in data.obs:
            for det in ob.local_detectors:
                dc = np.random.uniform(
                    low=-5.0 * tod_rms,
                    high=5.0 * tod_rms,
                    size=1,
                )[0]
                ob.detdata[defaults.det_data][det] += dc
        ops.Copy(detdata=[(defaults.det_data, "original")]).apply(data)

        # Make fake flags
        fake_flags(data)

        # Filter
        ops.Copy(detdata=[(defaults.det_data, "alt_filtered")]).apply(data)
        hwp_filter = ops.HWPFilter(
            filter_order=n_harmonics,
            detrend=True,
            det_data="alt_filtered",
        )
        hwp_filter.apply(data)

        debug = None
        if self.debug_plots:
            debug = os.path.join(testdir, "debug")
        hwpss_model = ops.HWPSynchronousModel(
            harmonics=n_harmonics,
            relcal_fixed="calibration",
            subtract_model=True,
            fill_gaps=True,
            debug=debug,
        )
        hwpss_model.apply(data)

        ops.Copy(detdata=[(defaults.det_data, "filtered")]).apply(data)

        # Apply estimated relative calibration
        ops.CalibrateDetectors(cal_name=hwpss_model.relcal_fixed).apply(data)

        for ob in data.obs:
            self.plot_compare(testdir, ob, defaults.det_mask_invalid)
            # Check that filtered and calibrated signal has smaller rms than
            # the original.
            for det in ob.select_local_detectors(flagmask=255):
                good = ob.detdata[defaults.det_flags][det] == 0
                original_rms = np.std(ob.detdata["original"][det][good])
                filtered_rms = np.std(ob.detdata[defaults.det_data][det][good])
                # print(f"{ob.name}[{det}]:  {filtered_rms} <? {original_rms}")
                self.assertTrue(filtered_rms < original_rms)
        close_data(data)

    def test_hwpss_relcal_continuous(self):
        testdir = os.path.join(self.outdir, "continuous")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        data, tod_rms, coeff = self.create_test_data(testdir)
        n_harmonics = len(coeff[data.obs[0].name]) // 4

        # Apply a random inverse relative calibration that is time-varying
        np.random.seed(123456)
        for ob in data.obs:
            ob.detdata.create("input_cal", units=ob.detdata[defaults.det_data].units)
            common = 1.0 + 0.5 * np.sin(
                np.arange(ob.n_local_samples) * 2 * np.pi / ob.n_local_samples
            )
            for det in ob.local_detectors:
                ob.detdata["input_cal"][det] = (
                    np.random.uniform(low=0.5, high=1.5, size=1)[0] * common
                )
        ops.Combine(
            first=defaults.det_data,
            second="input_cal",
            result=defaults.det_data,
            op="divide",
        ).apply(data)

        # Add random DC level
        for ob in data.obs:
            for det in ob.local_detectors:
                dc = np.random.uniform(
                    low=-5.0 * tod_rms,
                    high=5.0 * tod_rms,
                    size=1,
                )[0]
                ob.detdata[defaults.det_data][det] += dc
        ops.Copy(detdata=[(defaults.det_data, "original")]).apply(data)

        # Make fake flags
        fake_flags(data)

        # Filter
        ops.Copy(detdata=[(defaults.det_data, "alt_filtered")]).apply(data)
        hwp_filter = ops.HWPFilter(
            filter_order=n_harmonics,
            detrend=True,
            det_data="alt_filtered",
        )
        hwp_filter.apply(data)

        debug = None
        if self.debug_plots:
            debug = os.path.join(testdir, "debug")
        hwpss_model = ops.HWPSynchronousModel(
            harmonics=n_harmonics,
            relcal_continuous="calibration",
            # chunk_view="scanning",
            chunk_time=60 * u.second,
            subtract_model=True,
            fill_gaps=True,
            debug=debug,
        )
        hwpss_model.apply(data)

        ops.Copy(detdata=[(defaults.det_data, "filtered")]).apply(data)

        # Apply estimated relative calibration
        ops.Combine(
            first=defaults.det_data,
            second=hwpss_model.relcal_continuous,
            result=defaults.det_data,
            op="multiply",
        ).apply(data)

        for ob in data.obs:
            self.plot_compare(testdir, ob, defaults.det_mask_invalid)
            # Check that filtered and calibrated signal has smaller rms than
            # the original.
            for det in ob.select_local_detectors(flagmask=255):
                good = ob.detdata[defaults.det_flags][det] == 0
                original_rms = np.std(ob.detdata["original"][det][good])
                filtered_rms = np.std(ob.detdata[defaults.det_data][det][good])
                # print(f"{ob.name}[{det}]:  {filtered_rms} <? {original_rms}")
                self.assertTrue(filtered_rms < original_rms)
        close_data(data)
