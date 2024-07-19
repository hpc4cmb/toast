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
from ._helpers import (
    close_data,
    create_ground_data,
    create_outdir,
    fake_flags,
    fake_hwpss,
    create_fake_constant_sky_tod,
)
from .mpi import MPITestCase


class HWPFilterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
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

    def test_hwpss_filter(self):
        # Create a fake ground observations set for testing
        data = create_ground_data(self.comm)

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
        map_key = create_fake_constant_sky_tod(
            data,
            pixels,
            weights,
            map_vals=(1.0, 0.2, 0.2),
        )

        # Simulate noise from this model and save the result for comparison
        # sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        # sim_noise.apply(data)
        ops.Copy(detdata=[(defaults.det_data, "input")]).apply(data)

        # Create HWPSS
        hwpss_scale = 20.0
        tod_rms = np.std(data.obs[0].detdata["input"][0])
        coeff = fake_hwpss(data, defaults.det_data, hwpss_scale * tod_rms)
        n_harmonics = len(coeff) // 4

        # Apply a random inverse relative calibration
        np.random.seed(123456)
        for ob in data.obs:
            fake_relcal = dict()
            for det in ob.local_detectors:
                fake_relcal[det] = 1.0 / np.random.uniform(low=0.5, high=1.5, size=1)[0]
            ob["input_cal"] = fake_relcal
        # ops.CalibrateDetectors(cal_name="input_cal").apply(data)

        # Add random DC level
        # for ob in data.obs:
        #     for det in ob.local_detectors:
        #         dc = np.random.uniform(
        #             low=-2.0 * tod_rms,
        #             high=2.0 * tod_rms,
        #             size=1,
        #         )[0]
        #         ob.detdata[defaults.det_data][det] += dc
        ops.Copy(detdata=[(defaults.det_data, "original")]).apply(data)

        # Make fake flags
        # fake_flags(data)

        # Filter
        ops.Copy(detdata=[(defaults.det_data, "alt_filtered")]).apply(data)
        hwp_filter = ops.HWPFilter(
            filter_order=n_harmonics,
            detrend=True,
            det_data="alt_filtered",
        )
        hwp_filter.apply(data)

        hwpss_filter = ops.HWPSynchronousFilter(
            harmonics=n_harmonics,
            relcal="calibration",
            fill_gaps=True,
        )
        hwpss_filter.apply(data)

        ops.Copy(detdata=[(defaults.det_data, "filtered")]).apply(data)

        # Apply estimated relative calibration
        # ops.CalibrateDetectors(cal_name=hwpss_filter.relcal).apply(data)

        for ob in data.obs:
            if self.make_plots and ob.comm.group_rank == 0:
                import matplotlib.pyplot as plt

                cmap = plt.get_cmap("tab10")
                # Recall that we have flagged the first half of each detector above.
                # Plot the whole TOD and also a slice near the transition.
                n_all_samp = ob.n_all_samples
                plot_ranges = list()
                plot_files = list()
                n_plot = 4
                fig_height = 6 * n_plot
                axes = dict()
                for first, last in [
                    (0, n_all_samp),
                    (n_all_samp // 2 - 100, n_all_samp // 2 + 100),
                    (n_all_samp - 200, n_all_samp),
                ]:
                    rangestr = f"{first}-{last}"
                    axes[rangestr] = dict()
                    axes[rangestr]["range"] = (first, last)
                    axes[rangestr]["file"] = os.path.join(
                        self.outdir,
                        f"compare_{ob.name}_{first}-{last}.pdf",
                    )
                    axes[rangestr]["fig"] = plt.figure(figsize=(12, fig_height), dpi=72)
                    axes[rangestr]["ax"] = list()
                    for pl in range(n_plot):
                        axes[rangestr]["ax"].append(
                            axes[rangestr]["fig"].add_subplot(
                                n_plot, 1, pl + 1, aspect="auto"
                            )
                        )

            for idet, det in enumerate(
                ob.select_local_detectors(flagmask=hwpss_filter.det_flag_mask)
            ):
                # original_cal = 1.0 / ob["input_cal"][det]
                # rel_cal = ob[hwpss_filter.relcal][det]
                # print(f"{ob.name}:{det}:  input = {original_cal}, solved = {rel_cal}")
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                original = ob.detdata["original"][det]
                input = ob.detdata["input"][det]
                filtered = ob.detdata["filtered"][det]
                alt = ob.detdata["alt_filtered"][det]
                calibrated = ob.detdata[defaults.det_data][det]
                residual = calibrated - input
                alt_resid = alt - input

                if self.make_plots and ob.comm.group_rank == 0:
                    if det != "D2A-150":
                        continue
                    for rangestr, props in axes.items():
                        rg = props["range"]
                        file = props["file"]
                        fig = props["fig"]
                        ax = props["ax"]
                        plot_slc = slice(rg[0], rg[1], 1)
                        # Plot input signal and starting data with hwpss + calibration
                        ax[0].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            input[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Sky",
                        )
                        ax[0].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            original[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Total",
                        )
                        # Plot the filtered data and flags
                        ax[1].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            filtered[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Filtered",
                        )
                        ax[1].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            filtered[plot_slc],
                            color="black",
                            linestyle="dotted",
                            marker="o",
                            label=f"{det} Alt Filtered",
                        )
                        ax[1].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            flags[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Flags",
                        )
                        # Plot the calibrated data
                        ax[2].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            input[plot_slc],
                            color="red",
                            label=f"{det} Input",
                        )
                        ax[2].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            calibrated[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Calibrated",
                        )
                        ax[2].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            flags[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Flags",
                        )
                        # Plot residual
                        ax[3].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            residual[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Residual",
                        )
                        ax[3].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            alt_resid[plot_slc],
                            color="black",
                            linestyle="dotted",
                            marker="o",
                            label=f"{det} Alt Residual",
                        )
                        ax[3].plot(
                            ob.shared[defaults.times].data[plot_slc],
                            flags[plot_slc],
                            color=cmap(idet),
                            label=f"{det} Flags",
                        )

                # FIXME: understand what is going on here...
                # self.assertTrue(
                #     np.absolute(rel_cal - original_cal)
                #     < 1.0e-4
                # )

                # Check that the filtered signal is cleaner than the input signal
                # self.assertTrue(np.std(calibrated[good]) < np.std(input[good]))

                # Check that the flagged samples were also cleaned and not,
                # for example, set to zero. Use np.diff() to remove any
                # residual trend
                # if (
                #     np.std(np.diff(output) - np.diff(original))
                #     < 0.1 * np.std(np.diff(output))
                # ):
                #     print(f"")
                #     self.assertTrue()

            if self.make_plots and ob.comm.group_rank == 0:
                for rangestr, props in axes.items():
                    rg = props["range"]
                    file = props["file"]
                    fig = props["fig"]
                    ax = props["ax"]
                    for a in ax:
                        a.legend(loc="best")
                    fig.suptitle(f"Obs {ob.name}:{rangestr}")
                    fig.savefig(file)
                    plt.close(fig)

        close_data(data)
