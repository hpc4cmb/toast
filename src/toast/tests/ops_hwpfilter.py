# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
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

    def test_hwpfilter(self):
        # Create a fake ground observations set for testing
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model and save for comparison
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)
        ops.Copy(detdata=[(defaults.det_data, "original")]).apply(data)

        # Add HWPSS
        hwpss_scale = 20.0
        tod_rms = np.std(data.obs[0].detdata["original"][0])
        coeff = fake_hwpss(data, defaults.det_data, hwpss_scale * tod_rms)
        n_harmonics = len(coeff) // 4
        ops.Copy(detdata=[(defaults.det_data, "input")]).apply(data)

        # Make fake flags
        fake_flags(data)

        # Filter
        hwpfilter = ops.HWPFilter(
            trend_order=3,
            filter_order=n_harmonics,
            detrend=True,
        )
        hwpfilter.apply(data)

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=hwpfilter.det_flag_mask):
                flags = ob.shared[defaults.shared_flags].data & self.shared_flag_mask
                flags |= ob.detdata[defaults.det_flags][det]
                good = flags == 0
                original = ob.detdata["original"][det]
                input = ob.detdata["input"][det]
                output = ob.detdata[defaults.det_data][det]

                if self.make_plots:
                    import matplotlib.pyplot as plt

                    n_samp = len(output)
                    # for ns in [500, n_samp]:
                    for ns in [200]:
                        plot_slc = slice(0, ns, 1)
                        savefile = os.path.join(
                            self.outdir,
                            f"compare_{ob.name}_{det}_{ns}.pdf",
                        )
                        n_plot = 3
                        fig_height = 6 * n_plot

                        # Find the data range of the input
                        dmin = np.amin(input[plot_slc])
                        dmax = np.amax(input[plot_slc])
                        dhalf = (dmax - dmin) / 2

                        fig = plt.figure(figsize=(12, fig_height), dpi=72)

                        ax = fig.add_subplot(n_plot, 1, n_plot - 2, aspect="auto")
                        ax.plot(
                            ob.shared[defaults.times].data[plot_slc],
                            original[plot_slc],
                            c="black",
                            label=f"Original Signal",
                        )
                        ax.set_ylim(bottom=dmin, top=dmax)
                        ax.legend(loc="best")

                        ax = fig.add_subplot(n_plot, 1, n_plot - 1, aspect="auto")
                        ax.plot(
                            ob.shared[defaults.times].data[plot_slc],
                            input[plot_slc],
                            c="blue",
                            label=f"Original + HWPSS",
                        )
                        ax.plot(
                            ob.shared[defaults.times].data[plot_slc],
                            output[plot_slc],
                            c="red",
                            label=f"Filtered",
                        )
                        ax.set_ylim(bottom=dmin, top=dmax)
                        ax.legend(loc="best")

                        ax = fig.add_subplot(n_plot, 1, n_plot, aspect="auto")
                        residual = (
                            original[plot_slc]
                            - output[plot_slc]
                        )
                        ax.plot(
                            ob.shared[defaults.times].data[plot_slc],
                            original[plot_slc],
                            c="black",
                            label=f"Original",
                        )
                        ax.plot(
                            ob.shared[defaults.times].data[plot_slc],
                            residual,
                            c="cyan",
                            label=f"Residual",
                        )
                        ax.legend(loc="best")
                        plt.title(f"Obs {ob.name}, det {det}")
                        plt.savefig(savefile)
                        plt.close()

                # Check that the filtered signal is cleaner than the input signal
                self.assertTrue(np.std(output[good]) < np.std(input[good]))

                # Check that the flagged samples were also cleaned and not,
                # for example, set to zero. Use np.diff() to remove any
                # residual trend
                self.assertTrue(
                    np.std(np.diff(output) - np.diff(original))
                    < 0.1 * np.std(np.diff(output))
                )

        close_data(data)
