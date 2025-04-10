# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from .helpers import (
    close_data,
    create_ground_data,
    create_outdir,
    fake_flags,
)
from .mpi import MPITestCase


class FillGapsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def test_gap_fill(self):
        # Create some test data.  Disable HWPSS, since we are not demodulating
        # in this example.
        data, input_rms = self.create_test_data()

        # Make a copy for later comparison
        ops.Copy(detdata=[(defaults.det_data, "input")]).apply(data)

        # Linear fit plus noise
        filler = ops.FillGaps(
            shared_flag_mask=defaults.shared_mask_nonscience,
            buffer=1.0 * u.s,
            poly_order=1,
        )
        filler.apply(data)

        # Diagnostic plots of one detector on each process.
        if self.make_plots:
            import matplotlib.pyplot as plt

            for ob in data.obs:
                det = ob.select_local_detectors(flagmask=defaults.det_mask_nonscience)[
                    0
                ]
                n_all_samp = ob.n_all_samples
                n_plot = 2
                fig_height = 6 * n_plot
                pltsamp = 200

                for first, last in [
                    (0, n_all_samp),
                    (n_all_samp // 2 - pltsamp, n_all_samp // 2 + pltsamp),
                ]:
                    plot_slc = slice(first, last, 1)
                    outfile = os.path.join(
                        self.outdir,
                        f"filled_{ob.name}_{det}_{first}-{last}.pdf",
                    )

                    times = ob.shared[defaults.times].data
                    samp_indx = np.arange(n_all_samp)
                    input = ob.detdata["input"][det]
                    signal = ob.detdata[defaults.det_data][det]
                    detflags = ob.detdata[defaults.det_flags][det]
                    shflags = ob.shared[defaults.shared_flags].data

                    fig = plt.figure(figsize=(12, fig_height), dpi=72)
                    ax = fig.add_subplot(n_plot, 1, 1, aspect="auto")
                    # Plot signal
                    ax.plot(
                        samp_indx[plot_slc],
                        input[plot_slc],
                        color="black",
                        label=f"{det} Input",
                    )
                    ax.plot(
                        samp_indx[plot_slc],
                        signal[plot_slc],
                        color="red",
                        label=f"{det} Filled",
                    )
                    ax.legend(loc="best")
                    # Plot flags
                    ax = fig.add_subplot(n_plot, 1, 2, aspect="auto")
                    ax.plot(
                        samp_indx[plot_slc],
                        shflags[plot_slc],
                        color="blue",
                        label="Shared Flags",
                    )
                    ax.plot(
                        samp_indx[plot_slc],
                        detflags[plot_slc],
                        color="red",
                        label=f"{det} Flags",
                    )
                    ax.legend(loc="best")
                    fig.suptitle(f"Obs {ob.name}: {first} - {last}")
                    fig.savefig(outfile)
                    plt.close(fig)

        close_data(data)

    def create_test_data(self):
        # Slightly slower than 0.5 Hz
        hwp_rpm = 29.0
        hwp_rate = 2 * np.pi * hwp_rpm / 60.0  # rad/s

        sample_rate = 30 * u.Hz
        ang_per_sample = hwp_rate / sample_rate.to_value(u.Hz)

        # Create a fake ground observations set for testing.
        data = create_ground_data(
            self.comm,
            sample_rate=sample_rate,
            hwp_rpm=hwp_rpm,
            pixel_per_process=1,
            single_group=True,
            fp_width=5.0 * u.degree,
        )

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate fake instrumental noise
        sim_noise = ops.SimNoise(noise_model="noise_model")
        sim_noise.apply(data)

        # Create flagged samples
        fake_flags(data)

        # Now we will increase the noise amplitude of flagged samples to
        # make it easier to check that we have filled gaps with something
        # reasonable.
        rms = dict()
        for ob in data.obs:
            for det in ob.local_detectors:
                input = np.std(ob.detdata[defaults.det_data][det])
                rms[det] = input
                flags = np.array(ob.shared[defaults.shared_flags].data)
                flags[:] |= ob.detdata[defaults.det_flags][det, :]
                bad = flags != 0
                ob.detdata[defaults.det_data][det, bad] *= 20

        return data, rms
