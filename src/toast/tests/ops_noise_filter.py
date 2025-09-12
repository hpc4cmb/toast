# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column
from scipy import interpolate

from .. import ops as ops
from .. import qarray as qa
from ..noise import Noise
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import (
    close_data,
    create_fake_healpix_scanned_tod,
    create_ground_data,
    create_outdir,
    fake_flags,
)
from .mpi import MPITestCase

XAXIS, YAXIS, ZAXIS = np.eye(3)


class NoiseFilterTest(MPITestCase):
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

    def plot_noise_model(
        self, fname, net, input_freq, input_psd, est_freq, est_psd, fit_freq, fit_psd
    ):
        set_matplotlib_backend()
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[12, 8])
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            input_freq.to_value(u.Hz),
            input_psd.to_value(u.K**2 * u.s),
            color="black",
            label="Input",
        )
        ax.loglog(
            est_freq.to_value(u.Hz),
            est_psd.to_value(u.K**2 * u.s),
            color="red",
            label="Estimated",
        )
        ax.loglog(
            fit_freq.to_value(u.Hz),
            fit_psd.to_value(u.K**2 * u.s),
            color="green",
            label="Fit to 1/f Model",
        )
        net = net.to_value(u.K / u.Hz**0.5)
        ax.axhline(
            net**2,
            label=f"NET = {net:.3f} K" + r" / $\sqrt{\mathrm{Hz}}$",
            linestyle="--",
            color="blue",
        )
        ax.set_xlim(est_freq[0].to_value(u.Hz), est_freq[-1].to_value(u.Hz))
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [K$^2$ / Hz]")
        ax.legend(loc="best")
        fig.savefig(fname)
        plt.close()

    def check_timestreams(self, data, testdir, wmin, wmax):
        # First estimate the noise on the filtered timestreams
        estim = ops.NoiseEstim(
            name="estimate_model",
            output_dir=testdir,
            out_model="noise_estimate",
            lagmax=200,
            nbin_psd=64,
            nsum=10,
        )
        estim.apply(data)

        # Compute a 1/f fit to this
        noise_fitter = ops.FitNoiseModel(
            noise_model=estim.out_model,
            out_model="fit_noise_model",
            white_noise_min=wmin,
            white_noise_max=wmax,
        )
        noise_fitter.apply(data)

        # Check each detector, optionally with comparison plots
        for obs in data.obs:
            times = obs.shared[defaults.times].data
            shflags = obs.shared[defaults.shared_flags].data
            for det in obs.select_local_detectors(flagmask=defaults.det_mask_invalid):
                flags = np.array(shflags)
                flags |= obs.detdata[defaults.det_flags][det]
                good = flags == 0
                original = obs.detdata["original"][det]
                filtered = obs.detdata[defaults.det_data][det]

                nse_in = obs["noise_model"]
                net_in = nse_in.NET(det)
                fknee_in = nse_in.fknee(det)
                nse_out = obs["fit_noise_model"]
                net_out = nse_out.NET(det)
                fknee_out = nse_out.fknee(det)

                if self.make_plots:
                    import matplotlib.pyplot as plt

                    pltroot = os.path.join(testdir, f"tod_{obs.name}-{det}")
                    for prange in [(0, 500), (0, len(times))]:
                        pslc = slice(prange[0], prange[1], 1)
                        plotfile = f"{pltroot}_{prange[0]}-{prange[1]}.pdf"
                        fig = plt.figure(figsize=(12, 12), dpi=72)
                        ax = fig.add_subplot(2, 1, 1, aspect="auto")
                        ax.plot(
                            times[good][pslc],
                            original[good][pslc],
                            color="black",
                            label=f"{det} Input",
                        )
                        ax.legend(loc="best")
                        ax = fig.add_subplot(2, 1, 2, aspect="auto")
                        ax.plot(
                            times[good][pslc],
                            filtered[good][pslc],
                            color="red",
                            label=f"{det} Filtered",
                        )
                        ax.legend(loc="best")
                        fig.savefig(plotfile)
                        plt.close(fig)
                    self.plot_noise_model(
                        os.path.join(testdir, f"psd_{obs.name}-{det}.pdf"),
                        net_in,
                        nse_in.freq(det),
                        nse_in.psd(det),
                        obs["noise_estimate"].freq(det),
                        obs["noise_estimate"].psd(det),
                        nse_out.freq(det),
                        nse_out.psd(det),
                    )

                # Check that the f_knee of the fit model is small enough
                if fknee_out > 0.1 * fknee_in:
                    msg = f"{obs.name}:{det} resulting f_knee ({fknee_out}) too high"
                    msg += f" compared to input ({fknee_in})"
                    print(msg, flush=True)
                    self.assertTrue(False)

    def test_clean_filter(self):
        testdir = os.path.join(self.outdir, "clean")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create a fake ground data set for testing.  Use a high fknee.
        data = create_ground_data(
            self.comm, sample_rate=100.0 * u.Hz, fknee=10.0 * u.Hz
        )

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model")
        sim_noise.apply(data)

        # Make a copy of the original data
        ops.Copy(detdata=[(sim_noise.det_data, "original")]).apply(data)

        # Since we are using an analytic noise model, it is already smooth
        # enough to use for the kernel.
        wmin = 40.0 * u.Hz
        wmax = 50.0 * u.Hz
        nse_filter = ops.NoiseFilter(
            noise_model=sim_noise.noise_model,
            white_noise_min=wmin,
            white_noise_max=wmax,
            debug=None,
        )
        nse_filter.apply(data)

        # Compare filtered data.
        self.check_timestreams(data, testdir, wmin, wmax)

        close_data(data)
