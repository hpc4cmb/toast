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
        self,
        fname,
        net,
        true_freq,
        true_psd,
        est_freq,
        est_psd,
        fit_freq=None,
        fit_psd=None,
    ):
        set_matplotlib_backend()
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[12, 8])
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            true_freq.to_value(u.Hz),
            true_psd.to_value(u.K**2 * u.s),
            color="black",
            label="Input",
        )
        ax.loglog(
            est_freq.to_value(u.Hz),
            est_psd.to_value(u.K**2 * u.s),
            color="red",
            label="Estimated",
        )
        if fit_freq is not None:
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
        ax.set_ylim(
            np.amin(est_psd.to_value(u.K**2 * u.s)),
            1.1 * np.amax(est_psd.to_value(u.K**2 * u.s)),
        )
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [K$^2$ / Hz]")
        ax.legend(loc="best")
        fig.savefig(fname)
        plt.close()

    def compare_timestreams(self, det, times, original, filtered, flags, plotroot=None):
        good = flags == 0
        if plotroot is not None:
            import matplotlib.pyplot as plt

            for prange in [(0, 500), (0, len(times))]:
                pslc = slice(prange[0], prange[1], 1)
                plotfile = f"{plotroot}_{prange[0]}-{prange[1]}.pdf"
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

    def test_clean_filter(self):
        testdir = os.path.join(self.outdir, "clean")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create a fake ground data set for testing.  Use a high fknee.
        data = create_ground_data(
            self.comm, sample_rate=100.0 * u.Hz, fknee=20.0 * u.Hz
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

        nse_filter = ops.NoiseFilter(noise_model=sim_noise.noise_model, debug=None)
        nse_filter.apply(data)

        # Compare filtered data.
        for ob in data.obs:
            times = ob.shared[defaults.times].data
            shflags = ob.shared[defaults.shared_flags].data
            for det in ob.select_local_detectors(flagmask=nse_filter.det_mask):
                flags = np.array(shflags)
                flags |= ob.detdata[defaults.det_flags][det]
                pltroot = None
                if self.make_plots:
                    pltroot = os.path.join(testdir, f"tod_{ob.name}:{det}")
                self.compare_timestreams(
                    det,
                    times,
                    ob.detdata["original"][det],
                    ob.detdata[defaults.det_data][det],
                    flags,
                    plotroot=pltroot,
                )

        close_data(data)

    # def test_bandpassed(self):
    #     # Test the challenging case where the timestreams have been bandpassed.

    #     # Create a fake ground data set for testing
    #     data = create_ground_data(self.comm, sample_rate=100.0 * u.Hz, fknee=5.0 * u.Hz)

    #     # Create some detector pointing matrices
    #     detpointing = ops.PointingDetectorSimple()
    #     pixels = ops.PixelsHealpix(
    #         nside=64,
    #         create_dist="pixel_dist",
    #         detector_pointing=detpointing,
    #     )
    #     pixels.apply(data)
    #     weights = ops.StokesWeights(
    #         mode="IQU",
    #         hwp_angle="hwp_angle",
    #         detector_pointing=detpointing,
    #     )
    #     weights.apply(data)

    #     # Create an uncorrelated noise model from focalplane detector properties
    #     default_model = ops.DefaultNoiseModel(noise_model="noise_model")
    #     default_model.apply(data)

    #     # Simulate noise from this model
    #     sim_noise = ops.SimNoise(noise_model="noise_model")
    #     sim_noise.apply(data)

    #     # Estimate noise
    #     estim = ops.NoiseEstim(
    #         name="estimate_model",
    #         output_dir=self.outdir,
    #         out_model="noise_estimate",
    #         lagmax=200,
    #         nbin_psd=128,
    #         nsum=4,
    #     )
    #     estim.apply(data)

    #     # Compute a 1/f fit to this
    #     noise_fitter = ops.FitNoiseModel(
    #         noise_model=estim.out_model,
    #         out_model="fit_noise_model",
    #     )
    #     noise_fitter.apply(data)

    #     for ob in data.obs:
    #         if ob.comm.group_rank == 0:
    #             input_model = ob["noise_model"]
    #             estim_model = ob["noise_estimate"]
    #             fit_model = ob[noise_fitter.out_model]
    #             for det in ob.select_local_detectors(flagmask=estim.det_flag_mask):
    #                 fname = os.path.join(
    #                     self.outdir, f"estimate_model_{ob.name}_{det}.pdf"
    #                 )
    #                 plot_noise_estim_compare(
    #                     fname,
    #                     input_model.NET(det),
    #                     input_model.freq(det),
    #                     input_model.psd(det),
    #                     estim_model.freq(det),
    #                     estim_model.psd(det),
    #                     fit_freq=fit_model.freq(det),
    #                     fit_psd=fit_model.psd(det),
    #                 )
    #         if ob.comm.comm_group is not None:
    #             ob.comm.comm_group.barrier()

    #     if data.comm.comm_world is not None:
    #         data.comm.comm_world.barrier()

    #     # Verify that the fit NET is close to the input.
    #     for ob in data.obs:
    #         input_model = ob["noise_model"]
    #         estim_model = ob["noise_estimate"]
    #         fit_model = ob[noise_fitter.out_model]
    #         for det in ob.select_local_detectors(flagmask=estim.det_flag_mask):
    #             np.testing.assert_almost_equal(
    #                 np.mean(input_model.psd(det)[-5:]).value,
    #                 np.mean(fit_model.psd(det)[-5:]).value,
    #                 decimal=3,
    #             )

    #     close_data(data)
