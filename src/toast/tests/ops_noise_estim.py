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
from ..pixels import PixelData, PixelDistribution
from ..pixels_io import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import (
    create_fake_sky,
    create_ground_data,
    create_outdir,
    create_satellite_data,
    fake_flags,
)
from .mpi import MPITestCase

XAXIS, YAXIS, ZAXIS = np.eye(3)


class NoiseEstimTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_noise_estim(self):
        # Create a fake ground data set for testing
        data = create_ground_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle="hwp_angle",
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Write map to a file
        map_file = os.path.join(self.outdir, "fake_map.fits")
        write_healpix_fits(data["fake_map"], map_file, nest=pixels.nest)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model")
        sim_noise.apply(data)

        # Make fake flags
        fake_flags(data)

        # Estimate noise

        estim = ops.NoiseEstim(
            name="noise_with_signal",
            det_flag_mask=1,
            shared_flag_mask=1,
            # view="scanning",
            output_dir=self.outdir,
            lagmax=1000,
            nbin_psd=300,
        )
        estim.apply(data)

        estim = ops.NoiseEstim(
            name="noise_without_signal",
            mapfile=map_file,
            det_flag_mask=1,
            shared_flag_mask=1,
            detector_pointing=detpointing,
            pixel_pointing=pixels,
            stokes_weights=weights,
            # view="scanning",
            output_dir=self.outdir,
            lagmax=1000,
            nbin_psd=300,
        )
        estim.apply(data)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()

            import matplotlib.pyplot as plt
            import astropy.io.fits as pf

            obs = data.obs[0]
            det = obs.local_detectors[0]
            fig = plt.figure(figsize=[12, 8])
            ax = fig.add_subplot(1, 1, 1)
            for label in "with_signal", "without_signal":
                fname = os.path.join(
                    self.outdir, f"noise_{label}_{obs.name}_{det}.fits"
                )
                hdulist = pf.open(fname)
                freq, psd = hdulist[2].data.field(0)
                ax.loglog(freq, psd, label=label)
            net = obs.telescope.focalplane["D0A"]["psd_net"]
            net = net.to_value(u.K / u.Hz**0.5)
            ax.axhline(
                net**2,
                label=f"NET = {net:.3f} K" + " / $\sqrt{\mathrm{Hz}}$",
                linestyle="--",
                color="k",
            )
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD [K$^2$ / Hz]")
            ax.legend(loc="best")
            fname = os.path.join(self.outdir, "psds.png")
            fig.savefig(fname)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()
        del data
        return

    def test_noise_estim_model(self):
        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, sample_rate=100.0 * u.Hz, fknee=0.2 * u.Hz)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle="hwp_angle",
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model")
        sim_noise.apply(data)

        # Estimate noise
        estim = ops.NoiseEstim(
            name="estimate_model",
            output_dir=self.outdir,
            out_model="noise_estimate",
            lagmax=1000,
            nbin_psd=128,
            nsum=4,
        )
        estim.apply(data)

        # Compute a 1/f fit to this
        noise_fitter = ops.FitNoiseModel(
            noise_model=estim.out_model,
            out_model="fit_noise_model",
        )
        noise_fitter.apply(data)

        def plot_compare(
            fname, true_freq, true_psd, est_freq, est_psd, fit_freq=None, fit_psd=None
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
            net = ob.telescope.focalplane[det]["psd_net"]
            net = net.to_value(u.K / u.Hz**0.5)
            ax.axhline(
                net**2,
                label=f"NET = {net:.3f} K" + " / $\sqrt{\mathrm{Hz}}$",
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

        for ob in data.obs:
            if ob.comm.group_rank == 0:
                input_model = ob["noise_model"]
                estim_model = ob["noise_estimate"]
                fit_model = ob[noise_fitter.out_model]
                for det in ob.local_detectors:
                    fname = os.path.join(
                        self.outdir, f"estimate_model_{ob.name}_{det}.pdf"
                    )
                    plot_compare(
                        fname,
                        input_model.freq(det),
                        input_model.psd(det),
                        estim_model.freq(det),
                        estim_model.psd(det),
                        fit_freq=fit_model.freq(det),
                        fit_psd=fit_model.psd(det),
                    )
            if ob.comm.comm_group is not None:
                ob.comm.comm_group.barrier()

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Verify that the fit NET is close to the input.
        for ob in data.obs:
            input_model = ob["noise_model"]
            estim_model = ob["noise_estimate"]
            fit_model = ob[noise_fitter.out_model]
            for det in ob.local_detectors:
                np.testing.assert_almost_equal(
                    np.mean(input_model.psd(det)[-5:]).value,
                    np.mean(fit_model.psd(det)[-5:]).value,
                    decimal=3,
                )

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()
        del data
        return
