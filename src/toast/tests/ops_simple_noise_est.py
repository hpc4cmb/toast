# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
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
    create_satellite_data,
    create_outdir,
)
from .mpi import MPITestCase

try:
    import finufft

    have_finufft = True
except ImportError:
    have_finufft = False


def plot_noise_compare(
    fname,
    net,
    true_freq,
    true_psd,
    est_freq,
    est_psd,
    fit_freq=None,
    fit_psd=None,
    tod=None,
):
    set_matplotlib_backend()
    import matplotlib.pyplot as plt

    print(f"true: {true_freq}, {true_psd}")
    print(f"est: {est_freq}, {est_psd}")

    if tod is None:
        n_plot = 1
    else:
        n_plot = 2

    fig = plt.figure(figsize=[12, 8 * n_plot])
    ax = fig.add_subplot(n_plot, 1, 1)
    ax.loglog(
        est_freq.to_value(u.Hz),
        est_psd.to_value(u.K**2 * u.s),
        color="red",
        marker="x",
        label="Estimated",
    )
    ax.loglog(
        true_freq.to_value(u.Hz),
        true_psd.to_value(u.K**2 * u.s),
        marker="o",
        color="black",
        label="Input",
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
    ax.set_xlim(0.9 * est_freq[0].to_value(u.Hz), 1.1 * est_freq[-1].to_value(u.Hz))
    ax.set_ylim(
        0.9 * np.amin(est_psd.to_value(u.K**2 * u.s)),
        1.1 * np.amax(est_psd.to_value(u.K**2 * u.s)),
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [K$^2$ / Hz]")
    ax.legend(loc="best")
    if tod is not None:
        ax = fig.add_subplot(n_plot, 1, 2)
        ax.plot(np.arange(len(tod)), tod, color="black", label="Input TOD")
        ax.set_xlabel("TOD Sample")
        ax.set_ylabel("TOD Value")

    fig.savefig(fname)
    plt.close()


class SimpleNoiseEstTest(MPITestCase):
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

    def test_basic(self):
        if not have_finufft:
            print("Skipping SimpleNoiseEstim tests, finufft not importable", flush=True)
            return
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm,
            single_group=True,
            sample_rate=100.0 * u.Hz,
            obs_time=10.0 * u.minute,
        )

        testdir = os.path.join(self.outdir, "basic")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model")
        sim_noise.apply(data)

        # Estimate noise

        estim = ops.SimpleNoiseEstim(
            noise_model="est_noise_model",
            binned=True,
        )
        estim.apply(data)

        std_estim = ops.NoiseEstim(
            out_model="std_noise_model",
            lagmax=20000,
            nbin_psd=32,
        )
        std_estim.apply(data)

        if data.comm.group_rank == 0 and self.make_plots:
            first_ob = data.obs[0]
            input = first_ob["noise_model"]
            est = first_ob["est_noise_model"]
            stdest = first_ob["std_noise_model"]
            for det in first_ob.select_local_detectors(
                flagmask=defaults.det_mask_invalid
            ):
                out_plot = os.path.join(testdir, f"psd_{det}.png")
                net = input.NET(det)
                plot_noise_compare(
                    out_plot,
                    net,
                    input.freq(det),
                    input.psd(det),
                    est.freq(det),
                    est.psd(det),
                    # tod=first_ob.detdata["signal"],
                )
                out_plot = os.path.join(testdir, f"std_psd_{det}.png")
                plot_noise_compare(
                    out_plot,
                    net,
                    input.freq(det),
                    input.psd(det),
                    stdest.freq(det),
                    stdest.psd(det),
                    # tod=first_ob.detdata["signal"],
                )

        close_data(data)

    def test_full(self):
        if not have_finufft:
            print("Skipping SimpleNoiseEstim tests, finufft not importable", flush=True)
            return
        # Create a fake satellite data set for testing
        data = create_ground_data(
            self.comm,
            single_group=True,
        )

        testdir = os.path.join(self.outdir, "full")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        nside = 128

        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing_radec,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
        )

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(noise_model=default_model.noise_model)
        sim_noise.apply(data)

        # Add fake sky
        sky_file = os.path.join(testdir, "fake_sky.fits")
        map_key = "fake_map"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights,
            sky_file,
            "pixel_dist",
            map_key=map_key,
            fwhm=1.0 * u.deg,
            lmax=3 * nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Simulate atmosphere signal and accumulate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            zmax=200 * u.m,
        )
        sim_atm.apply(data)

        # Demodulate

        demod_weights_in = ops.StokesWeights(
            weights="demod_weights_in",
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_azel,
        )

        demod_weights = ops.StokesWeightsDemod(
            detector_pointing_in=detpointing_azel,
            detector_pointing_out=detpointing_radec,
            mode="IQU",
        )

        downsample = 3
        demod = ops.Demodulate(
            stokes_weights=demod_weights_in,
            nskip=downsample,
            in_place=True,
            mode="IQU",
        )
        demod.apply(data)

        # Estimate noise

        estim = ops.SimpleNoiseEstim(
            noise_model="est_noise_model",
            binned=True,
        )
        estim.apply(data)

        std_estim = ops.NoiseEstim(
            out_model="std_noise_model",
            lagmax=20000,
            nbin_psd=32,
        )
        std_estim.apply(data)

        if data.comm.group_rank == 0 and self.make_plots:
            first_ob = data.obs[0]
            input = first_ob["noise_model"]
            est = first_ob["est_noise_model"]
            stdest = first_ob["std_noise_model"]
            for det in first_ob.select_local_detectors(
                flagmask=defaults.det_mask_invalid
            ):
                out_plot = os.path.join(testdir, f"psd_{det}.png")
                net = np.sqrt(1.0 * u.s / input.detector_weight(det))
                plot_noise_compare(
                    out_plot,
                    net,
                    input.freq(det),
                    input.psd(det),
                    est.freq(det),
                    est.psd(det),
                    # tod=first_ob.detdata["signal"],
                )
                out_plot = os.path.join(testdir, f"std_psd_{det}.png")
                plot_noise_compare(
                    out_plot,
                    net,
                    input.freq(det),
                    input.psd(det),
                    stdest.freq(det),
                    stdest.psd(det),
                    # tod=first_ob.detdata["signal"],
                )

        close_data(data)
