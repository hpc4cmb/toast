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
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_fake_sky,
    create_ground_data,
    create_outdir,
    create_satellite_data,
    fake_flags,
)
from .mpi import MPITestCase

XAXIS, YAXIS, ZAXIS = np.eye(3)


def plot_noise_estim_compare(
    fname, net, true_freq, true_psd, est_freq, est_psd, fit_freq=None, fit_psd=None
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


class NoiseEstimTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_algorithm(self):
        if self.comm is None or self.comm.rank == 0:
            set_matplotlib_backend()

            import matplotlib.pyplot as plt

            # Test fit on a particularly bad 1/f spectrum.

            test_freq = [
                np.array(
                    [
                        0.0610297,
                        0.1220594,
                        0.1830891,
                        0.2441188,
                        0.3051485,
                        0.3661782,
                        0.4272079,
                        0.4882376,
                        0.57978215,
                        0.6713267,
                        0.73235641,
                        0.82390096,
                        0.94596036,
                        1.06801976,
                        1.19007916,
                        1.34265341,
                        1.52574251,
                        1.70883161,
                        1.92243556,
                        2.19706922,
                        2.50221772,
                        2.80736622,
                        3.14302957,
                        3.53972262,
                        3.99744538,
                        4.51619783,
                        5.12649484,
                        5.79782154,
                        6.53017795,
                        7.3540789,
                        8.30003926,
                        9.36805901,
                        10.55813817,
                        11.93130643,
                        13.48756379,
                        15.22691026,
                        17.17986067,
                        19.37692988,
                        21.84863275,
                        24.65599897,
                        27.82954339,
                        31.43029572,
                        35.4887708,
                        40.03548348,
                        45.16197831,
                        50.95979985,
                        57.52049265,
                    ]
                )
                * u.Hz,
                np.array(
                    [
                        0.10005003,
                        0.15007504,
                        0.20010005,
                        0.25012506,
                        0.30015008,
                        0.35017509,
                        0.4002001,
                        0.45022511,
                        0.50025013,
                        0.57528764,
                        0.65032516,
                        0.72536268,
                        0.82541271,
                        0.92546273,
                        1.02551276,
                        1.15057529,
                        1.30065033,
                        1.47573787,
                        1.67583792,
                        1.87593797,
                        2.10105053,
                        2.37618809,
                        2.67633817,
                        3.02651326,
                        3.42671336,
                        3.85192596,
                        4.32716358,
                        4.87743872,
                        5.50275138,
                        6.20310155,
                        7.00350175,
                        7.87893947,
                        8.85442721,
                        9.97998999,
                        11.25562781,
                        12.68134067,
                        14.28214107,
                        16.10805403,
                        18.15907954,
                        20.46023012,
                        23.06153077,
                        25.987994,
                        29.28964482,
                        32.99149575,
                        37.16858429,
                        41.89594797,
                        47.1985993,
                    ]
                )
                * u.Hz,
            ]

            test_psd = [
                np.array(
                    [
                        6.86488638e05,
                        1.23862278e06,
                        9.26303299e05,
                        3.60721675e05,
                        1.63842724e05,
                        1.26530158e05,
                        1.00421640e05,
                        6.82068600e04,
                        2.88951257e04,
                        1.24890704e04,
                        6.66632965e03,
                        1.92539118e03,
                        2.00925747e02,
                        1.83700849e02,
                        3.63300537e02,
                        6.50880497e02,
                        5.77821197e02,
                        2.56325420e02,
                        3.37924769e01,
                        2.50135085e01,
                        6.67960884e01,
                        2.61745286e01,
                        4.42448813e00,
                        1.63095757e01,
                        3.90120276e00,
                        5.71014135e00,
                        1.73051847e00,
                        2.44029925e00,
                        1.26352827e00,
                        7.20185443e-01,
                        5.10883413e-01,
                        3.67615781e-01,
                        2.45368711e-01,
                        1.78405279e-01,
                        1.42542399e-01,
                        1.04279944e-01,
                        7.37068316e-02,
                        5.71334305e-02,
                        4.65681100e-02,
                        3.55492571e-02,
                        2.90944455e-02,
                        2.37740527e-02,
                        1.97769160e-02,
                        1.68633739e-02,
                        1.46859944e-02,
                        1.32646366e-02,
                        1.24911487e-02,
                    ]
                )
                * u.K**2
                * u.s,
                np.array(
                    [
                        0.11589497,
                        0.12278325,
                        0.07503257,
                        0.04684548,
                        0.04663187,
                        0.0444642,
                        0.03609387,
                        0.0264311,
                        0.02011643,
                        0.01928994,
                        0.02383489,
                        0.02487129,
                        0.01966836,
                        0.01863879,
                        0.01781315,
                        0.01567394,
                        0.01289552,
                        0.01161635,
                        0.00844275,
                        0.00942161,
                        0.00850452,
                        0.00756728,
                        0.00720028,
                        0.00612499,
                        0.00716753,
                        0.00624518,
                        0.00550925,
                        0.00429895,
                        0.00512276,
                        0.00497993,
                        0.00439096,
                        0.00398326,
                        0.00376943,
                        0.00372136,
                        0.00347776,
                        0.00307749,
                        0.00342084,
                        0.00339543,
                        0.00319416,
                        0.00300875,
                        0.00314169,
                        0.00284596,
                        0.00296397,
                        0.00277,
                        0.00281466,
                        0.00281973,
                        0.00274093,
                    ]
                )
                * u.K**2
                * u.s,
            ]

            fitter = ops.FitNoiseModel()

            for case, (input_freq, input_psd) in enumerate(zip(test_freq, test_psd)):
                fit = fitter._fit_log_psd(input_freq, input_psd)
                result = fit["fit_result"]
                print(f"result solution = {result.x}")
                print(f"result cost = {result.cost}")
                print(f"result fun = {result.fun}")
                print(f"result nfev = {result.nfev}")
                print(f"result njev = {result.njev}")
                print(f"result status = {result.status}")
                print(f"result message = {result.message}")
                print(f"result status = {result.status}")
                print(f"result active_mask = {result.active_mask}")
                fit_data = fitter._evaluate_model(
                    input_freq,
                    fit["fmin"],
                    fit["NET"],
                    fit["fknee"],
                    fit["alpha"],
                )

                fig = plt.figure(figsize=[12, 8])
                ax = fig.add_subplot(1, 1, 1)
                ax.loglog(
                    input_freq.to_value(u.Hz),
                    input_psd.to_value(u.K**2 * u.s),
                    color="black",
                    label="Input PSD",
                )
                ax.loglog(
                    input_freq.to_value(u.Hz),
                    fit_data.to_value(u.K**2 * u.s),
                    color="red",
                    label="Final (Fit or Original)",
                )
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("PSD [K$^2$ / Hz]")
                ax.legend(loc="best")
                outfile = os.path.join(self.outdir, f"noise_fit_bad_{case}.pdf")
                fig.savefig(outfile)

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

            import astropy.io.fits as pf
            import matplotlib.pyplot as plt

            obs = data.obs[0]
            det = obs.local_detectors[0]
            fig = plt.figure(figsize=[12, 8])
            ax = fig.add_subplot(1, 1, 1)
            for label in "with_signal", "without_signal":
                fname = os.path.join(
                    self.outdir, f"noise_{label}_{obs.name}_{det}.fits"
                )
                try:
                    hdulist = pf.open(fname)
                    freq, psd = hdulist[2].data.field(0)
                    ax.loglog(freq, psd, label=label)
                except Exception:
                    print(f"File {fname} does not exist.  Skipping")
            net = obs.telescope.focalplane["D0A-150"]["psd_net"]
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

        close_data(data)

    def test_noise_estim_model(self):
        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, sample_rate=100.0 * u.Hz, fknee=5.0 * u.Hz)

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

        for ob in data.obs:
            if ob.comm.group_rank == 0:
                input_model = ob["noise_model"]
                estim_model = ob["noise_estimate"]
                fit_model = ob[noise_fitter.out_model]
                for det in ob.local_detectors:
                    fname = os.path.join(
                        self.outdir, f"estimate_model_{ob.name}_{det}.pdf"
                    )
                    plot_noise_estim_compare(
                        fname,
                        input_model.NET(det),
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

        close_data(data)
