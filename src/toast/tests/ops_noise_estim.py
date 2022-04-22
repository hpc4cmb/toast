# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u
from astropy.table import Column

from .mpi import MPITestCase

from ..noise import Noise

from .. import ops as ops

from ..vis import set_matplotlib_backend

from ..pixels import PixelDistribution, PixelData
from ..pixels_io import write_healpix_fits

from .. import qarray as qa

from ._helpers import (
    create_outdir,
    create_satellite_data,
    create_ground_data,
    create_fake_sky,
    fake_flags,
)

from ..observation import default_values as defaults


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
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
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

        del data
        return
