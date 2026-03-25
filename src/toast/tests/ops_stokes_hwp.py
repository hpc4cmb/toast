# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..instrument_coords import quat_to_xieta
from ..instrument_sim import plot_focalplane
from ..observation import default_values as defaults
from ..pixels import PixelData
from .helpers import (
    close_data,
    create_outdir,
    create_ground_data,
    create_fake_healpix_scanned_tod,
)
from .mpi import MPITestCase


class StokesWeightsHWPTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.nside = 64

    def create_test_data(self, testdir):
        # Slightly slower than 1 Hz
        hwp_rpm = 59.0
        sample_rate = 60 * u.Hz

        # Create a fake ground observations set for testing
        data = create_ground_data(self.comm, sample_rate=sample_rate, hwp_rpm=hwp_rpm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Generic pointing matrix for sampling from the map
        detpointing = ops.PointingDetectorSimple(shared_flag_mask=0)
        pixels = ops.PixelsHealpix(
            nside=self.nside,
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

        # Clean up simulated pointing used for the map scanning
        ops.Delete(detdata=[pixels.pixels, weights.weights]).apply(data)
        return data


    def test_nominal(self):
        data = self.create_test_data(self.outdir)

        # Pointing model
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeightsHWP(
            mode="nominal",
            detector_pointing=detpointing,
        )

        # Binned mapmaking
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
        )

        mapper = ops.MapMaker(
            name="mapmaker",
            det_data=defaults.det_data,
            binning=binner,
            map_rcond_threshold=1.0e-1,
            write_hits=True,
            write_map=True,
            write_noiseweighted_map=True,
            write_invcov=True,
            write_rcond=True,
            output_dir=self.outdir,
        )

        # Make the map
        mapper.apply(data)

        close_data(data)
