# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column
import healpy as hp

from .. import ops as ops
from ..footprint import footprint_distribution
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_ground_data,
    create_outdir,
    fake_flags,
    create_fake_healpix_map,
)
from .mpi import MPITestCase


class AccumObsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def create_observations(self):
        """Generate observations with boresight pointing and no det data."""
        # Slightly slower than 1 Hz
        hwp_rpm = 59.0

        sample_rate = 60 * u.Hz

        # Create a fake ground observation set for testing.
        # Detector data is not created.
        data = create_ground_data(
            self.comm,
            sample_rate=sample_rate,
            hwp_rpm=hwp_rpm,
            no_det_data=True,
        )
        return data

    def create_input_sky(self, data, pixel_dist, map_key, test_dir):
        """Create the input sky map."""
        if map_key in data:
            msg = f"Generated map '{map_key}' already exists in data"
            raise RuntimeError(msg)
        npix = data[pixel_dist].n_pix
        nside = hp.npix2nside(npix)
        lmax = 3 * nside
        fwhm = 10 * u.arcmin
        data[map_key] = create_fake_healpix_map(
            os.path.join(test_dir, "input_sky.fits"),
            data[pixel_dist],
            fwhm=fwhm,
            lmax=lmax,
        )

    def create_sim_pipeline(self, data, map_key):
        # Create an uncorrelated noise model from focalplane detector properties.
        # We do this outside of the pipeline since it does not generate any
        # timestreams.
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        npix = data[map_key].distribution.n_pix
        nside = hp.npix2nside(npix)

        # Now build up the operators to use in the simulation pipeline
        operators = list()

        # Generic pointing matrix
        detpointing = ops.PointingDetectorSimple(
            shared_flag_mask=0,
            quats="sim_quats",
        )
        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing,
            pixels="sim_pixels",
        )
        operators.append(pixels)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="sim_weights",
        )
        operators.append(weights)

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key=map_key,
        )
        operators.append(scanner)

        # Simulate noise from the default model
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, out=defaults.det_data
        )
        operators.append(sim_noise)

        # The pipeline to run on each obs
        pipe = ops.Pipeline(
            detector_sets=["ALL"],
            operators=operators,
        )
        return pipe

    def add_sim_loader(self, data, pixel_dist, test_dir):
        """Go through all observations and add a loader."""

        # Add to the observations
        for ob in data.obs:
            ob.loader = ops.PipelineLoader(pipeline=pipe)

    def test_accum_sim(self):
        testdir = os.path.join(self.outdir, "accum_sim")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        data = self.create_observations()

        # Generate a full-sky footprint for testing
        nside = 256
        pixel_dist = "pixel_dist"
        data[pixel_dist] = footprint_distribution(
            healpix_nside=nside,
            healpix_nside_submap=16,
        )

        # Create a
        map_key = "input_sky"

        # First simulate the sky map and store it in the data
        self.create_input_sky(data, pixel_dist, map_key, test_dir)

        pipe = self.create_sim_pipeline(data, map_key)

        self.add_sim_loader(data, pixel_dist, testdir)

        # Operator to accumulate our map-domain products with the loader
        detpointing = ops.PointingDetectorSimple(
            shared_flag_mask=0,
            quats="quats",
        )
        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing,
            pixels="pixels",
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="weights",
        )
        accum_obs = ops.AccumulateObservation(
            cache_dir=os.path.join(testdir, "cache"),
            pixel_dist=pixel_dist,
            inverse_covariance="invcov",
            hits="hits",
            zmap="zmap",
            rcond="rcond",
            covariance="cov",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
            obs_pointing=True,
        )
        accum_obs.load_apply(data)

        close_data(data)
