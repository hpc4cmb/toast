# Copyright (c) 2015-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import rng as rng
from ..noise import Noise
from ..ops.sim_tod_noise import sim_noise_timestream
from ..vis import set_matplotlib_backend
from ._helpers import create_outdir, create_satellite_data, close_data
from .mpi import MPITestCase


class CommonModeNoiseTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.oversample = 2
        self.nmc = 100

    def test_model(self):
        # Test the uncorrelated noise generation.
        # Verify that the white noise part of the spectrum is normalized
        # correctly.

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # This is a simulation with the same focalplane for every obs...
        sample_rate = data.obs[0].telescope.focalplane.sample_rate

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        common_mode_model = ops.CommonModeNoise(
            fmin=1e-5 * u.Hz,
            fknee=1.0 * u.Hz,
            alpha=2.0,
            NET=1.0 * u.K / u.Hz**0.5,
        )
        common_mode_model.apply(data)

        return

    def test_sim(self):
        # Test the uncorrelated noise generation.
        # Verify that the white noise part of the spectrum is normalized
        # correctly.

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # This is a simulation with the same focalplane for every obs...
        sample_rate = data.obs[0].telescope.focalplane.sample_rate

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate data with common mode
        sim_noise = ops.SimNoise()
        sim_noise.det_data = "noise_uncorrelated"
        sim_noise.apply(data)

        # Add common mode noise
        common_mode_model = ops.CommonModeNoise(
            fmin=1e-5 * u.Hz,
            fknee=1.0 * u.Hz,
            alpha=2.0,
            NET=1.0 * u.K / u.Hz**0.5,
        )
        common_mode_model.apply(data)

        # Simulate data with common mode
        sim_noise = ops.SimNoise()
        sim_noise.det_data = "noise_correlated"
        sim_noise.apply(data)

        # Compare results
        for ob in data.obs:
            common_mode = None
            for det in ob.local_detectors:
                noise1 = ob.detdata["noise_uncorrelated"][det]
                noise2 = ob.detdata["noise_correlated"][det]
                if common_mode is None:
                    common_mode = noise2 - noise1
                    # Make sure we simulated a common mode
                    assert np.std(common_mode) > 1e-10
                else:
                    common_mode2 = noise2 - noise1
                    # Make sure the common mode agrees
                    np.testing.assert_array_almost_equal(common_mode, common_mode2)

        return
