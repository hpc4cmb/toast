# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..data import Data
from ..mpi import MPI
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class SignalDiffNoiseTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_diff_noise(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm, sample_rate=100 * u.Hz, fknee=1e-3 * u.Hz)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight="boresight_azel", quats="quats_azel"
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="analytic_noise")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(noise_model="analytic_noise")
        sim_noise.apply(data)

        # Estimate noise using sample differences
        noise_estim = ops.SignalDiffNoiseModel(noise_model="measured_noise")
        noise_estim.apply(data)

        # Check that the last frequency bin of analytic and measured
        # noise agrees

        dets = data.obs[0].local_detectors
        det = dets[0]
        # freq0 = data.obs[0]["analytic_noise"].freq(det)
        psd0 = data.obs[0]["analytic_noise"].psd(det)
        # freq1 = data.obs[0]["measured_noise"].freq(det)
        psd1 = data.obs[0]["measured_noise"].psd(det)

        assert np.abs((psd0[-1] - psd1[-1]) / psd0[-1]) < 1e-2

        close_data(data)
