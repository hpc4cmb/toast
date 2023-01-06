# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .. import ops
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class OpGainScramblerTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_scrambler(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)

        # Record the old RMS
        old_rms = []
        for obs in data.obs:
            orms = {}
            for det in obs.local_detectors:
                ref = obs.detdata[key][det]
                orms[det] = np.std(ref)
            old_rms.append(orms)

        # Scramble the timestreams

        op = ops.GainScrambler(det_data=key, center=2, sigma=1e-6)
        op.exec(data)

        # Ensure RMS changes for the implicated detectors

        for obs, orms in zip(data.obs, old_rms):
            for det in obs.local_detectors:
                ref = obs.detdata[key][det]
                rms = np.std(ref)
                old = orms[det]
                if np.abs(rms / old) - 2 > 1e-3:
                    raise RuntimeError(f"det {det} old rms = {old}, new rms = {rms}")

        close_data(data)
