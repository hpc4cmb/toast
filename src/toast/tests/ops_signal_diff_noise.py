# Copyright (c) 2024-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u

from ..mpi import MPI
from ..observation import default_values as defaults
from .. import ops as ops
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class SignalDiffNoiseTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_diff_noise(self):
        # Create fake observing of a small patch
        data = create_ground_data(
            self.comm, sample_rate=100 * u.Hz, fknee=1e-3 * u.Hz, schedule_hours=1
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
        fail = 0
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                psd0 = ob["analytic_noise"].psd(det)
                psd1 = ob["measured_noise"].psd(det)
                if np.abs((psd0[-1] - psd1[-1]) / psd0[-1]) > 5e-2:
                    msg = f"{ob.name}:{det} final PSD value disagrees "
                    msg += f"({psd0[-3:-1]} != {psd1[-3:-1]})"
                    print(msg, flush=True)
                    fail = 1

        if data.comm.comm_world is not None:
            data.comm.comm_world.allreduce(fail, op=MPI.SUM)
        self.assertTrue(fail == 0)

        close_data(data)
