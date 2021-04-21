# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import ops as ops
from .. import rng


from ._helpers import create_outdir, create_satellite_data


class SimGainTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)


    def test_linear_drift(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm, )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)
        old_data = []
        for obs in data.obs:
            old = {}
            for det in obs.local_detectors:
                ref = obs.detdata[key][det]
                old[det] =  (ref).copy()
            old_data.append(old )

        drifter  = ops.GainDrifter(det_data=key, drift_mode="linear")
        drifter.exec(data)
        for obs, old  in zip(data.obs, old_data):
            telescope = obs.telescope.uid
            focalplane = obs.telescope.focalplane
            obsindx = obs.uid

            key1 = (drifter.realization * 4294967296 +
                    telescope * 65536 + drifter.component )
            counter1 = 0
            counter2 = 0
            for det in obs.local_detectors:
                detindx = focalplane[det]["uid"]
                key2 = obsindx * 4294967296 + detindx
                rngdata = rng.random(
                    1,
                    sampler="gaussian",
                    key=(key1, key2),
                    counter=(counter1, counter2),
                )
                gf2 = 1 + rngdata[0] *  drifter.sigma_drift

                gf1 = (obs.detdata[key][det]/ old[det] )[-1]
                #assert whether the two values gf2 and gf1  are the same
                #within 1sigma of the distribution
                np.testing.assert_almost_equal(gf1,gf2 , decimal=np.log10(drifter.sigma_drift) -1 )
