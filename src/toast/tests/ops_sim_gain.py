# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import ops as ops
from .. import rng

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ..covariance import covariance_apply
from ._helpers import create_outdir, create_satellite_data,create_satellite_data_big, create_fake_sky


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

        drifter  = ops.GainDrifter(det_data=key, drift_mode="linear_drift")
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


    def test_thermal_drift(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm, )

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

        drifter  = ops.GainDrifter(det_data=key, drift_mode="thermal_drift")
        drifter.exec(data)



    def test_slow_drift(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm, )

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

        drifter  = ops.GainDrifter(det_data=key, drift_mode="slow_drift")
        drifter.exec(data)

    def test_slow_drift_commonmode(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm, )
        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nside=16,
            nest=False,
            mode="I",
            detector_pointing=detpointing,
        )
        # Generate timestreams
        key = "signal"
        sim_dipole = ops.SimDipole(det_data=key, mode="solar", coord="G")
        sim_dipole.exec(data)

        # Build the covariance and hits
        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pointing=pointing,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        # Set up binned map

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key ,
            pointing=pointing,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner.apply(data)


        oldmap = data[binner.binned][0].copy()

        drifter  = ops.GainDrifter(det_data=key, drift_mode="slow_drift",
                                        detector_mismatch=0. )
        drifter.exec(data)

        binner2 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key,
            pointing=pointing,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner2.apply(data)

        newmap = data[binner2.binned][0].copy()


        np.testing.assert_almost_equal(oldmap,newmap  , decimal=np.log10(drifter.sigma_drift)  )
