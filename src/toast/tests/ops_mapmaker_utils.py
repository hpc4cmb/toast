# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from astropy import units as u

from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import rng as rng

from ..noise import Noise

from .. import future_ops as ops

from ..future_ops.sim_tod_noise import sim_noise_timestream

from ._helpers import create_outdir, create_satellite_data


class MapmakerUtilsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_hits(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", create_dist="pixel_dist"
        )
        pointing.apply(data)

        build_hits = ops.BuildHitMap(pixel_dist="pixel_dist")
        hits = build_hits.apply(data)

        del data
        return

    def test_inv_cov(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", create_dist="pixel_dist"
        )
        pointing.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Construct a correlated analytic noise model for the detectors for each
        # observation.
        for ob in data.obs:
            nse = ob[default_model.noise_model]
            corr_freqs = {
                "noise_{}".format(i): nse.freq(x)
                for i, x in enumerate(ob.local_detectors)
            }
            corr_psds = {
                "noise_{}".format(i): nse.psd(x)
                for i, x in enumerate(ob.local_detectors)
            }
            corr_indices = {
                "noise_{}".format(i): 100 + i for i, x in enumerate(ob.local_detectors)
            }
            corr_mix = dict()
            for i, x in enumerate(ob.local_detectors):
                dmix = np.random.uniform(
                    low=-1.0, high=1.0, size=len(ob.local_detectors)
                )
                corr_mix[x] = {
                    "noise_{}".format(y): dmix[y]
                    for y in range(len(ob.local_detectors))
                }
            ob["noise_model_corr"] = Noise(
                detectors=ob.local_detectors,
                freqs=corr_freqs,
                psds=corr_psds,
                mixmatrix=corr_mix,
                indices=corr_indices,
            )

        # Simulate noise using both models

        sim_noise = ops.SimNoise(noise_model="noise_model", out="noise")
        sim_noise.apply(data)

        sim_noise_corr = ops.SimNoise(noise_model="noise_model_corr", out="noise_corr")
        sim_noise_corr.apply(data)

        # Build an inverse covariance from both

        build_invnpp = ops.BuildInverseCovariance(
            pixel_dist="pixel_dist", noise_model="noise_model"
        )
        invnpp = build_invnpp.apply(data)

        build_invnpp_corr = ops.BuildInverseCovariance(
            pixel_dist="pixel_dist", noise_model="noise_model_corr"
        )
        invnpp_corr = build_invnpp_corr.apply(data)

        del data
        return

    def test_zmap(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", create_dist="pixel_dist"
        )
        pointing.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Construct a correlated analytic noise model for the detectors for each
        # observation.
        for ob in data.obs:
            nse = ob[default_model.noise_model]
            corr_freqs = {
                "noise_{}".format(i): nse.freq(x)
                for i, x in enumerate(ob.local_detectors)
            }
            corr_psds = {
                "noise_{}".format(i): nse.psd(x)
                for i, x in enumerate(ob.local_detectors)
            }
            corr_indices = {
                "noise_{}".format(i): 100 + i for i, x in enumerate(ob.local_detectors)
            }
            corr_mix = dict()
            for i, x in enumerate(ob.local_detectors):
                dmix = np.random.uniform(
                    low=-1.0, high=1.0, size=len(ob.local_detectors)
                )
                corr_mix[x] = {
                    "noise_{}".format(y): dmix[y]
                    for y in range(len(ob.local_detectors))
                }
            ob["noise_model_corr"] = Noise(
                detectors=ob.local_detectors,
                freqs=corr_freqs,
                psds=corr_psds,
                mixmatrix=corr_mix,
                indices=corr_indices,
            )

        # Simulate noise using both models

        sim_noise = ops.SimNoise(noise_model="noise_model", out="noise")
        sim_noise.apply(data)

        sim_noise_corr = ops.SimNoise(noise_model="noise_model_corr", out="noise_corr")
        sim_noise_corr.apply(data)

        # Build a noise weighted map from both

        build_zmap = ops.BuildNoiseWeighted(
            pixel_dist="pixel_dist", noise_model="noise_model", det_data="noise"
        )
        zmap = build_zmap.apply(data)

        build_zmap_corr = ops.BuildNoiseWeighted(
            pixel_dist="pixel_dist",
            noise_model="noise_model_corr",
            det_data="noise_corr",
        )
        zmap_corr = build_zmap_corr.apply(data)

        del data
        return
