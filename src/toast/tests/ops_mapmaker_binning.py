# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

from .mpi import MPITestCase

from ..noise import Noise

from .. import ops as ops

from ..pixels import PixelDistribution, PixelData

from ._helpers import create_outdir, create_satellite_data


class MapmakerBinningTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_binned(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise
        sim_noise = ops.SimNoise(noise_model="noise_model", out="noise")
        sim_noise.apply(data)

        # Pointing operator
        pointing = ops.PointingHealpix(nside=64, mode="IQU", hwp_angle="hwp_angle")

        # Build the covariance and hits
        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist", pointing=pointing, noise_model="noise_model"
        )
        cov_and_hits.apply(data)

        # Set up binned map

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data="noise",
            pointing=pointing,
            noise_model="noise_model",
        )
        binner.apply(data)

        binmap = binner.binned

        # # Manual check
        #
        # check_invnpp = PixelData(data["pixel_dist"], np.float64, n_value=6)
        # check_invnpp_corr = PixelData(data["pixel_dist"], np.float64, n_value=6)
        #
        # for ob in data.obs:
        #     noise = ob["noise_model"]
        #     noise_corr = ob["noise_model_corr"]
        #
        #     for det in ob.local_detectors:
        #         detweight = noise.detector_weight(det)
        #         detweight_corr = noise_corr.detector_weight(det)
        #
        #         wt = ob.detdata["weights"][det]
        #         local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
        #             ob.detdata["pixels"][det]
        #         )
        #         for i in range(ob.n_local_samples):
        #             if local_pix[i] < 0:
        #                 continue
        #             off = 0
        #             for j in range(3):
        #                 for k in range(j, 3):
        #                     check_invnpp.data[local_sm[i], local_pix[i], off] += (
        #                         detweight * wt[i, j] * wt[i, k]
        #                     )
        #                     check_invnpp_corr.data[local_sm[i], local_pix[i], off] += (
        #                         detweight_corr * wt[i, j] * wt[i, k]
        #                     )
        #                     off += 1
        #
        # check_invnpp.sync_allreduce()
        # check_invnpp_corr.sync_allreduce()
        #
        # for sm in range(invnpp.distribution.n_local_submap):
        #     for px in range(invnpp.distribution.n_pix_submap):
        #         if invnpp.data[sm, px, 0] != 0:
        #             nt.assert_almost_equal(
        #                 invnpp.data[sm, px], check_invnpp.data[sm, px]
        #             )
        #         if invnpp_corr.data[sm, px, 0] != 0:
        #             nt.assert_almost_equal(
        #                 invnpp_corr.data[sm, px], check_invnpp_corr.data[sm, px]
        #             )
        del data
        return
