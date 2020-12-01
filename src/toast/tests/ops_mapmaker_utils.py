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


class MapmakerUtilsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        self.mix_coeff = np.random.uniform(low=-1.0, high=1.0, size=1000)

    def create_corr_noise(self, dets, nse):
        corr_freqs = {"noise_{}".format(i): nse.freq(x) for i, x in enumerate(dets)}
        corr_psds = {"noise_{}".format(i): nse.psd(x) for i, x in enumerate(dets)}
        corr_indices = {"noise_{}".format(i): 100 + i for i, x in enumerate(dets)}
        corr_mix = dict()
        for i, x in enumerate(dets):
            dmix = self.mix_coeff[: len(dets)]
            corr_mix[x] = {"noise_{}".format(y): dmix[y] for y in range(len(dets))}
        return Noise(
            detectors=dets,
            freqs=corr_freqs,
            psds=corr_psds,
            mixmatrix=corr_mix,
            indices=corr_indices,
        )

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

        # Manual check
        check_hits = PixelData(data["pixel_dist"], np.int64, n_value=1)
        for ob in data.obs:
            for det in ob.local_detectors:
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata["pixels"][det]
                )
                for i in range(ob.n_local_samples):
                    if local_pix[i] >= 0:
                        check_hits.data[local_sm[i], local_pix[i], 0] += 1
        check_hits.sync_allreduce()

        nt.assert_equal(hits.data, check_hits.data)

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
            ob["noise_model_corr"] = self.create_corr_noise(ob.local_detectors, nse)

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

        # Manual check

        check_invnpp = PixelData(data["pixel_dist"], np.float64, n_value=6)
        check_invnpp_corr = PixelData(data["pixel_dist"], np.float64, n_value=6)

        for ob in data.obs:
            noise = ob["noise_model"]
            noise_corr = ob["noise_model_corr"]

            for det in ob.local_detectors:
                detweight = noise.detector_weight(det)
                detweight_corr = noise_corr.detector_weight(det)

                wt = ob.detdata["weights"][det]
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata["pixels"][det]
                )
                for i in range(ob.n_local_samples):
                    if local_pix[i] < 0:
                        continue
                    off = 0
                    for j in range(3):
                        for k in range(j, 3):
                            check_invnpp.data[local_sm[i], local_pix[i], off] += (
                                detweight * wt[i, j] * wt[i, k]
                            )
                            check_invnpp_corr.data[local_sm[i], local_pix[i], off] += (
                                detweight_corr * wt[i, j] * wt[i, k]
                            )
                            off += 1

        check_invnpp.sync_allreduce()
        check_invnpp_corr.sync_allreduce()

        for sm in range(invnpp.distribution.n_local_submap):
            for px in range(invnpp.distribution.n_pix_submap):
                if invnpp.data[sm, px, 0] != 0:
                    nt.assert_almost_equal(
                        invnpp.data[sm, px], check_invnpp.data[sm, px]
                    )
                if invnpp_corr.data[sm, px, 0] != 0:
                    nt.assert_almost_equal(
                        invnpp_corr.data[sm, px], check_invnpp_corr.data[sm, px]
                    )
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
            ob["noise_model_corr"] = self.create_corr_noise(ob.local_detectors, nse)

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

        # Manual check

        check_zmap = PixelData(data["pixel_dist"], np.float64, n_value=3)
        check_zmap_corr = PixelData(data["pixel_dist"], np.float64, n_value=3)

        for ob in data.obs:
            noise = ob["noise_model"]
            noise_corr = ob["noise_model_corr"]
            for det in ob.local_detectors:
                wt = ob.detdata["weights"][det]
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata["pixels"][det]
                )

                for i in range(ob.n_local_samples):
                    if local_pix[i] < 0:
                        continue
                    for j in range(3):
                        check_zmap.data[local_sm[i], local_pix[i], j] += (
                            noise.detector_weight(det)
                            * ob.detdata["noise"][det, i]
                            * wt[i, j]
                        )
                        check_zmap_corr.data[local_sm[i], local_pix[i], j] += (
                            noise_corr.detector_weight(det)
                            * ob.detdata["noise_corr"][det, i]
                            * wt[i, j]
                        )

        check_zmap.sync_allreduce()
        check_zmap_corr.sync_allreduce()

        np.testing.assert_almost_equal(zmap.data, check_zmap.data)
        np.testing.assert_almost_equal(zmap_corr.data, check_zmap_corr.data)

        del data
        return
