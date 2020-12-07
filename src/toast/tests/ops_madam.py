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


class MadamTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def create_fake_sky(self, data, dist_key, map_key):
        dist = data[dist_key]
        pix_data = PixelData(dist, np.float64, n_value=3)
        # Just replicate the fake data across all local submaps
        pix_data.data[:, :, 0] = 100.0 * np.random.uniform(size=dist.n_pix_submap)
        pix_data.data[:, :, 1] = np.random.uniform(size=dist.n_pix_submap)
        pix_data.data[:, :, 2] = np.random.uniform(size=dist.n_pix_submap)
        data[map_key] = pix_data

    def test_scan(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", create_dist="pixel_dist"
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        self.create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

    def test_madam_output(self):
        if not ops.Madam.available:
            print("libmadam not available, skipping tests")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", create_dist="pixel_dist"
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        self.create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out="signal")
        sim_noise.apply(data)

        # Run madam on this

        # Madam assumes constant sample rate- just get it from the noise model for
        # the first detector.
        sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

        pars = {}
        pars["kfirst"] = "T"
        pars["iter_max"] = 100
        pars["base_first"] = 5.0
        pars["fsample"] = sample_rate
        pars["nside_map"] = pointing.nside
        pars["nside_cross"] = pointing.nside
        pars["nside_submap"] = min(8, pointing.nside)
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = self.outdir
        pars["info"] = 0

        # FIXME: add a view here once our test data includes it

        madam = ops.Madam(
            params=pars,
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            pixels_nested=pointing.nest,
            det_out="destriped",
            noise_model="noise_model",
            copy_groups=2,
            purge_det_data=False,
            purge_pointing=True,
        )
        madam.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                # Do some check...
                pass

        del data
        return
