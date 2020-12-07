# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

from .mpi import MPITestCase

from .. import ops as ops

from ..pixels import PixelData

from ._helpers import create_outdir, create_satellite_data


class ScanMapTest(MPITestCase):
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

        # Manual check of the projection of map values to timestream

        map_data = data["fake_map"]
        for ob in data.obs:
            for det in ob.local_detectors:
                wt = ob.detdata[pointing.weights][det]
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata[pointing.pixels][det]
                )
                for i in range(ob.n_local_samples):
                    if local_pix[i] < 0:
                        continue
                    val = 0.0
                    for j in range(3):
                        val += wt[i, j] * map_data.data[local_sm[i], local_pix[i], j]
                    np.testing.assert_almost_equal(val, ob.detdata["signal"][det, i])

        del data
        return
