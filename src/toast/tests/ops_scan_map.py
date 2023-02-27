# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..accelerator import ImplementationType
from ..observation import default_values as defaults
from ..pixels import PixelData
from ._helpers import close_data, create_fake_sky, create_outdir, create_satellite_data
from .mpi import MPITestCase


class ScanMapTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_scan(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")
        map_data = data["fake_map"]

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )

        for impl in [ImplementationType.DEFAULT, ImplementationType.NUMPY]:
            scanner.zero = True
            scanner.kernel_implementation = impl
            scanner.apply(data)

            # Manual check of the projection of map values to timestream
            for ob in data.obs:
                for det in ob.local_detectors:
                    wt = ob.detdata[weights.weights][det]
                    local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                        ob.detdata[pixels.pixels][det]
                    )
                    for i in range(ob.n_local_samples):
                        if local_pix[i] < 0:
                            continue
                        val = 0.0
                        for j in range(3):
                            val += (
                                wt[i, j] * map_data.data[local_sm[i], local_pix[i], j]
                            )
                        np.testing.assert_almost_equal(
                            val, ob.detdata[defaults.det_data][det, i]
                        )

        close_data(data)

    def test_scan_add_subtract(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams twice, adding once and then subtracting.  Also test
        # zero option.

        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        rms = list()
        for ob in data.obs:
            for det in ob.local_detectors:
                rms.append(np.std(ob.detdata[defaults.det_data][det]))
        rms = np.array(rms)

        scanner.zero = True
        scanner.apply(data)

        trms = list()
        for ob in data.obs:
            for det in ob.local_detectors:
                trms.append(np.std(ob.detdata[defaults.det_data][det]))
        trms = np.array(trms)

        np.testing.assert_equal(trms, rms)

        scanner.zero = False
        scanner.subtract = True
        scanner.apply(data)

        trms = list()
        for ob in data.obs:
            for det in ob.local_detectors:
                trms.append(np.std(ob.detdata[defaults.det_data][det]))
        trms = np.array(trms)

        np.testing.assert_equal(trms, np.zeros_like(trms))

        close_data(data)

    def test_mask(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Generate a mask
        data["fake_mask"] = PixelData(data["pixel_dist"], np.uint8, n_value=1)
        small_vals = data["fake_map"].data[:, :, 0] < 10.0
        # print("{} map vals masked".format(np.sum(small_vals)))
        data["fake_mask"].data[small_vals] = 1

        # Scan mask into flags
        scanner = ops.ScanMask(
            det_flags="mask_flags",
            det_flags_value=1,
            pixels=pixels.pixels,
            mask_key="fake_mask",
            mask_bits=1,
        )
        scanner.apply(data)

        # Manual check of the values

        mask_data = data["fake_mask"]
        for ob in data.obs:
            for det in ob.local_detectors:
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata[pixels.pixels][det]
                )
                for i in range(ob.n_local_samples):
                    if local_pix[i] < 0:
                        continue
                    mask_val = mask_data.data[local_sm[i], local_pix[i], 0]
                    if mask_val > 0:
                        self.assertTrue(ob.detdata["mask_flags"][det, i] == 1)
                    else:
                        self.assertTrue(ob.detdata["mask_flags"][det, i] == 0)

        close_data(data)
