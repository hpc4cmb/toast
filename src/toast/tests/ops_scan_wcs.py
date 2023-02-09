# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..pixels import PixelData
from ..pixels_io_wcs import write_wcs_fits
from ._helpers import (
    close_data,
    create_fake_mask,
    create_fake_sky,
    create_ground_data,
    create_outdir,
)
from .mpi import MPITestCase


class ScanWCSTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_wcs_fits(self):
        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
        )

        # Pixelization
        pixels = ops.PixelsWCS(
            projection="CAR",
            resolution=(0.05 * u.degree, 0.05 * u.degree),
            auto_bounds=True,
            detector_pointing=detpointing_radec,
            create_dist="pixel_dist",
            use_astropy=True,
        )
        pixels.apply(data)
        pixels.create_dist = None

        # Stokes weights
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
        )
        weights.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Write this to a file
        input_file = os.path.join(self.outdir, "fake.fits")
        write_wcs_fits(data["fake_map"], input_file)

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Run the scanning from the file
        scan_wcs = ops.ScanWCSMap(
            file=input_file,
            det_data="test",
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_wcs.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_almost_equal(
                    ob.detdata["test"][det], ob.detdata[defaults.det_data][det]
                )

        close_data(data)

    def test_wcs_mask(self):
        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
        )

        # Pixelization
        pixels = ops.PixelsWCS(
            projection="CAR",
            resolution=(0.05 * u.degree, 0.05 * u.degree),
            auto_bounds=True,
            detector_pointing=detpointing_radec,
            create_dist="pixel_dist",
            use_astropy=True,
        )
        pixels.apply(data)
        pixels.create_dist = None

        # Stokes weights
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
        )
        weights.apply(data)

        # Create fake mask pixel values locally
        create_fake_mask(data, "pixel_dist", "fake_mask")

        # Write this to a file
        input_file = os.path.join(self.outdir, "fake_mask.fits")
        write_wcs_fits(data["fake_mask"], input_file)

        # Scan map into timestreams
        scanner = ops.ScanMask(
            det_flags=defaults.det_flags,
            det_flags_mask=defaults.det_mask_invalid,
            pixels=pixels.pixels,
            mask_key="fake_mask",
        )
        scanner.apply(data)

        # Run the scanning from the file
        scan_wcs = ops.ScanWCSMask(
            file=input_file,
            det_flags="test_flags",
            det_flags_value=defaults.det_mask_invalid,
            pixel_pointing=pixels,
        )
        scan_wcs.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_equal(
                    ob.detdata["test_flags"][det], ob.detdata[defaults.det_flags][det]
                )

        close_data(data)
