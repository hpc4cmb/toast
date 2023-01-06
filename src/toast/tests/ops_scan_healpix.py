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
from ..pixels_io_healpix import write_healpix_fits, write_healpix_hdf5
from ._helpers import (
    close_data,
    create_fake_mask,
    create_fake_sky,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class ScanHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_healpix_fits(self):
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

        # Write this to a file
        hpix_file = os.path.join(self.outdir, "fake.fits")
        write_healpix_fits(data["fake_map"], hpix_file, nest=pixels.nest)

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Run the scanning from the file

        scan_hpix = ops.ScanHealpixMap(
            file=hpix_file,
            det_data="test",
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_almost_equal(
                    ob.detdata["test"][det],
                    ob.detdata[defaults.det_data][det],
                    decimal=5,
                )

        close_data(data)

    def test_healpix_mask(self):
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

        # Create fake mask pixel values locally
        create_fake_mask(data, "pixel_dist", "fake_mask")

        # Write this to a file
        hpix_file = os.path.join(self.outdir, "fake_mask.fits")
        write_healpix_fits(data["fake_mask"], hpix_file, nest=pixels.nest)

        # Scan map into timestreams
        scanner = ops.ScanMask(
            det_flags=defaults.det_flags,
            det_flags_mask=defaults.det_mask_invalid,
            pixels=pixels.pixels,
            mask_key="fake_mask",
        )
        scanner.apply(data)

        # Run the scanning from the file

        scan_hpix = ops.ScanHealpixMask(
            file=hpix_file,
            det_flags="test_flags",
            det_flags_value=defaults.det_mask_invalid,
            pixel_pointing=pixels,
        )
        scan_hpix.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_equal(
                    ob.detdata["test_flags"][det], ob.detdata[defaults.det_flags][det]
                )

        close_data(data)

    def test_healpix_hdf5(self):
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

        # Write this to a file
        hpix_file = os.path.join(self.outdir, "fake.h5")
        write_healpix_hdf5(data["fake_map"], hpix_file, nest=pixels.nest)

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Run the scanning from the file

        scan_hpix = ops.ScanHealpixMap(
            file=hpix_file,
            det_data="test",
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_almost_equal(
                    ob.detdata["test"][det],
                    ob.detdata[defaults.det_data][det],
                    decimal=5,
                )

        close_data(data)
