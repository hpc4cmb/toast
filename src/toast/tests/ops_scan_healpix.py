# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

from .mpi import MPITestCase

from .. import ops as ops

from ..observation import default_names as obs_names

from ..pixels import PixelData

from ..pixels_io import write_healpix_fits

from ._helpers import create_outdir, create_satellite_data, create_fake_sky


class ScanHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_healpix(self):
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
            hwp_angle=obs_names.hwp_angle,
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
            det_data=obs_names.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Run the scanning from the file

        scan_hpix = ops.ScanHealpix(
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
                    ob.detdata["test"][det], ob.detdata[obs_names.det_data][det]
                )

        del data
        return
