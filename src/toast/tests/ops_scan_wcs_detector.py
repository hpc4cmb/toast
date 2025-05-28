# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..pixels import PixelData
from .helpers import (close_data, create_fake_mask,
                      create_fake_wcs_scanned_tod, create_outdir,
                      create_satellite_data)
from .mpi import MPITestCase


class ScanWCSDetectorTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)

    def test_wcs_fits(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsWCS(
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

        pixel_names = np.unique(data.obs[0].telescope.focalplane.detector_data["pixel"])

        wcs_file = os.path.join(self.outdir, "fake_fits_{pixel}.fits")
        # Create fake polarized sky signal independently for each pixel
        for i, pixel in enumerate(pixel_names):
            map_key = f"fake_map_{pixel}"
            create_fake_wcs_scanned_tod(
                data,
                pixels,
                weights,
                wcs_file.format(pixel=pixel),
                "pixel_dist",
                map_key=map_key,
                fwhm=30.0 * u.arcmin,
                I_scale=i,  # T-only maps, each scaled different
                Q_scale=0,
                U_scale=0,
                det_data=f"det_data_{pixel}",
            )

        # Run the scanning from the file. Each pixel will scan a different file
        scan_wcs = ops.ScanWCSDetectorMap(
            file=wcs_file,
            det_data="test",
            focalplane_keys="pixel,psi_pol",
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_wcs.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                pixel = data.obs[0].telescope.focalplane[det]["pixel"]
                det_data = f"det_data_{pixel}"
                try:
                    np.testing.assert_almost_equal(
                        ob.detdata["test"][det],
                        ob.detdata[det_data][det],
                        decimal=5,
                    )
                except:
                    import pdb

                    pdb.set_trace()

        close_data(data)
