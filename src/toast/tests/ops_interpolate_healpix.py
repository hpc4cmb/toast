# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from .helpers import (
    close_data,
    create_fake_healpix_scanned_tod,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class InterpolateHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)

    def test_interpolate(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=256,
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

        # Create fake polarized sky signal
        hpix_file = os.path.join(self.outdir, "fake.fits")
        map_key = "fake_map"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights,
            hpix_file,
            "pixel_dist",
            map_key=map_key,
            fwhm=30.0 * u.degree,
            lmax=3 * pixels.nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Interpolate the map from the file
        interp_hpix = ops.InterpolateHealpixMap(
            file=hpix_file,
            det_data="interp_data",
            detector_pointing=detpointing,
            stokes_weights=weights,
        )
        interp_hpix.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                np.testing.assert_almost_equal(
                    ob.detdata[defaults.det_data][det],
                    ob.detdata["interp_data"][det],
                    decimal=1,
                )

        close_data(data)
