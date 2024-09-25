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
from ..pixels_io_healpix import write_healpix_fits, write_healpix_hdf5
from ._helpers import (
    close_data,
    create_fake_mask,
    create_fake_sky,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class InterpolateHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
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

        hpix_file = os.path.join(self.outdir, "fake.fits")
        if data.comm.comm_world is None or data.comm.comm_world.rank == 0:
            # Create a smooth sky
            lmax = 3 * pixels.nside
            cls = np.ones([4, lmax + 1])
            np.random.seed(98776)
            fake_sky = hp.synfast(cls, pixels.nside, fwhm=np.radians(30))
            # Write this to a file
            hp.write_map(hpix_file, fake_sky)

        # Scan the map from the file

        scan_hpix = ops.ScanHealpixMap(
            file=hpix_file,
            det_data="scan_data",
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

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
                    ob.detdata["scan_data"][det],
                    ob.detdata["interp_data"][det],
                    decimal=1,
                )

        close_data(data)
