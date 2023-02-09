# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..covariance import covariance_apply
from ..mpi import MPI
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class CadenceMapTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_cadence_map(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()
        pixelpointing = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
            create_dist="pixel_dist",
        )
        pixelpointing.apply(data)

        # Cadence map
        cadence_map = ops.CadenceMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixelpointing,
            output_dir=self.outdir,
        )

        cadence_map.apply(data)

        close_data(data)
