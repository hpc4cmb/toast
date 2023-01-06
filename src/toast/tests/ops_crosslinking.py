# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..mpi import MPI
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class CrossLinkingTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_crosslinking(self):
        np.random.seed(123456)

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()

        pixelpointing = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
            pixels=defaults.pixels,
        )

        # Crosslinking
        crosslinking = ops.CrossLinking(
            pixel_pointing=pixelpointing,
            pixel_dist="pixel_dist",
            output_dir=self.outdir,
        )
        crosslinking.apply(data)

        # Check that the total number of samples makes sense
        nsample = 0
        for obs in data.obs:
            nsample += obs.n_local_samples * len(obs.local_detectors)
        if self.comm is not None:
            nsample = self.comm.reduce(nsample)

        if self.comm is None or self.comm.rank == 0:
            fname = os.path.join(crosslinking.output_dir, f"{crosslinking.name}.fits")
            m = hp.read_map(fname)
            nhit = np.sum(m)
            assert nhit == nsample

        close_data(data)
