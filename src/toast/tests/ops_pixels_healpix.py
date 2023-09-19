# Copyright (c) 2023 by the parties listed in the AUTHORS file.
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
from .._libtoast import pixels_healpix
from ..intervals import IntervalList
from .. import qarray as qa


class PixelsHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_pixels_healpix(self):
        nsample = 2
        quat_index = np.array([0], dtype=np.int32)
        quats = np.zeros([1, nsample, 4], dtype=np.float64)
        quats[0][0] = np.array([
            -0.5130854625967908866357447550399228930473,
            0.8174841998445969704079061557422392070293,
            -0.1390968346448042680663093051407486200333,
            0.2216189560215287845945653089074767194688,
        ])
        flags = np.zeros(1, dtype=np.uint8)
        flag_mask = 1
        pixel_index = np.array([0], dtype=np.int32)
        pixels = np.zeros([1, nsample], dtype=np.int64)
        times = np.arange(nsample)
        intervals = IntervalList(times, samplespans=[(0, 1000000000)])
        nside = 4096
        npix = 12 * nside**2
        npix_submap = 3072
        nsubmap = npix // npix_submap
        hit_submaps = np.zeros(nsubmap, dtype=np.uint8)
        nest = True
        use_accel = False

        ret = pixels_healpix(
            quat_index,
            quats,
            flags,
            flag_mask,
            pixel_index,
            pixels,
            intervals.data,
            hit_submaps,
            npix_submap,
            nside,
            nest,
            use_accel,
        )

        assert np.all(pixels < npix)
        
        return
