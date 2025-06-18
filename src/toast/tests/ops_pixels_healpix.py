# Copyright (c) 2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from .._libtoast import pixels_healpix
from ..covariance import covariance_apply
from ..intervals import IntervalList
from ..mpi import MPI
from ..noise import Noise
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import create_outdir
from .mpi import MPITestCase


class PixelsHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_pixels_healpix(self):
        nsample = 1
        quat_index = np.array([0], dtype=np.int32)
        quats = np.zeros([1, nsample, 4], dtype=np.float64)
        quats[0][0] = np.array(
            [
                -0.5130854625967908866357447550399228930473,
                0.8174841998445969704079061557422392070293,
                -0.1390968346448042680663093051407486200333,
                0.2216189560215287845945653089074767194688,
            ]
        )
        flags = np.zeros(1, dtype=np.uint8)
        flag_mask = 1
        pixel_index = np.array([0], dtype=np.int32)
        pixels = np.zeros([1, nsample], dtype=np.int64)
        times = np.arange(nsample)
        intervals = IntervalList(times, samplespans=[(0, nsample - 1)])
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
        bad = pixels[0] >= npix
        for samp in np.arange(nsample)[bad]:
            print(f"{samp}: {quats[0][samp]} -> {pixels[0][samp]}", flush=True)

        if np.count_nonzero(bad) > 0:
            print(f"Some pointings failed")
            self.assertTrue(False)

        return
