# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os
import sys

import healpy as hp
import numpy as np
import numpy.testing as nt
import scipy.sparse
from astropy import units as u
from astropy.table import Column

from toast import ObsMat

from .. import ops as ops
from ..mpi import MPI, Comm
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import read_healpix
from ..vis import set_matplotlib_backend
from .helpers import (
    close_data,
    create_ground_data,
    create_outdir,
    fake_flags,
)
from .mpi import MPITestCase


class FlagIntervalsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_flag_intervals(self):
        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, turnarounds_invalid=True)

        mask1 = np.uint8(128)
        nflagged0 = 0
        ntot = 0
        for ob in data.obs:
            flags = ob.shared[defaults.shared_flags].data
            ntot += flags.size
            nflagged0 += np.sum(flags & mask1 != 0)
        assert nflagged0 == 0

        # Flag all samples in the leftright interval
        
        flag_intervals = ops.FlagIntervals(
            view_mask=[(defaults.scan_leftright_interval, mask1)],
            reset=True,
        )
        flag_intervals.apply(data)

        nflagged1 = 0
        for ob in data.obs:
            flags = ob.shared[defaults.shared_flags].data
            nflagged1 += np.sum(flags & mask1 != 0)
        assert nflagged1 != 0

        # Flag all samples outside the leftright interval

        mask2 = np.uint8(64)
        flag_intervals = ops.FlagIntervals(
            view_mask=[(f"~{defaults.scan_leftright_interval}", mask2)],
            reset=True,
        )
        flag_intervals.apply(data)

        # Compare the two sets

        for iob, ob in enumerate(data.obs):
            flags = ob.shared[defaults.shared_flags].data
            flagged1 = flags & mask1 != 0
            flagged2 = flags & mask2 != 0
            # Test completeness
            assert not np.any(flagged1 + flagged2 == 0)
            # Test overlap
            assert not np.any(flagged1 + flagged2 == 2)

        close_data(data)
