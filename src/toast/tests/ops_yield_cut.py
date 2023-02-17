# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import numpy as np
import scipy.stats as stats
from astropy import units as u

from .. import ops as ops
from ..mpi import MPI
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class YieldCutTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_yield_cut(self):
        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, pixel_per_process=100)

        cut = ops.YieldCut(
            det_flags=defaults.det_flags,
            det_flag_mask=defaults.det_mask_invalid,
            fixed=True,
            keep_frac=0.75,
        )
        cut.apply(data)

        # Measure the cut fraction
        ngood = 0
        ntotal = 0
        for obs in data.obs:
            for det in obs.local_detectors:
                det_flags = obs.detdata[cut.det_flags][det]
                good = (det_flags & cut.det_flag_mask) == 0
                ngood += np.sum(good)
                ntotal += good.size
        if data.comm.comm_world is not None:
            ngood = data.comm.comm_world.allreduce(ngood, op=MPI.SUM)
            ntotal = data.comm.comm_world.allreduce(ntotal, op=MPI.SUM)
        keep_frac = ngood / ntotal
        assert np.abs(cut.keep_frac - keep_frac) < 0.1

        close_data(data)
