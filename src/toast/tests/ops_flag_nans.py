# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..data import Data
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import (
    close_data,
    create_comm,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class FlagNaNsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_flag(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        data = create_satellite_data(self.comm)

        # Simulate noise in the data object
        ops.DefaultNoiseModel().apply(data)
        ops.SimNoise().apply(data)

        # Put NaNs in the data

        obs = data.obs[0]
        dets = obs.local_detectors

        # Flag a segment of the data

        det1 = dets[0]
        ind = slice(100, 200)
        obs.detdata[defaults.det_data][det1][ind] = np.nan

        # Flag all of the data

        det2 = dets[1]
        obs.detdata[defaults.det_data][det2][:] = np.nan

        ops.FlagNaNs().apply(data)

        # Verify success

        assert np.all(obs.detdata[defaults.det_data][det1][ind] == 0)
        assert np.all(
            obs.detdata[defaults.det_flags][det1][ind] & defaults.det_mask_invalid != 0
        )
        assert np.all(obs.detdata[defaults.det_data][det2] == 0)
        assert np.all(obs.local_detector_flags[det2] & defaults.det_mask_invalid != 0)
        
        close_data(data)
