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
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class YieldCutTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_yield_cut(self):
        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, pixel_per_process=100)

        cut = ops.YieldCut(
            det_mask=defaults.det_mask_invalid,
            det_flags=defaults.det_flags,
            det_flag_mask=defaults.det_mask_invalid,
            fixed=True,
            keep_frac=0.75,
        )

        # Before applying operator, check the per-detector and per-sample flags
        pre_total_dets = 0
        pre_total_samps = 0
        for obs in data.obs:
            for det in obs.select_local_detectors(flagmask=defaults.det_mask_invalid):
                det_flags = obs.detdata[cut.det_flags][det]
                good = (det_flags & cut.det_flag_mask) == 0
                pre_total_samps += np.sum(good)
                pre_total_dets += 1

        cut.apply(data)

        # Measure the cut fraction.  We check both the per-detector flags and
        # the per sample flags, both of which are updated by the operator.
        cut_total_dets = 0
        cut_total_samps = 0
        for obs in data.obs:
            for det in obs.select_local_detectors(flagmask=defaults.det_mask_invalid):
                det_flags = obs.detdata[cut.det_flags][det]
                good = (det_flags & cut.det_flag_mask) == 0
                cut_total_samps += np.sum(good)
                cut_total_dets += 1
        if data.comm.comm_world is not None:
            pre_total_dets = data.comm.comm_world.allreduce(pre_total_dets, op=MPI.SUM)
            pre_total_samps = data.comm.comm_world.allreduce(
                pre_total_samps, op=MPI.SUM
            )
            cut_total_dets = data.comm.comm_world.allreduce(cut_total_dets, op=MPI.SUM)
            cut_total_samps = data.comm.comm_world.allreduce(
                cut_total_samps, op=MPI.SUM
            )
        keep_frac_det = cut_total_dets / pre_total_dets
        keep_frac_samp = cut_total_samps / pre_total_samps
        if np.abs(keep_frac_det - cut.keep_frac) > 0.1:
            print(f"Cut {keep_frac_det:0.2f} of detectors instead of {cut.keep_frac}")
            self.assertTrue(False)
        if np.abs(keep_frac_samp - cut.keep_frac) > 0.1:
            print(f"Cut {keep_frac_samp:0.2f} of samples instead of {cut.keep_frac}")
            self.assertTrue(False)

        close_data(data)
