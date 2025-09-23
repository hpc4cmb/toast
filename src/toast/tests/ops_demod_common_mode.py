# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import (close_data, create_fake_healpix_scanned_tod,
                      create_ground_data, create_outdir,
                      create_overdistributed_data)
from .mpi import MPITestCase


class DemodCommonModeTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def _test_demod_common_mode(self, weight_mode, data, suffix=""):
        nside = 128

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operator

        detpointing = ops.PointingDetectorSimple(shared_flag_mask=0)

        # Demodulate

        demod_weights_in = ops.StokesWeights(
            weights="demod_weights_in",
            mode=weight_mode,
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        downsample = 3
        demod = ops.Demodulate(
            stokes_weights=demod_weights_in,
            nskip=downsample,
            purge=False,
            mode=weight_mode,
        )
        demod_data = demod.apply(data)

        # Add a common mode

        input_rms = {}
        for ob in demod_data.obs:
            nsample = ob.n_local_samples
            common_mode = np.arange(nsample)
            good = (
                ob.shared[defaults.shared_flags].data & defaults.shared_mask_invalid
            ) == 0
            input_rms[ob.name] = np.std(common_mode[good])
            for det in ob.local_detectors:
                ob.detdata["signal"][det][:] = common_mode

        # Apply the filter

        common = ops.DemodCommonModeFilter(
            mode=weight_mode,
        )
        common.apply(demod_data)

        # Verify the mode is gone

        for ob in demod_data.obs:
            good = (
                ob.shared[defaults.shared_flags].data & defaults.shared_mask_invalid
            ) == 0
            rms0 = input_rms[ob.name]
            for det in ob.local_detectors:
                rms = np.std(ob.detdata["signal"][det][good])
                assert rms < 1e-3 * rms0

        if self.comm is not None:
            self.comm.barrier()
        close_data(demod_data)
        close_data(data)

    def test_demod_common_mode_IQU(self):
        data = create_ground_data(self.comm)
        self._test_demod_common_mode(weight_mode="IQU", data=data)

    def test_demod_common_mode_QU(self):
        data = create_ground_data(self.comm)
        self._test_demod_common_mode(weight_mode="QU", data=data)

    def test_demod_common_mode_I(self):
        data = create_ground_data(self.comm)
        self._test_demod_common_mode(weight_mode="I", data=data)
