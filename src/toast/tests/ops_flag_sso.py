# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from astropy import units as u

import healpy as hp

from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import qarray as qa

from .. import ops as ops

from ..observation import default_names as obs_names

from ..pixels_io import write_healpix_fits

from ._helpers import create_outdir, create_ground_data


class FlagSSOTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_flag(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=obs_names.boresight_azel, quats="quats_azel"
        )
        detpointing_azel.apply(data)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Simulate noise
        sim_noise = ops.SimNoise()
        sim_noise.apply(data)

        # Flag
        flag_sso = ops.FlagSSO(
            detector_pointing=detpointing_azel,
            det_flags="flags",
            sso_names=["Sun", "Moon"],
            sso_radii=[93 * u.deg, 5 * u.deg],
        )
        flag_sso.apply(data)