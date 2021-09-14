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


class DemodulateTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_demodulate(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise
        sim_noise = ops.SimNoise(noise_model=default_model.noise_model)
        sim_noise.apply(data)

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle=obs_names.hwp_angle,
            detector_pointing=detpointing,
        )

        # Demodulate

        demod = ops.Demodulate(pointing=pointing)
        demod_data = demod.apply(data)

        # Bin signal

        pointing.hwp_angle = None

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pointing=pointing,
            noise_model=default_model.noise_model,
        )

        mapper = ops.MapMaker(
            det_data=obs_names.det_data,
            binning=binner,
            template_matrix=None,
            write_hits=False,
            write_map=True,
            write_cov=False,
            write_rcond=False,
            keep_final_products=True,
            output_dir=self.outdir,
        )

        # Make maps

        mapper.name = "modulated"
        mapper.apply(data)

        mapper.name = "demodulated"
        pointing.hwp_angle = None
        mapper.apply(demod_data)
