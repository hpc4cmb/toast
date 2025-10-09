# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import numpy as np
import scipy.stats as stats
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class StatisticsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_statcut(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # This is a simulation with the same focalplane for every obs...
        sample_rate = data.obs[0].telescope.focalplane.sample_rate

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate noise using this model
        sim_noise = ops.SimNoise()
        sim_noise.apply(data)

        # apply StatCut
        statcut = ops.SimpleStatCut()
        statcut.apply(data)

        close_data(data)

    def test_statcut_hwp(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # This is a simulation with the same focalplane for every obs...
        sample_rate = data.obs[0].telescope.focalplane.sample_rate

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate noise using this model
        sim_noise = ops.SimNoise()
        sim_noise.apply(data)

        # Demodulation requires Stokes weights

        detpointing = ops.PointingDetectorSimple()
        
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        # Demodulate
        demod = ops.Demodulate(
            stokes_weights=weights,
            in_place=True,
        )
        demod.apply(data)

        # apply StatCut
        statcut = ops.SimpleStatCut()
        statcut.apply(data)

        close_data(data)
