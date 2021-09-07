# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

import healpy as hp

from ..mpi import MPI

from .mpi import MPITestCase

from ..noise import Noise

from ..vis import set_matplotlib_backend

from .. import ops as ops

from ..pixels import PixelDistribution, PixelData

from ..observation import default_names as obs_names

from ._helpers import create_outdir, create_satellite_data


class CrossLinkingTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_crosslinking(self):
        np.random.seed(123456)

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()

        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle=obs_names.hwp_angle,
            detector_pointing=detpointing,
            pixels=obs_names.pixels,
        )

        # Crosslinking
        crosslinking = ops.CrossLinking(
            pointing=pointing,
            noise_model="noise_model",
            pixel_dist="pixel_dist",
            det_flags=None,
            output_dir=self.outdir,
        )
        crosslinking.apply(data)

        del data
        return

    def test_crosslinking_cached(self):
        np.random.seed(123456)

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()

        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle=obs_names.hwp_angle,
            detector_pointing=detpointing,
            pixels=obs_names.pixels,
        )

        # Cache pointing
        pointing.apply(data)

        # Crosslinking
        crosslinking = ops.CrossLinking(
            name="cached_crosslinking",
            pointing=pointing,
            noise_model="noise_model",
            pixel_dist="pixel_dist",
            det_flags=None,
            output_dir=self.outdir,
        )
        crosslinking.apply(data)

        del data
        return
