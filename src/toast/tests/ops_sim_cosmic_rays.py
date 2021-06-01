# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import healpy as hp
from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import ops as ops
from .. import rng

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ..covariance import covariance_apply
from ._helpers import (
    create_outdir,
    create_satellite_data,
    create_satellite_data_big,
    create_fake_sky,
)


class SimCosmicRayTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_cosmic_rays(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm,
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        dir="/Users/peppe/work/satellite_sims/cosmic_rays/"
        sim_cosmic_rays = ops.InjectCosmicRays(det_data=key,
                        crfile=f"{dir}/cosmic_ray_glitches_detector.npz",
                    )
        sim_cosmic_rays.apply(data)
        for obs  in  (data.obs ):
            telescope = obs.telescope.uid
            focalplane = obs.telescope.focalplane
            obsindx = obs.uid

            for det in obs.local_detectors:
                detindx = focalplane[det]["uid"]
                print(obs.detdata[key][det]  )
