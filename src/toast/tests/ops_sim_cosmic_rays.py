# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .. import ops as ops
from .. import rng
from ..covariance import covariance_apply
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_fake_sky,
    create_outdir,
    create_satellite_data,
    create_satellite_data_big,
)
from .mpi import MPITestCase


class SimCosmicRayTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def make_mock_cosmic_ray_data(self, data, crfile):
        # so far we hardcode typical values
        low_noise = [-3e-17, 1e-17]
        for obs in data.obs:
            for kk, det in enumerate(obs.local_detectors):
                direct_hits = np.zeros((400, 3))
                # import pdb; pdb.set_trace()

                direct_hits[:, 0] = 5e-17 * np.random.randn(400) + 3e-17  # C1, Watts
                direct_hits[:, 1] = 1e-15 * np.random.randn(400) + 2e-15  # C2, watts
                direct_hits[:, 2] = 2 * np.random.randn(400) + 4  # time constant, msec
                filename = crfile.replace("detector", f"det{kk}")
                np.savez(
                    filename,
                    low_noise=low_noise,
                    direct_hits=direct_hits,
                    sampling_rate=[156],
                )
        return

    def test_cosmic_rays_wafer_noise(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm,
        )
        crfile = f"{self.outdir}/cosmic_ray_glitches_detector.npz"
        self.make_mock_cosmic_ray_data(data, crfile)
        if self.comm is not None:
            self.comm.Barrier()
        # Simulate noise using this model
        key = "my_signal"
        sim_cosmic_rays = ops.InjectCosmicRays(
            det_data=key,
            crfile=crfile,
        )
        sim_cosmic_rays.apply(data)
        for obs in data.obs:
            telescope = obs.telescope.uid
            focalplane = obs.telescope.focalplane
            obsindx = obs.uid

            for det in obs.local_detectors:
                detindx = focalplane[det]["uid"]
                assert obs.detdata[key][det].sum() != 0.0
        close_data(data)

    def test_cosmic_rays_glitches(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm,
        )

        crfile = f"{self.outdir}/cosmic_ray_glitches_detector.npz"
        self.make_mock_cosmic_ray_data(data, crfile)
        if self.comm is not None:
            self.comm.Barrier()
        # Simulate noise using this model
        key = "my_signal"
        sim_cosmic_rays = ops.InjectCosmicRays(
            det_data=key,
            crfile=crfile,
            inject_direct_hits=True,
            eventrate=0.1,  # we increase the eventrate artificially for testing
        )
        sim_cosmic_rays.apply(data)
        for obs in data.obs:
            telescope = obs.telescope.uid
            focalplane = obs.telescope.focalplane
            obsindx = obs.uid
            for det in obs.local_detectors:
                detindx = focalplane[det]["uid"]
                assert obs.detdata[key][det].sum() != 0.0
        close_data(data)

    def test_cosmic_rays_commonmode(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm,
        )
        crfile = f"{self.outdir}/cosmic_ray_glitches_detector.npz"
        self.make_mock_cosmic_ray_data(data, crfile)
        if self.comm is not None:
            self.comm.Barrier()
        # Simulate noise using this model
        key = "my_signal"
        sim_cosmic_rays = ops.InjectCosmicRays(
            det_data=key, crfile=crfile, include_common_mode=True
        )
        sim_cosmic_rays.apply(data)
        for obs in data.obs:
            telescope = obs.telescope.uid
            focalplane = obs.telescope.focalplane
            obsindx = obs.uid

            for det in obs.local_detectors:
                detindx = focalplane[det]["uid"]
                assert obs.detdata[key][det].sum() != 0.0
        close_data(data)
