# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
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

from ..pixels_io import write_healpix_fits


from ._helpers import create_outdir, create_healpix_ring_satellite


class SimGainTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nside = 16

    def test_sim(self):
        # Create a fake scan strategy that hits every pixel once.
        data = create_healpix_ring_satellite(self.comm, nside=self.nside)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # make a simple pointing matrix
        pointing = ops.PointingHealpix(nside=self.nside, nest=False, mode="I")

        # Generate timestreams
        sim_dipole = ops.SimDipole(mode="solar", coord="G")
        sim_dipole.exec(data)

        drifter  = ops.GainDrifter(fknee_drift=20, coord=)
        drifter.exec(data)
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        if rank == 0:
            for ob in data.obs:
                import pdb; pdb.set_trace()
                #for vw in range(len(views)):
                #    focalplane = ob.telescope.focalplane
                #    for kdet , det in enumerate(dets):

            import matplotlib.pyplot as plt


            #np.testing.assert_almost_equal(maxmap, self.dip_check, decimal=5)
            #np.testing.assert_almost_equal(minmap, -self.dip_check, decimal=5)
