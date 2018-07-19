# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

import numpy as np
import healpy as hp
try:
    import libsharp
except:
    libsharp = None

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_map import *
from ..map.pixels import *
from ..map.rings import DistRings
from ..map.smooth import LibSharpSmooth

from ._helpers import (create_outdir, create_comm)


class LibSharpSmoothTest(MPITestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.toastcomm = create_comm(self.comm)

        # create the same noise input map on all processes
        self.nside = 32
        np.random.seed(100)
        npix = hp.nside2npix(self.nside)
        self.input_map = np.random.normal(size=(3, npix))
        self.fwhm_deg = 10
        self.lmax = self.nside

        # distribute longitudinal rings
        # this will be performed by a dedicated operator before
        # calling the OpSmooth operator

        self.dist_rings = DistRings(self.toastcomm.comm_world,
            nside=self.nside, nnz=3)
        self.data = dict()
        self.data["signal_map"] = \
            self.input_map[:, self.dist_rings.local_pixels]


    def tearDown(self):
        del self.data


    def test_smooth(self):
        op = LibSharpSmooth(comm=self.comm, signal_map="signal_map",
            lmax=self.lmax, grid=self.dist_rings.libsharp_grid,
            fwhm_deg=self.fwhm_deg, beam=None)
        op.exec(self.data)

        # Copy our local piece into a buffer of zeros that we will
        # reduce to the root process.  We could also use a gather, but
        # this is a small buffer.
        local_output_map = np.zeros(self.input_map.shape, dtype=np.float64)
        local_output_map[:, self.dist_rings.local_pixels] = \
            self.data["smoothed_signal_map"]

        output_map = None
        if self.comm.rank == 0:
            output_map = np.zeros(self.input_map.shape, dtype=np.float64)
        self.comm.Reduce(local_output_map, output_map,
            root=0, op=MPI.SUM)

        if self.comm.rank == 0:
            hp_smoothed = hp.smoothing(self.input_map,
                fwhm=np.radians(self.fwhm_deg), lmax=self.lmax)
            np.testing.assert_array_almost_equal(hp_smoothed, output_map,
                decimal=2)
            print("Std of difference between libsharp and healpy",
                (hp_smoothed-output_map).std())

        return
