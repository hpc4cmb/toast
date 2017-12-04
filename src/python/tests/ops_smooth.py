# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

import numpy as np
import healpy as hp
import libsharp

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_map import *
from ..map.pixels import *
from ..map.rings import DistRings

class OpSmoothTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

        # Note: self.comm is set by the test infrastructure
        self.worldsize = self.comm.size
        if (self.worldsize >= 2):
            self.groupsize = int( self.worldsize / 2 )
            self.ngroup = 2
        else:
            self.groupsize = 1
            self.ngroup = 1
        self.toastcomm = Comm(world=self.comm, groupsize=self.groupsize)
        self.data = {} #FIXME once Data supports map, use Data: Data(self.toastcomm)

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
                            nside = self.nside,
                            nnz = 3)
        self.data["signal_map"] = self.input_map[:, self.dist_rings.local_pixels]

    def tearDown(self):
        del self.data


    def test_smooth(self):
        start = MPI.Wtime()

        # construct the PySM operator.  Pass in information needed by PySM...
        op = OpSmooth(comm=self.comm, signal_map="signal_map",
                lmax=self.lmax, grid=self.dist_rings.libsharp_grid,
                fwhm_deg=self.fwhm_deg, beam=None)
        op.exec(self.data)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("test_opsmooth took {:.3f} s".format(elapsed))

        output_map = np.zeros(self.input_map.shape, dtype=np.float64) if self.comm.rank == 0 else None
        self.comm.Reduce(self.data["smoothed_signal_map"], output_map, root=0, op=MPI.SUM)

        if self.comm.rank == 0:
            hp_smoothed = hp.smoothing(self.input_map, fwhm=np.radians(self.fwhm_deg), lmax=self.lmax)
            np.testing.assert_array_almost_equal(hp_smoothed, output_map, decimal=2)
            print("Std of difference between libsharp and healpy", (hp_smoothed-output_map).std())
