# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os
import numpy as np

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_map import *
from ..map.pixels import *
from ..map.rings import DistRings
from ..map.pysm import PySMSky


class OpSimPySMTest(MPITestCase):

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
        self.data = Data(self.toastcomm)

        spread = 0.1 * np.pi / 180.0
        angterm = np.cos(spread / 2.0)
        axiscoef = np.sin(spread / 2.0)

        self.dets = {
            '1a' : np.array([axiscoef, 0.0, 0.0, angterm]),
            '1b' : np.array([-axiscoef, 0.0, 0.0, angterm]),
            '2a' : np.array([0.0, axiscoef, 0.0, angterm]),
            '2b' : np.array([0.0, -axiscoef, 0.0, angterm])
            }

        self.nside = 8
        self.totsamp = 12 * self.nside**2

        # Every process group creates a single observation that will
        # have the same data.
        nobs = 1

        for i in range(nobs):
            # create the TOD for this observation.  This fake TOD just has
            # boresight pointing that spirals around the healpix rings.

            tod = TODHpixSpiral(
                self.toastcomm.comm_group,  
                self.dets,
                self.totsamp,
                rate=1.0,
                nside=self.nside,
                detranks=self.toastcomm.group_size,
            )

            ob = {}
            ob['name'] = 'test'
            ob['id'] = 0
            ob['tod'] = tod
            ob['intervals'] = None
            ob['baselines'] = None
            ob['noise'] = None

            self.data.obs.append(ob)


    def tearDown(self):
        del self.data


    def test_pysm_local_pix(self):
        start = MPI.Wtime()

        npix = 12 * self.nside * self.nside

        npix_local = int(np.ceil(npix / float(self.toastcomm.comm_world.size)))
        local_pixels = np.arange(
            self.toastcomm.comm_world.rank * npix_local,
            min(self.toastcomm.comm_world.rank * npix_local + npix_local, npix)
            )

        # construct the PySM operator.  Pass in information needed by PySM...

        pysm_sky_config = {
            'synchrotron': "s1",
            'dust': "d8",
            'freefree': "f1",
            'cmb': "c1",
            'ame': "a1",
        }
        bandpasses = {
                "1a": (np.linspace(20, 25, 10), np.ones(10)),
                "1b": (np.linspace(21, 26, 10), np.ones(10)),
                "2a": (np.linspace(18, 23, 10), np.ones(10)),
                "2b": (np.linspace(19, 24, 10), np.ones(10)),
        }
        op = PySMSky(local_pixels=local_pixels, nside=self.nside,
                       pysm_sky_config=pysm_sky_config, bandpasses=bandpasses)
        local_map = {} # it should be Cache in production
        op.exec(local_map, out="sky")

        # Now we have timestreams in the cache.  We could compare the
        # timestream values or we could make a binned map and look at those
        # values.

        np.testing.assert_almost_equal(local_map["sky_1a"][0, 0, :3],
            np.array([121.40114346, 79.86737489, 77.23336053]))

        np.testing.assert_almost_equal(local_map["sky_1b"][0, 2, -3:],
            np.array([1.57564944, -0.22345616, -3.55604102]))

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("test_pysm took {:.3f} s".format(elapsed))

    def test_pysm_distrings(self):
        start = MPI.Wtime()

        dist_rings = DistRings(self.toastcomm.comm_world,
                            nside = self.nside,
                            nnz = 3)

        # construct the PySM operator.  Pass in information needed by PySM...

        pysm_sky_config = {
            'synchrotron': "s1",
            'dust': "d8",
            'freefree': "f1",
            'cmb': "c1",
            'ame': "a1",
        }
        bandpasses = {
                "1a": (np.linspace(20, 25, 10), np.ones(10)),
                "1b": (np.linspace(21, 26, 10), np.ones(10)),
                "2a": (np.linspace(18, 23, 10), np.ones(10)),
                "2b": (np.linspace(19, 24, 10), np.ones(10)),
        }
        op = PySMSky(local_pixels=dist_rings.local_pixels, nside=self.nside,
                       pysm_sky_config=pysm_sky_config, bandpasses=bandpasses)
        local_map = {} # it should be Cache in production
        op.exec(local_map, out="sky")

        # Now we have timestreams in the cache.  We could compare the 
        # timestream values or we could make a binned map and look at those
        # values.

        np.testing.assert_almost_equal(local_map["sky_1a"][0, 0, :3],
            np.array([121.40114346, 79.86737489, 77.23336053]))

        np.testing.assert_almost_equal(local_map["sky_1b"][0, 2, -3:],
            np.array([1.57564944, -0.22345616, -3.55604102]))

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("test_pysm took {:.3f} s".format(elapsed))

