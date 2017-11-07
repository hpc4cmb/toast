# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_map import *
from ..map.pixels import *


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


    def test_pysm(self):
        start = MPI.Wtime()

        # expand the pointing into a low-res pointing matrix
        pointing = OpPointingHpix(nside=self.nside, nest=False)
        pointing.exec(self.data)

        # Get locally hit pixels.  Only do this if the PySM operator
        # needs local pixels...
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)
        submapsize = np.floor_divide(self.nside, 4)
        localsm = np.unique(np.floor_divide(localpix, submapsize))

        # construct a distributed map so that we can use the global to
        # local pixel mapping.  FIXME:  change this after fixing:
        # https://github.com/hpc4cmb/toast/issues/97
        #
        npix = 12 * self.nside * self.nside
        hits = DistPixels(comm=self.toastcomm.comm_world, size=npix, nnz=1, dtype=np.int32, submap=submapsize, local=localsm)

        # construct the PySM operator.  Pass in information needed by PySM...

        pysm_sky_config = {
            'synchrotron' : "s1",
            'dust' : "d8",
            'freefree' : "f1",
            'cmb' : "c1",
            'ame' : "a1",
        }
        op = OpSimPySM(distmap=hits,
                       pysm_sky_config=pysm_sky_config,
        )
        op.exec(self.data)

        # Now we have timestreams in the cache.  We could compare the 
        # timestream values or we could make a binned map and look at those
        # values.



        #self.assertTrue(False)
        
        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("test_pysm took {:.3f} s".format(elapsed))

