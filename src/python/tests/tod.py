# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

from ..dist import *
from ..tod.tod import *


class TODTest(MPITestCase):


    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
            
        # Note: self.comm is set by the test infrastructure
        self.dets = ['1a', '1b', '2a', '2b']
        self.mynsamp = 10
        self.myoff = self.mynsamp * self.comm.rank
        self.totsamp = self.mynsamp * self.comm.size
        self.tod = TODCache(mpicomm=self.comm, timedist=True, detectors=self.dets, samples=self.totsamp)
        self.rms = 10.0
        self.pntgvec = np.ravel(np.random.random((self.mynsamp, 4))).reshape(-1,4)
        self.pflagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)

        self.datavec = np.random.normal(loc=0.0, scale=self.rms, size=self.mynsamp)
        self.flagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)
        for d in self.dets:
            self.tod.write_common_flags(local_start=0, flags=self.pflagvec)
            self.tod.write_det_flags(detector=d, local_start=0, flags=self.flagvec)
            self.tod.write(detector=d, local_start=0, data=self.datavec)
            self.tod.write_pntg(detector=d, local_start=0, data=self.pntgvec)

    def tearDown(self):
        self.tod.cache.clear()


    def test_props(self):
        start = MPI.Wtime()
        
        self.assertEqual(sorted(self.tod.detectors), sorted(self.dets))
        self.assertEqual(sorted(self.tod.local_dets), sorted(self.dets))
        self.assertEqual(self.tod.total_samples, self.totsamp)
        self.assertEqual(self.tod.local_samples[0], self.myoff)
        self.assertEqual(self.tod.local_samples[1], self.mynsamp)
        self.assertTrue(self.tod.timedist)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_read(self):
        start = MPI.Wtime()

        for d in self.dets:
            data = self.tod.read(detector=d, local_start=0, n=self.mynsamp)
            flags, common = self.tod.read_flags(detector=d, local_start=0, n=self.mynsamp)
            np.testing.assert_equal(flags, self.flagvec)
            np.testing.assert_equal(common, self.pflagvec)
            np.testing.assert_almost_equal(data, self.datavec)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_read_pntg(self):
        start = MPI.Wtime()

        for d in self.dets:
            pntg = self.tod.read_pntg(detector=d, local_start=0, n=self.mynsamp)
            np.testing.assert_almost_equal(pntg, self.pntgvec)

        stop = MPI.Wtime()
        elapsed = stop - start

