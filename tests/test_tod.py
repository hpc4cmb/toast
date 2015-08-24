# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from .mpirunner import MPITestCase
import sys

from toast.tod import *


class TODTest(MPITestCase):


    def setUp(self):
        # Note: self.comm is set by the test infrastructure
        self.dets = {
            '1a' : (0.0, 1.0),
            '1b' : (0.0, -1.0),
            '2a' : (1.0, 0.0),
            '2b' : (-1.0, 0.0)
            }
        self.flavs = ['proc1', 'proc2']
        self.flavscheck = [TOD.DEFAULT_FLAVOR] + self.flavs
        self.mynsamp = 10
        self.myoff = self.mynsamp * self.comm.rank
        self.totsamp = self.mynsamp * self.comm.size
        self.tod = TOD(mpicomm=self.comm, timedist=True, detectors=self.dets.keys(), flavors=self.flavs, samples=self.totsamp)
        self.rms = 10.0
        self.pntgvec = np.ravel(np.random.random((self.mynsamp, 4)))
        self.pflagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)

        self.datavec = np.random.normal(loc=0.0, scale=self.rms, size=self.mynsamp)
        self.flagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)
        for d in self.dets:
            self.tod.write_pntg(detector=d, local_start=0, data=self.pntgvec, flags=self.pflagvec)
            for f in self.flavs:
                self.tod.write(detector=d, flavor=f, local_start=0, data=self.datavec, flags=self.flagvec)
        self.whtnse = TODSimple(mpicomm=self.comm, timedist=True, detectors=self.dets, rms=self.rms, samples=self.totsamp)


    def test_props(self):
        start = MPI.Wtime()
        
        self.assertEqual(sorted(self.tod.detectors), sorted(self.dets.keys()))
        self.assertEqual(sorted(self.tod.local_dets), sorted(self.dets.keys()))
        self.assertEqual(self.tod.flavors, self.flavscheck)
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
            for f in self.flavs:
                data, flags = self.tod.read(detector=d, flavor=f, local_start=0, n=self.mynsamp)
                np.testing.assert_equal(flags, self.flagvec)
                np.testing.assert_almost_equal(data, self.datavec)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_read_pntg(self):
        start = MPI.Wtime()

        for d in self.dets:
            pntg, pflags = self.tod.read_pntg(detector=d, local_start=0, n=self.mynsamp)
            np.testing.assert_equal(pflags, self.pflagvec)
            np.testing.assert_almost_equal(pntg, self.pntgvec)

        stop = MPI.Wtime()
        elapsed = stop - start


    def test_white(self):
        start = MPI.Wtime()

        for d in self.dets:
            data, flags = self.whtnse.read(detector=d, local_start=0, n=self.mynsamp)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))

