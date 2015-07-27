# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from .mpirunner import MPITestCase
import sys

from toast.tod.streams import *


class StreamsTest(MPITestCase):


    def setUp(self):
        # Note: self.comm is set by the test infrastructure
        self.dets = ['1a', '1b', '2a', '2b']
        self.flavs = ['proc1', 'proc2']
        self.flavscheck = [Streams.DEFAULT_FLAVOR] + self.flavs
        self.mynsamp = 10
        self.myoff = self.mynsamp * self.comm.rank
        self.totsamp = self.mynsamp * self.comm.size
        self.strms = Streams(mpicomm=self.comm, timedist=True, detectors=self.dets, flavors=self.flavs, samples=self.totsamp)
        self.rms = 10.0
        self.datavec = np.random.normal(loc=0.0, scale=self.rms, size=self.mynsamp)
        self.flagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)
        for d in self.dets:
            for f in self.flavs:
                self.strms.write(detector=d, flavor=f, local_start=0, data=self.datavec, flags=self.flagvec)
        self.whtnse = StreamsWhiteNoise(mpicomm=self.comm, timedist=True, detectors=self.dets, rms=self.rms, samples=self.totsamp)


    def test_props(self):
        start = MPI.Wtime()
        
        self.assertEqual(self.strms.detectors, self.dets)
        self.assertEqual(self.strms.local_dets, self.dets)
        self.assertEqual(self.strms.flavors, self.flavscheck)
        self.assertEqual(self.strms.total_samples, self.totsamp)
        self.assertEqual(self.strms.local_samples[0], self.myoff)
        self.assertEqual(self.strms.local_samples[1], self.mynsamp)
        self.assertTrue(self.strms.timedist)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_read(self):
        start = MPI.Wtime()

        for d in self.dets:
            for f in self.flavs:
                data, flags = self.strms.read(detector=d, flavor=f, local_start=0, n=self.mynsamp)
                np.testing.assert_equal(flags, self.flagvec)
                np.testing.assert_almost_equal(data, self.datavec)


        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_white(self):
        start = MPI.Wtime()

        for d in self.dets:
            data, flags = self.whtnse.read(detector=d, local_start=0, n=self.mynsamp)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))

