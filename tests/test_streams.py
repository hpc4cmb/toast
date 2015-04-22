# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

from toast.tod.streams import *


class StreamsTest(unittest.TestCase):


    def setUp(self):
        self.dets = ['1a', '1b', '2a', '2b']
        self.flavs = ['proc1', 'proc2']
        self.flavscheck = [Streams.DEFAULT_FLAVOR] + self.flavs
        self.nsamp = 100
        self.strms = Streams(mpicomm=MPI.COMM_WORLD, timedist=True, detectors=self.dets, flavors=self.flavs, samples=self.nsamp)
        self.rms = 10.0
        self.datavec = np.random.normal(loc=0.0, scale=self.rms, size=self.nsamp)
        self.flagvec = np.random.uniform(low=0, high=1, size=self.nsamp).astype(np.uint8, copy=True)
        for d in self.dets:
            for f in self.flavs:
                self.strms.write(detector=d, flavor=f, start=0, data=self.datavec, flags=self.flagvec)
        self.whtnse = StreamsWhiteNoise(mpicomm=MPI.COMM_WORLD, timedist=True, detectors=self.dets, rms=self.rms, samples=self.nsamp)


    def test_props(self):
        start = MPI.Wtime()
        
        self.assertEqual(self.strms.valid_dets(), self.dets)
        self.assertEqual(self.strms.valid_flavors(), self.flavscheck)
        self.assertEqual(self.strms.nsamp(), self.nsamp)
        self.assertTrue(self.strms.is_timedist())

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_read(self):
        start = MPI.Wtime()

        for d in self.dets:
            for f in self.flavs:
                data, flags = self.strms.read(detector=d, flavor=f, start=0, n=self.nsamp)
                np.testing.assert_equal(flags, self.flagvec)
                np.testing.assert_almost_equal(data, self.datavec)


        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_white(self):
        start = MPI.Wtime()

        for d in self.dets:
            data, flags = self.whtnse.read(detector=d, start=0, n=self.nsamp)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))




if __name__ == "__main__":
    unittest.main()



