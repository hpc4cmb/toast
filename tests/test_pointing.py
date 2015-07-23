# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
import mpiunittest as unittest
import sys

from toast.tod.pointing import *


class PointingTest(unittest.TestCase):


    def setUp(self):
        self.dets = ['1a', '1b', '2a', '2b']
        self.mynsamp = 100
        self.myoff = self.mynsamp * MPI.COMM_WORLD.rank
        self.totsamp = self.mynsamp * MPI.COMM_WORLD.size
        self.pntg = Pointing(mpicomm=MPI.COMM_WORLD, timedist=True, detectors=self.dets, samples=self.totsamp)
        self.datavec = np.zeros(4*self.mynsamp, dtype=np.float64)
        self.flagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)
        # generate some fake pointing for each of the fake detectors
        for d in self.dets:
            self.pntg.write(detector=d, local_start=0, data=self.datavec, flags=self.flagvec)


    def test_props(self):
        start = MPI.Wtime()
        
        self.assertEqual(self.pntg.detectors, self.dets)
        self.assertEqual(self.pntg.local_dets, self.dets)
        self.assertEqual(self.pntg.total_samples, self.totsamp)
        self.assertEqual(self.pntg.local_samples[0], self.myoff)
        self.assertEqual(self.pntg.local_samples[1], self.mynsamp)
        self.assertTrue(self.pntg.timedist)

        stop = MPI.Wtime()
        elapsed = stop - start


    def test_read(self):
        start = MPI.Wtime()

        for d in self.dets:
            data, flags = self.pntg.read(detector=d, local_start=0, n=self.mynsamp)
            np.testing.assert_equal(flags, self.flagvec)
            np.testing.assert_almost_equal(data, self.datavec)

        stop = MPI.Wtime()
        elapsed = stop - start
            

if __name__ == "__main__":
    unittest.main()



