# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

from toast.planck.streams import *


class StreamsPlanckEFFTest(unittest.TestCase):


    def setUp(self):
        self.dets = ['217-5a']
        #self.eff = StreamsPlanckEFF(mpicomm=MPI.COMM_WORLD, timedist=True, detectors=self.dets)


    def test_props(self):
        start = MPI.Wtime()
        
        #self.assertEqual(self.eff.detectors, self.dets)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_write(self):
        start = MPI.Wtime()

        for d in self.dets:
            pass

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_read(self):
        start = MPI.Wtime()

        for d in self.dets:
            pass

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))





if __name__ == "__main__":
    unittest.main()



