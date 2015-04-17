# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

from dist import *


class DistTest(unittest.TestCase):

    def setUp(self):
        self.mpicomm = MPI.COMM_WORLD
        worldsize = self.mpicomm.size
        groupsize = int( worldsize / 2 )
        self.comm = Comm(MPI.COMM_WORLD, groupsize=groupsize)

    def test_construction(self):
        start = MPI.Wtime()
        self.dist = Dist(self.comm)
        stop = MPI.Wtime()
        elapsed = stop - start
        print('Proc {}:  test took {:.4f} s'.format( self.comm.comm_world.rank, elapsed ))



if __name__ == "__main__":
    unittest.main()



