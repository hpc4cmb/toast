# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from . import mpiunittest as unittest
import sys

from toast.dist import *


class DistTest(unittest.TestCase):

    def setUp(self):
        self.mpicomm = MPI.COMM_WORLD
        self.worldsize = self.mpicomm.size
        if (self.worldsize >= 2):
            self.groupsize = int( self.worldsize / 2 )
            self.ngroup = 2
        else:
            self.groupsize = 1
            self.ngroup = 1
        self.comm = Comm(MPI.COMM_WORLD, groupsize=self.groupsize)

    def test_construction(self):
        start = MPI.Wtime()
        
        self.assertEqual(self.comm.ngroups, self.ngroup)
        self.assertEqual(self.comm.group_size, self.groupsize)
        
        self.dist = Dist(self.comm)
        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))



if __name__ == "__main__":
    unittest.main()



