# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest


class Streams(object):
    """
    Base class for an object that provides a collection of streams.

    Each Streams class has one or more d 
    """

    def __init__(self, world=MPI.COMM_WORLD, groupsize=0):



class StreamsTest(unittest.TestCase):


    def test_construction(self):
        start = MPI.Wtime()
        #str = Streams(self.comm)
        stop = MPI.Wtime()
        elapsed = stop - start
        print('Proc {}:  test took {:.4f} s'.format( self.comm.comm_world.rank, elapsed ))



if __name__ == "__main__":
    unittest.main()



