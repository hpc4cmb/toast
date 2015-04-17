# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

from toast.tod.streams import *


class StreamsTest(unittest.TestCase):


    def test_construction(self):
        start = MPI.Wtime()
        strm = Streams(MPI.COMM_WORLD)
        stop = MPI.Wtime()
        elapsed = stop - start
        print('Proc {}:  test took {:.4f} s'.format( strm.mpicomm.rank, elapsed ))



if __name__ == "__main__":
    unittest.main()



