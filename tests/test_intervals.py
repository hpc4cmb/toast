# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

if 'PYTOAST_NOMPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

from toast.dist import *
from toast.tod.interval import *
from toast.tod.sim_interval import *

from toast.mpirunner import MPITestCase


class IntervalTest(MPITestCase):


    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.rate = 123.456
        self.duration = 24 * 3601.23
        self.gap = 3600.0
        self.chunks = 24
        self.start = 5432.1
        self.first = 10
        self.nint = 3


    def test_regular(self):
        start = MPI.Wtime()
        
        intrvls = regular_intervals(self.nint, self.start, self.first, self.rate, self.duration, self.gap, chunks=self.chunks)

        totsamp = self.duration + self.gap

        # for i in range(len(intrvls)):
        #     print("--- {} ---".format(i))
        #     print(intrvls[i].first)
        #     print(intrvls[i].last)
        #     print(intrvls[i].start)
        #     print(intrvls[i].stop)

        #self.assertTrue(False)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))

