# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from .mpirunner import MPITestCase
import sys

from toast.tod.tod import *
from toast.map.madam import *


class OpMadamTest(MPITestCase):

    def setUp(self):
        pass


    def test_madam_gradient(self):
        start = MPI.Wtime()

        
        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("copy test took {:.3f} s".format(elapsed))

