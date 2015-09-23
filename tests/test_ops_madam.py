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
        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = Data(self.toastcomm)

        self.dets = {
            '1a' : (0.0, 1.0),
            '1b' : (0.0, -1.0),
            '2a' : (1.0, 0.0),
            '2b' : (-1.0, 0.0)
            }

        self.totsamp = 100000
        self.rms = 10.0

        # madam only supports a single observation
        nobs = 1

        for i in range(nobs):
            # create the TOD for this observation

            tod = TODFake(
                mpicomm = self.data.comm.comm_group, 
                timedist = True, 
                detectors = self.dets,
                rms = self.rms,
                samples = self.totsamp
            )

            self.data.obs.append( 
                Obs( 
                    tod = tod,
                    intervals = [],
                    baselines = None, 
                    noise = None
                )
            )


    def test_madam_gradient(self):
        start = MPI.Wtime()

        par = {}

        op = OpMadam(params=par)
        op.exec(self.data)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

