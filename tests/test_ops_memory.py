# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from .mpirunner import MPITestCase
import sys

from toast.tod.streams import StreamsWhiteNoise
from toast.ops.memory import *


class OperatorMemoryTest(MPITestCase):

    def setUp(self):
        # Note: self.comm is set by the test infrastructure
        self.worldsize = self.comm.size
        if (self.worldsize >= 2):
            self.groupsize = int( self.worldsize / 2 )
            self.ngroup = 2
        else:
            self.groupsize = 1
            self.ngroup = 1
        self.toastcomm = Comm(self.comm, groupsize=self.groupsize)
        self.dist = Dist(self.toastcomm)

        self.dets = ['1a', '1b', '2a', '2b']
        self.flavs = ['proc1', 'proc2']
        self.flavscheck = [Streams.DEFAULT_FLAVOR] + self.flavs
        self.totsamp = 100
        self.rms = 10.0

        # every process group creates some number of observations
        nobs = self.toastcomm.group + 1

        for i in range(nobs):
            # create the streams and pointing for this observation

            pntg = Pointing(
                mpicomm = self.dist.comm.comm_group, 
                timedist = True, 
                detectors = self.dets, 
                samples = self.totsamp
            )

            str = StreamsWhiteNoise(
                mpicomm = self.dist.comm.comm_group,
                timedist = True,
                detectors = self.dets,
                rms = self.rms,
                samples = self.totsamp
            )

            self.dist.obs.append( 
                Obs(
                    mpicomm = self.dist.comm.comm_group, 
                    streams = str, 
                    pointing = pntg, 
                    baselines = None, 
                    noise = None
                )
            )


    def test_copy(self):
        start = MPI.Wtime()

        op = OperatorCopy(timedist=True)

        outdist = op.exec(self.dist)
        
        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))

