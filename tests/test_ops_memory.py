# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from .mpirunner import MPITestCase
import sys

from toast.tod.tod import *
from toast.tod.memory import *
from toast.tod.sim import *

import quaternionarray as qa


class OpCopyTest(MPITestCase):

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
        self.data = Data(self.toastcomm)

        spread = 0.1 * np.pi / 180.0
        angterm = np.cos(spread / 2.0)
        axiscoef = np.sin(spread / 2.0)

        self.dets = {
            '1a' : np.array([axiscoef, 0.0, 0.0, angterm]),
            '1b' : np.array([-axiscoef, 0.0, 0.0, angterm]),
            '2a' : np.array([0.0, axiscoef, 0.0, angterm]),
            '2b' : np.array([0.0, -axiscoef, 0.0, angterm])
            }
        self.flavs = ['proc1', 'proc2']
        self.flavscheck = [TOD.DEFAULT_FLAVOR] + self.flavs
        self.totsamp = 100
        self.rms = 10.0

        # every process group creates some number of observations
        nobs = self.toastcomm.group + 1

        for i in range(nobs):
            # create the TOD for this observation

            tod = TODFake(
                mpicomm = self.data.comm.comm_group, 
                detectors = self.dets,
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


    def test_copy(self):
        start = MPI.Wtime()

        op = OpCopy(timedist=True)

        outdata = op.exec(self.data)
        
        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("copy test took {:.3f} s".format(elapsed))

