# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from .mpirunner import MPITestCase
import sys
import os

from toast.tod.tod import *
from toast.tod.pointing import *
from toast.tod.sim import *
from toast.tod.noise import *


class OpSimNoiseTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

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

        self.dets = ["fake"]
        self.totsamp = 10000

        self.tod = TOD(mpicomm=self.toastcomm.comm_group, timedist=True, detectors=self.dets, flavors=None, samples=self.totsamp)

        self.tod.write_times(local_start=0, stamps=np.linspace(0.0, 0.01*float(self.tod.local_samples), num=self.tod.local_samples))

        self.freq = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 2.4, 3.2, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
        self.psd = np.zeros_like(self.freq)
        self.psd[0] = 200.0
        self.psd[1:] = 10.0 + 100.0 / (self.freq[1:]**1.2)

        self.nse = Noise(detectors=self.dets, freq=self.freq, psds={self.dets[0] : self.psd})

        ob = {}
        ob['id'] = 'noisetest-{}'.format(self.toastcomm.group)
        ob['tod'] = self.tod
        ob['intervals'] = []
        ob['baselines'] = None
        ob['noise'] = self.nse

        self.data.obs.append(ob)


    def test_sim(self):
        start = MPI.Wtime()

        op = OpSimNoise(stream=123456)
        op.exec(self.data)

        np.savetxt(os.path.join(self.outdir,"out_test_simnoise.txt"), self.tod.read(detector='fake', local_start=0, n=self.tod.local_samples), delimiter='\n')
        
        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("copy test took {:.3f} s".format(elapsed))

