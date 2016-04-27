# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

if 'PYTOAST_NOMPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

import numpy as np
import numpy.testing as nt

from toast.tod.tod import *
from toast.tod.pointing import *
from toast.tod.noise import *
from toast.tod.sim_noise import *
from toast.tod.sim_detdata import *
from toast.tod.sim_tod import *

from toast.mpirunner import MPITestCase


class OpSimNoiseTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
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
        self.toastcomm = Comm(world=self.comm, groupsize=self.groupsize)
        self.data = Data(self.toastcomm)

        self.dets = ["f1a", "f1b", "f2a", "f2b"]
        self.fp = {}
        for d in self.dets:
            self.fp[d] = np.array([0.0, 0.0, 1.0, 0.0])

        self.rate = 10.0
        self.fmin = 0.05
        self.fknee = {}
        self.alpha = {}
        self.NET = {}

        self.fknee["f1a"] = 0.1
        self.alpha["f1a"] = 1.5
        self.NET["f1a"] = 10.0

        self.fknee["f1b"] = 0.1
        self.alpha["f1b"] = 1.5
        self.NET["f1b"] = 10.0

        self.fknee["f2a"] = 0.2
        self.alpha["f2a"] = 1.5
        self.NET["f2a"] = 50.0

        self.fknee["f2b"] = 0.2
        self.alpha["f2b"] = 1.5
        self.NET["f2b"] = 50.0

        self.totsamp = 10000

        # in order to make sure that the noise realization is reproducible
        # all all concurrencies, we set the chunksize to something independent
        # of the number of ranks.

        nchunk = 10
        chunksize = int(self.totsamp / nchunk)
        chunks = np.ones(nchunk, dtype=np.int64)
        chunks *= chunksize
        remain = self.totsamp - (nchunk * chunksize)
        for r in range(remain):
            chunks[r] += 1

        # Construct an empty TOD (no pointing needed)

        self.tod = TODHpixSpiral(mpicomm=self.toastcomm.comm_group, detectors=self.fp, samples=self.totsamp, firsttime=0.0, rate=self.rate, nside=512, sizes=chunks)

        # construct an analytic noise model

        self.nse = AnalyticNoise(rate=self.rate, fmin=self.fmin, detectors=self.dets, fknee=self.fknee, alpha=self.alpha, NET=self.NET)

        ob = {}
        ob['id'] = 'noisetest-{}'.format(self.toastcomm.group)
        ob['tod'] = self.tod
        ob['intervals'] = None
        ob['baselines'] = None
        ob['noise'] = self.nse

        self.data.obs.append(ob)


    def test_sim(self):
        start = MPI.Wtime()

        # generate timestreams

        op = OpSimNoise(stream=123456)
        op.exec(self.data)

        ob = self.data.obs[0]
        tod = ob['tod']
        nse = ob['noise']

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_simnoise_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        # verify that the white noise part of the spectrum is normalized correctly

        if self.comm.rank == 0:
            np.savetxt(os.path.join(self.outdir,"out_test_simnoise_psd.txt"), np.transpose([nse.freq, nse.psd(self.dets[0]), nse.psd(self.dets[1]), nse.psd(self.dets[2]), nse.psd(self.dets[3])]), delimiter=' ')

        fsamp = nse.rate
        cutoff = 0.9 * (fsamp / 2.0)
        indx = np.where(nse.freq > cutoff)
        for det in tod.local_dets:
            NET = self.nse.NET(det)
            knee = self.nse.fknee(det)
            avg = np.mean(nse.psd(det)[indx])
            NETsq = NET*NET
            self.assertTrue((np.absolute(avg - NETsq)/NETsq) < 0.01)

        # write timestreams to disk for debugging

        check1 = tod.cache.reference("noise_{}".format(self.dets[0]))
        check2 = tod.cache.reference("noise_{}".format(self.dets[1]))
        check3 = tod.cache.reference("noise_{}".format(self.dets[2]))
        check4 = tod.cache.reference("noise_{}".format(self.dets[3]))

        if self.comm.rank == 0:
            np.savetxt(os.path.join(self.outdir,"out_test_simnoise_tod.txt"), np.transpose([check1, check2, check3, check4]), delimiter='\n')

        # verify that timestreams with the same PSD *DO NOT* have the same
        # values (this is a crude test that the random seeds are being incremented)

        dif = np.fabs(check1 - check2)
        check = np.mean(dif)
        self.assertTrue(check > (0.01 / np.sqrt(self.totsamp)))

        dif = np.fabs(check3 - check4)
        check = np.mean(dif)
        self.assertTrue(check > (0.01 / np.sqrt(self.totsamp)))
        
        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("simnoise test took {:.3f} s".format(elapsed))

