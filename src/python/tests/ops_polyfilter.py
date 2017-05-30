# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

import numpy as np
import numpy.testing as nt

import scipy.interpolate as si

from ..tod import OpPolyFilter
from ..tod.tod import *
from ..tod.pointing import *
from ..tod.noise import *
from ..tod.sim_noise import *
from ..tod.sim_det_noise import *
from ..tod.sim_tod import *

from .. import rng as rng


class OpPolyFilterTest(MPITestCase):

    def setUp(self):
        self.outdir = 'toast_test_output'
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

        self.order = 5

        self.dets = ['f1a', 'f1b', 'f2a', 'f2b', 'white', 'high']
        self.fp = {}
        for d in self.dets:
            self.fp[d] = np.array([0.0, 0.0, 1.0, 0.0])

        self.rate = 20.0

        self.rates = {}
        self.fmin = {}
        self.fknee = {}
        self.alpha = {}
        self.NET = {}

        self.rates['f1a'] = self.rate
        self.fmin['f1a'] = 1.0e-5
        self.fknee['f1a'] = 0.15
        self.alpha['f1a'] = 1.0
        self.NET['f1a'] = 10.0

        self.rates['f1b'] = self.rate
        self.fmin['f1b'] = 1.0e-5
        self.fknee['f1b'] = 0.1
        self.alpha['f1b'] = 1.0
        self.NET['f1b'] = 10.0

        self.rates['f2a'] = self.rate
        self.fmin['f2a'] = 1.0e-5
        self.fknee['f2a'] = 0.05
        self.alpha['f2a'] = 1.0
        self.NET['f2a'] = 10.0

        self.rates['f2b'] = self.rate
        self.fmin['f2b'] = 1.0e-5
        self.fknee['f2b'] = 0.001
        self.alpha['f2b'] = 1.0
        self.NET['f2b'] = 10.0

        self.rates['white'] = self.rate
        self.fmin['white'] = 0.0
        self.fknee['white'] = 0.0
        self.alpha['white'] = 1.0
        self.NET['white'] = 10.0

        self.rates['high'] = self.rate
        self.fmin['high'] = 1.0e-5
        self.fknee['high'] = 40.0
        self.alpha['high'] = 2.0
        self.NET['high'] = 10.0

        self.totsamp = 10000

        self.oversample = 2

        nchunk = 10
        chunksize = int(self.totsamp / nchunk)
        chunks = np.ones(nchunk, dtype=np.int64)
        chunks *= chunksize
        remain = self.totsamp - (nchunk * chunksize)
        for r in range(remain):
            chunks[r] += 1

        self.chunksize = chunksize

        # Construct an empty TOD (no pointing needed)

        self.tod = TODHpixSpiral(
            self.toastcomm.comm_group, 
            self.fp, 
            self.totsamp, 
            firsttime=0.0, 
            rate=self.rate, 
            nside=512, 
            sampsizes=chunks)

        # construct an analytic noise model

        self.nse = AnalyticNoise(
            rate=self.rates, 
            fmin=self.fmin, 
            detectors=self.dets, 
            fknee=self.fknee, 
            alpha=self.alpha, 
            NET=self.NET
        )

        ob = {}
        ob['name'] = 'noisetest-{}'.format(self.toastcomm.group)
        ob['id'] = 0
        ob['tod'] = self.tod
        ob['intervals'] = None
        ob['baselines'] = None
        ob['noise'] = self.nse

        self.data.obs.append(ob)

    def test_filter(self):
        start = MPI.Wtime()

        ob = self.data.obs[0]
        tod = ob['tod']
        nse = ob['noise']

        # generate timestreams

        op = OpSimNoise()
        op.exec(self.data)

        # Replace the noise with a polynomial fit

        old_rms = {}

        for det in tod.local_dets:
            cachename = 'noise_{}'.format(det)
            y = tod.cache.reference(cachename)
            x = np.arange(y.size)
            p = np.polyfit(x, y, self.order)
            y[:] = np.polyval(p, x)
            old_rms[det] = np.std(y)

        # Filter timestreams

        op = OpPolyFilter(name='noise', order=self.order)
        op.exec(self.data)

        stop = MPI.Wtime()
        elapsed = stop - start

        # Ensure all timestreams are zeroed out by the filter
        
        for det in tod.local_dets:
            cachename = 'noise_{}'.format(det)
            y = tod.cache.reference(cachename)
            rms = np.std(y)
            old = old_rms[det]
            if rms / old > 1e-10:
                raise RuntimeError('det {} old rms = {}, new rms = {}'.format(det, old, rms))

        self.print_in_turns('polyfilter test took {:.3f} s'.format(elapsed))

