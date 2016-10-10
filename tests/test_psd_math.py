# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

if 'TOAST_NO_MPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

import numpy as np

from toast.mpirunner import MPITestCase

from toast.fod import autocov_psd


class PSDTest(MPITestCase):


    def setUp(self):
        #data
        self.nsamp = 100000
        self.stationary_period = 10000
        self.lagmax = 1000
        self.fsample = 4.0
        self.times = np.arange(self.nsamp) / self.fsample
        self.sigma = 10.
        self.signal = np.random.randn(self.nsamp) * self.sigma
        self.flags = np.zeros(self.nsamp, dtype=np.bool)
        self.flags[int(self.nsamp/4):int(self.nsamp/2)] = True


    def test_autocov_psd(self):
        autocovs = autocov_psd(self.times, self.signal, self.flags, self.lagmax, self.stationary_period, self.fsample, comm=self.comm)

        for i in range(len(autocovs)):
            t0, t1, freq, psd = autocovs[i]

            n = len(psd)
            mn = np.mean( np.abs( psd ) )
            err = np.std( np.abs( psd ) )

            ref = self.sigma**2 / self.fsample
            if np.abs(mn - ref) > err / np.sqrt(n) * 4.:
                raise RuntimeError('White noise input failed to produce a properly normalized white noise spectrum')

        return
