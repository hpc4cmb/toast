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

from toast.mpirunner import MPITestCase

from toast.fod import autocov_psd


class PSDTest(MPITestCase):


    def setUp(self):
        #data
        self.nsamp = 100000
        self.stationary_period = 10000
        self.lagmax = 1000
        self.times = np.arange(self.nsamp)
        self.signal = np.random.randn(self.nsamp)
        self.flags = np.zeros(self.nsamp, dtype=np.bool)


    def test_autocov_psd(self):
        autocovs = autocov_psd(self.times, self.signal, self.flags, self.lagmax, self.stationary_period, comm=self.comm)
