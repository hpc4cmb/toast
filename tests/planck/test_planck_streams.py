# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from ..mpirunner import MPITestCase
import sys

import os

from toast.planck.streams import *


class StreamsPlanckEFFTest(MPITestCase):


    def setUp(self):
        # Note: self.comm is set by the test infrastructure
        self.dets = ['100-1a','100-1b','100-4a','100-4b']
        self.ringdb = 'testdata/rings.db'
        if not os.path.isfile(self.ringdb):
            self.no_data = True
            return
        else:
            self.no_data = False
        self.effdir = 'testdata/EFF'
        self.freq = 100
        self.eff = StreamsPlanckEFF(mpicomm=self.comm, timedist=True, detectors=self.dets, ringdb=self.ringdb, effdir=self.effdir, freq=self.freq)


    def test_props(self):
        if self.no_data: return
        
        start = MPI.Wtime()
        
        #self.assertEqual(self.eff.detectors, self.dets)

        print( 'Running Planck stream properties test. Total samples {}, local samples {}'.format( self.eff.total_samples, self.eff.local_samples ) )

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))


    def test_readwrite(self):
        if self.no_data: return
        
        print('Running Planck EFF write test')        

        start = MPI.Wtime()

        n = 10

        for i, d in enumerate(self.dets):
            dat = np.ones(n)*i
            flg = np.zeros(n, dtype=np.byte)
            self.eff.write(detector=d, local_start=0, data=dat, flags=flg)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))

        print('Running Planck EFF read test')
        
        start = MPI.Wtime()

        print( 'Local samples: {}'.format(self.eff.local_samples) )
        print( 'Local detectors: {}'.format(self.eff.local_dets) )

        for i, d in enumerate(self.dets):
            data, flags = self.eff.read(detector=d, local_start=0, n=n)
            np.testing.assert_almost_equal(data, np.ones(n)*i)
            np.testing.assert_equal(flags, np.zeros(n, dtype=np.byte) )
            print( 'Planck EFF Steam read: {}: \n {}, {} \n'.format(d, data, flags) )

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))

