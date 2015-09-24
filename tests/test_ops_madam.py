# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI
from .mpirunner import MPITestCase
import sys

from toast.tod.tod import *
from toast.tod.memory import *
from toast.tod.pointing import *
from toast.tod.sim import *
from toast.map.madam import *


class OpMadamTest(MPITestCase):

    def setUp(self):
        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = Data(self.toastcomm)

        self.dets = {
            'bore' : np.array([0.0, 0.0, 1.0, 0.0])
            }

        self.totsamp = 100000
        self.rms = 10.0
        self.nside = 256
        self.rate = 50.0

        # madam only supports a single observation
        nobs = 1

        for i in range(nobs):
            # create the TOD for this observation

            tod = TODFake(
                mpicomm=self.toastcomm.comm_group, 
                detectors=self.dets,
                samples=self.totsamp,
                rate=self.rate
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

        # cache the data in memory
        cache = OpCopy()
        data = cache.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingFake(nside=self.nside)
        pointing.exec(data)

        pars = {}
        pars[ 'base_first' ] = 1.0
        pars[ 'fsample' ] = self.rate
        pars[ 'nside_map' ] = self.nside
        pars[ 'nside_cross' ] = self.nside
        pars[ 'nside_submap' ] = self.nside
        pars[ 'write_map' ] = True
        pars[ 'write_binmap' ] = True
        pars[ 'write_matrix' ] = True
        pars[ 'write_wcov' ] = True
        pars[ 'write_hits' ] = True
        pars[ 'kfilter' ] = False
        pars[ 'path_output' ] = './'

        madam = OpMadam(params=pars)
        madam.exec(data)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

