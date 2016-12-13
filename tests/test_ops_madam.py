# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os
import shutil

if 'TOAST_NO_MPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

from toast.tod.tod import *
from toast.tod.pointing import *
from toast.tod.sim_tod import *
from toast.tod.sim_detdata import *
from toast.map.madam import *

from toast.mpirunner import MPITestCase


class OpMadamTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "madam")
        if self.comm.rank == 0:
            if os.path.isdir(self.mapdir):
                shutil.rmtree(self.mapdir)
            os.mkdir(self.mapdir)

        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = Data(self.toastcomm)

        self.dets = {
            'bore' : np.array([0.0, 0.0, 1.0, 0.0])
            }

        self.sim_nside = 64
        self.totsamp = 3 * 49152
        #self.totsamp = 20
        self.rms = 10.0
        self.map_nside = 64
        self.rate = 50.0

        # madam only supports a single observation
        nobs = 1

        for i in range(nobs):
            # create the TOD for this observation

            tod = TODHpixSpiral(
                mpicomm=self.toastcomm.comm_group, 
                detectors=self.dets,
                samples=self.totsamp,
                rate=self.rate,
                nside=self.sim_nside
            )

            ob = {}
            ob['name'] = 'test'
            ob['id'] = 0
            ob['tod'] = tod
            ob['intervals'] = None
            ob['baselines'] = None
            ob['noise'] = None

            self.data.obs.append(ob)


    def test_madam_gradient(self):
        start = MPI.Wtime()

        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_madam_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        pars = {}
        pars[ 'kfirst' ] = 'F'
        pars[ 'base_first' ] = 1.0
        pars[ 'fsample' ] = self.rate
        pars[ 'nside_map' ] = self.map_nside
        pars[ 'nside_cross' ] = self.map_nside
        pars[ 'nside_submap' ] = self.map_nside
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = self.mapdir
        pars[ 'info' ] = 0

        madam = OpMadam(params=pars, name='grad', dets=self.dets)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))
        else:
            print("libmadam not available, skipping tests")

    def test_madam_output(self):
        start = MPI.Wtime()

        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_madam_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        pars = {}
        pars[ 'kfirst' ] = 'T'
        pars[ 'iter_max' ] = 100
        pars[ 'base_first' ] = 1.0
        pars[ 'fsample' ] = self.rate
        pars[ 'nside_map' ] = self.map_nside
        pars[ 'nside_cross' ] = self.map_nside
        pars[ 'nside_submap' ] = self.map_nside
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = self.mapdir
        pars[ 'info' ] = 0

        madam = OpMadam(params=pars, name='grad', name_out='destriped', dets=self.dets)
        if madam.available:
            tod = self.data.obs[0]['tod']
            det = 'bore'
            ref_in = tod.cache.reference('grad_'+det)
            rms0 = np.std(ref_in)
            ref_in[ref_in.size//2:] += 1e6 # Add an offset
            rms1 = np.std(ref_in)

            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

            ref_out = tod.cache.reference('destriped_'+det)
            rms2 = np.std(ref_out)
            if rms1 < 0.9*rms2:
                raise Exception('Destriped TOD does not have lower RMS')
        else:
            print("libmadam not available, skipping tests")
