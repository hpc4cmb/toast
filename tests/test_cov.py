# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI

import sys
import os
import shutil

import matplotlib.pyplot as plt

import numpy as np
import numpy.testing as nt
import healpy as hp

from toast.tod.tod import *
from toast.tod.memory import *
from toast.tod.pointing import *
from toast.tod.sim_tod import *
from toast.tod.sim_detdata import *
from toast.tod.sim_noise import *
from toast.map import *

from toast.mpirunner import MPITestCase


class CovarianceTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "covariance")
        if not os.path.isdir(self.mapdir):
            os.mkdir(self.mapdir)

        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = Data(self.toastcomm)

        self.detnames = ['bore']
        self.dets = {
            'bore' : np.array([0.0, 0.0, 1.0, 0.0])
            }

        self.sim_nside = 64
        self.totsamp = 200000
        self.map_nside = 64
        self.rate = 40.0
        self.spinperiod = 10.0
        self.spinangle = 30.0
        self.precperiod = 50.0
        self.precangle = 65.0

        self.NET = 7.0

        # madam only supports a single observation
        nobs = 1

        # give every process one chunk
        nchunk = self.toastcomm.group_size
        chunksize = int(self.totsamp / nchunk)
        chunks = np.ones(nchunk, dtype=np.int64)
        chunks *= chunksize
        remain = self.totsamp - (nchunk * chunksize)
        for r in range(remain):
            chunks[r] += 1

        for i in range(nobs):
            # create the TOD for this observation

            tod = TODSatellite(
                mpicomm=self.toastcomm.comm_group, 
                detectors=self.dets, 
                samples=self.totsamp, 
                firsttime=0.0, 
                rate=self.rate, 
                spinperiod=self.spinperiod,
                spinangle=self.spinangle,
                precperiod=self.precperiod, 
                precangle=self.precangle, 
                sizes=chunks)

            tod.set_prec_axis()

            # add analytic noise model with white noise

            nse = AnalyticNoise(
                rate=self.rate, 
                fmin=0.0,
                detectors=self.detnames,
                fknee=[0.0,], 
                alpha=[1.0,], 
                NET=[self.NET,])

            ob = {}
            ob['id'] = 'test'
            ob['tod'] = tod
            ob['intervals'] = []
            ob['baselines'] = None
            ob['noise'] = nse

            self.data.obs.append(ob)


    def test_invnpp(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpixSimple(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # pick a submap size and find the locally hit submaps.
        submapsize = np.floor_divide(self.sim_nside, 16)
        allsm = np.floor_divide(localpix, submapsize)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map to store the covariance
        npix = 12 * self.sim_nside * self.sim_nside
        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=npix, nnz=1, dtype=np.float64, submap=submapsize, local=localsm)

        # accumulate the inverse covariance
        build_invnpp = OpInvCovariance(invnpp=invnpp)
        build_invnpp.exec(self.data)

        
        

        # write this out...

        return

