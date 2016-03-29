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
import toast.map._helpers as mh

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


    def test_accum(self):
        nsm = 2
        npix = 3
        nnz = 4
        scale = 2.0
        nsamp = nsm * npix
        nelem = int(nnz * (nnz+1) / 2)
        fakedata = np.zeros((nsm, npix, nelem), dtype=np.float64)
        fakehits = np.zeros((nsm, npix, 1), dtype=np.int64)
        checkdata = np.zeros((nsm, npix, nelem), dtype=np.float64)
        checkhits = np.zeros((nsm, npix, 1), dtype=np.int64)
        sm = np.repeat(np.arange(nsm, dtype=np.int64), npix)
        pix = np.tile(np.arange(npix, dtype=np.int64), nsm)
        wt = np.tile(np.arange(nnz, dtype=np.float64), nsamp).reshape(-1, nnz)

        mh._accumulate_inverse_covariance(fakedata, sm, pix, wt, scale, fakehits)

        for i in range(nsamp):
            checkhits[sm[i], pix[i], 0] += 1
            off = 0
            for j in range(nnz):
                for k in range(j, nnz):
                    checkdata[sm[i], pix[i], off] += scale * wt[i,j] * wt[i,k]
                    off += 1

        nt.assert_equal(fakehits, checkhits)
        nt.assert_almost_equal(fakedata, checkdata)
        return


    def test_invert(self):
        nsm = 2
        npix = 3
        nnz = 4
        scale = 2.0
        nsamp = nsm * npix
        nelem = int(nnz * (nnz+1) / 2)
        threshold = 1.0e-6
        fakedata = np.zeros((nsm, npix, nelem), dtype=np.float64)
        checkdata = np.zeros((nsm, npix, nelem), dtype=np.float64)
        rowdata = 10.0 * np.arange(nnz, 0, -1)

        for i in range(nsm):
            for j in range(npix):
                off = 0
                for k in range(nnz):
                    for m in range(k, nnz):
                        fakedata[i,j,off] = rowdata[m-k]
                        checkdata[i,j,off] = fakedata[i,j,off]
                        off += 1

        # invert twice
        mh._invert_covariance(fakedata, threshold)
        mh._invert_covariance(fakedata, threshold)

        nt.assert_almost_equal(fakedata, checkdata)
        return


    def test_invnpp(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU')
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # pick a submap size and find the locally hit submaps.
        submapsize = np.floor_divide(self.sim_nside, 16)
        allsm = np.floor_divide(localpix, submapsize)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map to store the covariance and hits
        npix = 12 * self.sim_nside * self.sim_nside
        
        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=npix, nnz=6, dtype=np.float64, submap=submapsize, local=localsm)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=npix, nnz=1, dtype=np.int64, submap=submapsize, local=localsm)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        build_invnpp = OpInvCovariance(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        # invert it
        covariance_invert(invnpp.data, 1.0e-3)

        # write this out...

        return

