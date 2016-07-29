# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os
import shutil

if 'PYTOAST_NOMPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

import matplotlib.pyplot as plt

import numpy as np
import numpy.testing as nt
import healpy as hp

from toast.tod.tod import *
from toast.tod.pointing import *
from toast.tod.sim_tod import *
from toast.tod.sim_detdata import *
from toast.tod.sim_noise import *
from toast.map import *
import toast.map._noise as nh

from toast.mpirunner import MPITestCase


class CovarianceTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "covariance")
        if self.comm.rank == 0:
            if not os.path.isdir(self.mapdir):
                os.mkdir(self.mapdir)

        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = Data(self.toastcomm)

        self.detnames = ['bore']
        self.dets = {
            'bore' : np.array([0.0, 0.0, 1.0, 0.0])
            }

        self.sim_nside = 32
        self.sim_npix = 12 * self.sim_nside**2

        self.totsamp = 200000

        self.map_nside = 32
        self.map_npix = 12 * self.map_nside**2
        
        self.rate = 40.0
        self.spinperiod = 10.0
        self.spinangle = 30.0
        self.precperiod = 50.0
        self.precangle = 65.0

        self.subnside = int(self.map_nside / 4)
        self.subnpix = 12 * self.subnside**2
        self.nsubmap = int( self.map_npix / self.subnpix )

        self.NET = 7.0

        self.fknee = {
            'bore' : 0.0,
        }
        self.alpha = {
            'bore' : 1.0,
        }
        self.netd = {
            'bore' : self.NET
        }

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
                fknee=self.fknee,
                alpha=self.alpha,
                NET=self.netd)

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

        nh._accumulate_inverse_covariance(fakedata, sm, pix, wt, scale, fakehits)

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
        nh._invert_covariance(fakedata, threshold)
        nh._invert_covariance(fakedata, threshold)

        nt.assert_almost_equal(fakedata, checkdata)
        return


    def test_invnpp(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU')
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # find the locally hit submaps.
        allsm = np.floor_divide(localpix, self.subnpix)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=6, dtype=np.float64, submap=self.subnpix, local=localsm)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=1, dtype=np.int64, submap=self.subnpix, local=localsm)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        build_invnpp = OpInvCovariance(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        invnpp.allreduce()
        hits.allreduce()

        # invert it
        checkdata = np.copy(invnpp.data)
        covariance_invert(invnpp.data, 1.0e-3)
        covariance_invert(invnpp.data, 1.0e-3)

        # Matrices that failed the rcond test are set to zero
        nonzero = invnpp.data != 0
        if np.sum(nonzero) == 0: raise Exception('All matrices failed the rcond test.')

        nt.assert_almost_equal(invnpp.data[nonzero], checkdata[nonzero])

        return


    def test_distpix_init(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU')
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # find the locally hit submaps.
        allsm = np.floor_divide(localpix, self.subnpix)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=6, dtype=np.float64, submap=self.subnpix, local=localsm)
        invnpp2 = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=6, dtype=np.float64, submap=self.subnpix, localpix=localpix)

        nt.assert_equal( invnpp.local, invnpp2.local )

        return


    def test_multiply(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU')
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # find the locally hit submaps.
        allsm = np.floor_divide(localpix, self.subnpix)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=6, dtype=np.float64, submap=self.subnpix, local=localsm)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=1, dtype=np.int64, submap=self.subnpix, local=localsm)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        build_invnpp = OpInvCovariance(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        invnpp.allreduce()
        hits.allreduce()

        # invert it
        checkdata = np.copy(invnpp.data)
        covariance_invert(invnpp.data, 1.0e-3)

        # multiply the two
        covariance_multiply(checkdata, invnpp.data)

        # check that the multiplied matrices are unit matrices
        nsubmap, npix, nblock = checkdata.shape
        nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
        for i in range(nsubmap):
            for j in range(npix):
                if np.all( invnpp.data[i,j] == 0 ):
                    # Matrix failed to invert
                    continue
                off = 0                
                for k in range(nnz):
                    for m in range(k, nnz):
                        if k == m:
                            nt.assert_almost_equal( checkdata[i,j,off], 1. )
                        else:
                            nt.assert_almost_equal( checkdata[i,j,off], 0. )
                        off += 1

        return


    def test_fitsio(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU')
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # find the locally hit submaps.
        allsm = np.floor_divide(localpix, self.subnpix)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=self.map_npix, nnz=6, dtype=np.float64, submap=self.subnpix, local=localsm)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=self.map_npix, nnz=1, dtype=np.int64, submap=self.subnpix, local=localsm)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        build_invnpp = OpInvCovariance(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        #self.assertTrue(False)

        invnpp.allreduce()
        hits.allreduce()

        # invert it
        covariance_invert(invnpp.data, 1.0e-3)

        checkdata = np.copy(invnpp.data)

        checkhits = np.copy(hits.data)

        # write this out...

        subsum = [ np.sum(invnpp.data[x,:,:]) for x in range(len(invnpp.local)) ]

        print("proc {} submap sum = {} ({})".format(self.toastcomm.comm_group.rank, " ".join([ "{}:{}".format(x,y) for x,y in zip(invnpp.local, subsum) ]), np.sum(subsum)))

        outfile = os.path.join(self.mapdir, 'covtest.fits')
        if self.toastcomm.comm_group.rank == 0:
            if os.path.isfile(outfile):
                os.remove(outfile)

        hitfile = os.path.join(self.mapdir, 'covtest_hits.fits')
        if self.toastcomm.comm_group.rank == 0:
            if os.path.isfile(hitfile):
                os.remove(hitfile)

        invnpp.write_healpix_fits(outfile)

        print("proc {} invnpp.data on write sum = {}".format(self.toastcomm.comm_group.rank, np.sum(invnpp.data)))

        invnpp.data.fill(0.0)
        invnpp.read_healpix_fits(outfile)

        print("proc {} invnpp.data on read sum = {}".format(self.toastcomm.comm_group.rank, np.sum(invnpp.data)))

        nt.assert_almost_equal(invnpp.data, checkdata)

        hits.write_healpix_fits(hitfile)

        hits.data.fill(0)
        hits.read_healpix_fits(hitfile)

        nt.assert_equal(hits.data, checkhits)

        return

