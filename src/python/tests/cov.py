# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os
import shutil

import numpy as np
import numpy.testing as nt
import healpy as hp

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_detdata import *
from ..tod.sim_noise import *
from ..map import *

from .. import ctoast as ctoast

class CovarianceTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
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

        self.totsamp = 2000000

        self.map_nside = 32
        self.map_npix = 12 * self.map_nside**2
        
        self.rate = 40.0
        self.spinperiod = 10.0
        self.spinangle = 30.0
        self.precperiod = 50.0
        self.precangle = 65.0
        self.hwprpm = 50.0

        self.subnside = int(self.map_nside / 4)
        self.subnpix = 12 * self.subnside**2
        self.nsubmap = int( self.map_npix / self.subnpix )

        self.NET = 7.0

        self.fmins = {
            'bore' : 0.0,
        }
        self.rates = {
            'bore' : self.rate,
        }
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
                self.toastcomm.comm_group, 
                self.dets, 
                self.totsamp, 
                firsttime=0.0, 
                rate=self.rate, 
                spinperiod=self.spinperiod,
                spinangle=self.spinangle,
                precperiod=self.precperiod, 
                precangle=self.precangle, 
                sampsizes=chunks)

            tod.set_prec_axis()

            # add analytic noise model with white noise

            nse = AnalyticNoise(
                rate=self.rates, 
                fmin=self.fmins,
                detectors=self.detnames,
                fknee=self.fknee,
                alpha=self.alpha,
                NET=self.netd)

            ob = {}
            ob['name'] = 'test'
            ob['id'] = 0
            ob['tod'] = tod
            ob['intervals'] = None
            ob['baselines'] = None
            ob['noise'] = nse

            self.data.obs.append(ob)


    def tearDown(self):
        pass


    def test_accum(self):
        nsm = 2
        npix = 3
        nnz = 4
        block = int(nnz * (nnz+1) / 2)
        scale = 2.0
        nsamp = nsm * npix

        fake = DistPixels(comm=self.toastcomm.comm_group, size=nsm*npix, nnz=nnz, dtype=np.float64, submap=npix, local=np.arange(nsm))
        check = fake.duplicate()

        hits = DistPixels(comm=self.toastcomm.comm_group, size=nsm*npix, nnz=1, dtype=np.int64, submap=npix, local=np.arange(nsm))
        checkhits = hits.duplicate()

        invn = DistPixels(comm=self.toastcomm.comm_group, size=nsm*npix, nnz=block, dtype=np.float64, submap=npix, local=np.arange(nsm))
        checkinvn = invn.duplicate()

        sm = np.zeros(nsamp, dtype=np.int64)
        pix = np.zeros(nsamp, dtype=np.int64)
        wt = np.zeros(nsamp*nnz, dtype=np.float64)

        for i in range(nsamp):
            sm[i] = i % nsm
            pix[i] = i % npix
            for k in range(nnz):
                wt[i*nnz + k] = float(k+1)

        signal = np.random.normal(size=nsamp)

        ctoast.cov_accumulate_diagonal(nsm, npix, nnz, nsamp, sm, pix, wt, scale, 
            signal, fake.data, hits.data, invn.data)

        for i in range(nsamp):
            checkhits.data[sm[i], pix[i], 0] += 1
            off = 0
            for j in range(nnz):
                check.data[sm[i], pix[i], j] += scale * signal[i] * wt[i*nnz+j]
                for k in range(j, nnz):
                    checkinvn.data[sm[i], pix[i], off] += scale * wt[i*nnz+j] * wt[i*nnz+k]
                    off += 1

        # for i in range(nsamp):
        #     print("{}: {} {}".format(i, checkhits.data[sm[i], pix[i], 0], hits.data[sm[i], pix[i], 0]))
        #     off = 0
        #     for j in range(nnz):
        #         print("  {}:  {}  {}".format(j, check.data[sm[i], pix[i], j], fake.data[sm[i], pix[i], j]))
        #         for k in range(j, nnz):
        #             print("    {}:  {}  {}".format(off, checkinvn.data[sm[i], pix[i], off], invn.data[sm[i], pix[i], off]))
        #             off += 1

        nt.assert_equal(hits.data, checkhits.data)
        nt.assert_almost_equal(fake.data, check.data)
        nt.assert_almost_equal(invn.data, checkinvn.data)

        #self.assertTrue(False)

        return


    def test_invert(self):
        nsm = 2
        npix = 3
        nnz = 4
        scale = 2.0
        nsamp = nsm * npix
        nelem = int(nnz * (nnz+1) / 2)
        threshold = 1.0e-6

        invn = DistPixels(comm=self.toastcomm.comm_group, size=nsm*npix, nnz=nelem, dtype=np.float64, submap=npix, local=np.arange(nsm))
        check = invn.duplicate()

        rowdata = 10.0 * np.arange(nnz, 0, -1)

        for i in range(nsm):
            for j in range(npix):
                off = 0
                for k in range(nnz):
                    for m in range(k, nnz):
                        invn.data[i,j,off] = rowdata[m-k]
                        check.data[i,j,off] = invn.data[i,j,off]
                        off += 1

        # invert twice
        covariance_invert(invn, threshold)
        covariance_invert(invn, threshold)

        # for i in range(nsm):
        #     for j in range(npix):
        #         off = 0
        #         print("sm {}, pix {}:".format(i, j))
        #         for k in range(nnz):
        #             for m in range(k, nnz):
        #                 print("  {} {}".format(fakedata[i,j,off], checkdata[i,j,off]))
        #                 off += 1

        nt.assert_almost_equal(invn.data, check.data)
        return


    def test_invnpp(self):
        start = MPI.Wtime()

        op = OpSimNoise(realization=0)
        op.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU', hwprpm=self.hwprpm)
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # find the locally hit submaps.
        allsm = np.floor_divide(localpix, self.subnpix)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)
        #print(localsm)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=6, dtype=np.float64, submap=self.subnpix, local=localsm)
        invnpp.data.fill(0.0)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=1, dtype=np.int64, submap=self.subnpix, local=localsm)
        hits.data.fill(0)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits, name="noise")
        build_invnpp.exec(self.data)

        # for i in range(invnpp.data.shape[0]):
        #     for j in range(invnpp.data.shape[1]):
        #         print("sm {}, pix {}:  hits = {}".format(i, j, hits.data[i,j,0]))
        #         for k in range(invnpp.data.shape[2]):
        #             print("  {}".format(invnpp.data[i,j,k]))

        invnpp.allreduce()
        hits.allreduce()

        # invert it
        check = invnpp.duplicate()
        covariance_invert(invnpp, 1.0e-14)
        covariance_invert(invnpp, 1.0e-14)

        # Matrices that failed the rcond test are set to zero
        nonzero = (np.absolute(invnpp.data) > 1.0e-12)
        if np.sum(nonzero) == 0:
            raise Exception('All matrices failed the rcond test.')

        nt.assert_almost_equal(invnpp.data[nonzero], check.data[nonzero])

        return


    def test_distpix_init(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU', hwprpm=self.hwprpm)
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

        #self.assertTrue(False)

        return


    def test_multiply(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU', hwprpm=self.hwprpm)
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

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        invnpp.allreduce()
        hits.allreduce()

        # invert it
        check = invnpp.duplicate()
        covariance_invert(invnpp, 1.0e-3)

        # multiply the two
        covariance_multiply(check, invnpp)

        # check that the multiplied matrices are unit matrices
        nsubmap, npix, nblock = check.data.shape
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
                            nt.assert_almost_equal( check.data[i,j,off], 1. )
                        else:
                            nt.assert_almost_equal( check.data[i,j,off], 0. )
                        off += 1

        return


    def test_fitsio(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU', hwprpm=self.hwprpm)
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

        rcond = DistPixels(comm=self.toastcomm.comm_group, size=self.map_npix, nnz=1, dtype=np.float64, submap=self.subnpix, local=localsm)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=self.map_npix, nnz=1, dtype=np.int64, submap=self.subnpix, local=localsm)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        #self.assertTrue(False)

        invnpp.allreduce()
        hits.allreduce()

        # invert it

        covariance_invert(invnpp, 1.0e-3)
        rcond = covariance_rcond(invnpp)

        check = invnpp.duplicate()
        checkhits = hits.duplicate()
        checkrcond = rcond.duplicate()

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

        rcondfile = os.path.join(self.mapdir, 'covtest_rcond.fits')
        if self.toastcomm.comm_group.rank == 0:
            if os.path.isfile(rcondfile):
                os.remove(rcondfile)

        invnpp.write_healpix_fits(outfile)
        rcond.write_healpix_fits(rcondfile)
        hits.write_healpix_fits(hitfile)

        print("proc {} invnpp.data on write sum = {}".format(self.toastcomm.comm_group.rank, np.sum(invnpp.data)))

        invnpp.data.fill(0.0)
        invnpp.read_healpix_fits(outfile)

        print("proc {} invnpp.data on read sum = {}".format(self.toastcomm.comm_group.rank, np.sum(invnpp.data)))

        diffdata = invnpp.duplicate()
        diffdata.data -= check.data

        difffile = os.path.join(self.mapdir, 'readwrite_diff.fits')
        diffdata.write_healpix_fits(difffile)

        if self.comm.rank == 0:
            import matplotlib.pyplot as plt

            dat = hp.read_map(outfile)
            outfile = "{}.png".format(outfile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

            dat = hp.read_map(difffile)
            outfile = "{}.png".format(difffile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

            dat = hp.read_map(hitfile)
            outfile = "{}.png".format(hitfile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

            dat = hp.read_map(rcondfile)
            outfile = "{}.png".format(rcondfile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

        rcond.data.fill(0.0)
        rcond.read_healpix_fits(rcondfile)

        nt.assert_almost_equal(rcond.data, checkrcond.data, decimal=6)
        #nt.assert_almost_equal(invnpp.data, checkdata, decimal=6)

        hits.data.fill(0)
        hits.read_healpix_fits(hitfile)

        nt.assert_equal(hits.data, checkhits.data)

        return

