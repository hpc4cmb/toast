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


class BinnedTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "binned")
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

        self.hwprpm = 50

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

        # in order to make sure that the noise realization is reproducible
        # all all concurrencies, we set the chunksize to something independent
        # of the number of ranks.

        nchunk = 10
        chunksize = int(self.totsamp / nchunk)
        chunks = np.ones(nchunk, dtype=np.int64)
        chunks *= chunksize
        remain = self.totsamp - (nchunk * chunksize)
        for r in range(remain):
            chunks[r] += 1

        self.chunksize = chunksize

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
            ob['intervals'] = None
            ob['baselines'] = None
            ob['noise'] = nse

            self.data.obs.append(ob)


    def test_zmap_accum(self):
        nsm = 2
        npix = 3
        nnz = 4
        scale = 2.0
        nsamp = nsm * npix

        fakedata = np.zeros((nsm, npix, nnz), dtype=np.float64)
        fakehits = np.zeros((nsm, npix, 1), dtype=np.int64)
        checkdata = np.zeros((nsm, npix, nnz), dtype=np.float64)
        checkhits = np.zeros((nsm, npix, 1), dtype=np.int64)
        sm = np.repeat(np.arange(nsm, dtype=np.int64), npix)
        pix = np.tile(np.arange(npix, dtype=np.int64), nsm)
        wt = np.tile(np.arange(nnz, dtype=np.float64), nsamp).reshape(-1, nnz)

        signal = np.random.normal(size=nsamp)

        nh._accumulate_noiseweighted(fakedata, sm, signal, pix, wt, scale, fakehits)

        for i in range(nsamp):
            checkhits[sm[i], pix[i], 0] += 1
            for j in range(nnz):
                checkdata[sm[i], pix[i], j] += scale * signal[i] * wt[i,j]

        nt.assert_equal(fakehits, checkhits)
        nt.assert_almost_equal(fakedata, checkdata)
        return


    def test_binned(self):
        start = MPI.Wtime()

        # generate noise timestreams from the noise model
        nsig = OpSimNoise(stream=0)
        nsig.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode='IQU', hwprpm=self.hwprpm)
        pointing.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.mapdir,"info.txt"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # find the locally hit submaps.
        allsm = np.floor_divide(localpix, self.subnpix)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct distributed maps to store the covariance,
        # noise weighted map, and hits

        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=6, dtype=np.float64, submap=self.subnpix, local=localsm)

        zmap = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=3, dtype=np.float64, submap=self.subnpix, local=localsm)

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
        covariance_invert(invnpp.data, 1.0e-6)

        # accumulate the noise weighted map

        build_zmap = OpNoiseWeighted(zmap=zmap, detweights=detweights, name="noise")
        build_zmap.exec(self.data)

        zmap.allreduce()

        # compute the binned map, N_pp x Z

        covariance_apply(invnpp.data, zmap.data)
        zmap.write_healpix_fits(os.path.join(self.mapdir, "binned.fits"))

        # compare with MADAM

        madam_out = os.path.join(self.mapdir, "madam")
        if self.comm.rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

        pars = {}
        pars[ 'temperature_only' ] = 'F'
        pars[ 'force_pol' ] = 'T'
        pars[ 'kfirst' ] = 'F'
        pars[ 'base_first' ] = 1.0
        pars[ 'fsample' ] = self.rate
        pars[ 'nside_map' ] = self.map_nside
        pars[ 'nside_cross' ] = self.map_nside
        pars[ 'nside_submap' ] = self.map_nside
        pars[ 'pixlim_cross' ] = 1.0e-6
        pars[ 'pixlim_map' ] = 1.0e-6
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = madam_out

        madam = OpMadam(params=pars, detweights=detweights, name="noise")

        if madam.available:
            madam.exec(self.data)

            if self.comm.rank == 0:
                hitsfile = os.path.join(madam_out, 'madam_hmap.fits')
                hits = hp.read_map(hitsfile, nest=True)

                outfile = "{}.png".format(hitsfile)
                hp.mollview(hits, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                binfile = os.path.join(madam_out, 'madam_bmap.fits')
                bins = hp.read_map(binfile, nest=True)

                outfile = "{}.png".format(binfile)
                hp.mollview(bins, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                toastfile = os.path.join(self.mapdir, 'binned.fits')
                toastbins = hp.read_map(toastfile, nest=True)

                outfile = "{}.png".format(toastfile)
                hp.mollview(bins, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # compare binned map to madam output

                tothits = np.sum(hits)
                nt.assert_equal(self.totsamp, tothits)

                mask = (bins > -1.0e20)

                nt.assert_almost_equal(bins[mask], toastbins[mask], decimal=4)


            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))
        else:
            print("libmadam not available, skipping tests")

        return

