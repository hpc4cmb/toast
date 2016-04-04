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


class MapSatelliteTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "map_satellite")
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

        # this is an upolarized detector...
        self.epsilon = {
            'bore' : 1.0,
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


    def test_boresight_null(self):
        # verify that if all angles are zero, we get fixed pointing
        # along the x-axis.

        nsim = 1000

        zaxis = np.array([0,0,1], dtype=np.float64)

        borequat = satellite_scanning(nsim=1000, qprec=None, samplerate=100.0, spinperiod=1.0, spinangle=0.0, precperiod=20.0, precangle=0.0)

        data = qa.rotate(borequat, np.tile(zaxis, nsim).reshape(-1,3))

        np.testing.assert_almost_equal(data, np.tile(np.array([1.0, 0.0, 0.0]), nsim).reshape(-1,3))
        return


    def test_grad(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon)
        pointing.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_satellite_grad_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.mapdir, "madam_grad")
        if self.comm.rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

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
        pars[ 'path_output' ] = madam_out

        madam = OpMadam(params=pars)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

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

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.totsamp, tothits)

                sig = grad.sigmap()
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")


    def test_noise(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # generate noise timestreams from the noise model
        nsig = OpSimNoise(stream=12345)
        nsig.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon)
        pointing.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_satellite_noise_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        # For noise weighting in madam, we know we are using an analytic noise
        # and so we can use noise weights based on the NET.  This is instrument
        # specific.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        # make a binned map with madam
        madam_out = os.path.join(self.mapdir, "madam_noise")
        if self.comm.rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

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
        pars[ 'path_output' ] = madam_out

        madam = OpMadam(params=pars, detweights=detweights)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

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

                # check that pixel rms makes sense given the
                # number of hits and the timestream rms

                tothits = np.sum(hits)
                nt.assert_equal(self.totsamp, tothits)
                print("tothits = ", tothits)

                mask = (bins > -1.0e20)
                print("num good pix = ", len(mask))
                rthits = np.sqrt(hits[mask].astype(np.float64))
                print("rthits = ", rthits)
                print("bmap = ", bins[mask])
                weighted = bins[mask] * rthits
                pixrms = np.std(weighted)
                todrms = self.NET * np.sqrt(self.rate)
                relerr = np.absolute(pixrms - todrms) / todrms
                print("pixrms = ", pixrms)
                print("todrms = ", todrms)
                print("relerr = ", relerr)
                self.assertTrue(relerr < 0.01)

        else:
            print("libmadam not available, skipping tests")


    def test_scanmap(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon)
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # construct a sky gradient operator, just to get the signal
        # map- we are not going to use the operator on the data.
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        sig = grad.sigmap()

        # pick a submap size and find the local submaps.
        submapsize = np.floor_divide(self.sim_nside, 16)
        allsm = np.floor_divide(localpix, submapsize)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map which has the gradient        
        npix = 12 * self.sim_nside * self.sim_nside
        distsig = DistPixels(comm=self.toastcomm.comm_group, size=npix, nnz=1, dtype=np.float64, submap=submapsize, local=localsm)
        lsub, lpix = distsig.global_to_local(localpix)
        distsig.data[lsub,lpix,:] = np.array([ sig[x] for x in localpix ]).reshape(-1, 1)

        # create TOD from map
        scansim = OpSimScan(distmap=distsig)
        scansim.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_satellite_scanmap_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.mapdir, "madam_scansim")
        if self.comm.rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

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
        pars[ 'path_output' ] = madam_out

        madam = OpMadam(params=pars)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

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

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.totsamp, tothits)
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")


    def test_hwpfast(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # make a pointing matrix with a HWP that rotates 2*PI every sample
        hwprate = self.rate * 60.0
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon, hwprpm=hwprate)
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # construct a sky gradient operator, just to get the signal
        # map- we are not going to use the operator on the data.
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        sig = grad.sigmap()

        # pick a submap size and find the local submaps.
        submapsize = np.floor_divide(self.sim_nside, 16)
        allsm = np.floor_divide(localpix, submapsize)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map which has the gradient        
        npix = 12 * self.sim_nside * self.sim_nside
        distsig = DistPixels(comm=self.toastcomm.comm_group, size=npix, nnz=1, dtype=np.float64, submap=submapsize, local=localsm)
        lsub, lpix = distsig.global_to_local(localpix)
        distsig.data[lsub,lpix,:] = np.array([ sig[x] for x in localpix ]).reshape(-1, 1)

        # create TOD from map
        scansim = OpSimScan(distmap=distsig)
        scansim.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_satellite_hwpfast_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.mapdir, "madam_hwpfast")
        if self.comm.rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

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
        pars[ 'path_output' ] = madam_out

        madam = OpMadam(params=pars)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

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

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.totsamp, tothits)
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")


    def test_hwpconst(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # make a pointing matrix with a HWP that is constant
        hwpstep = 2.0 * np.pi
        hwpsteptime = (self.totsamp / self.rate) / 60.0
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon, hwpstep=hwpstep, hwpsteptime=hwpsteptime)
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # construct a sky gradient operator, just to get the signal
        # map- we are not going to use the operator on the data.
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        sig = grad.sigmap()

        # pick a submap size and find the local submaps.
        submapsize = np.floor_divide(self.sim_nside, 16)
        allsm = np.floor_divide(localpix, submapsize)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct a distributed map which has the gradient        
        npix = 12 * self.sim_nside * self.sim_nside
        distsig = DistPixels(comm=self.toastcomm.comm_group, size=npix, nnz=1, dtype=np.float64, submap=submapsize, local=localsm)
        lsub, lpix = distsig.global_to_local(localpix)
        distsig.data[lsub,lpix,:] = np.array([ sig[x] for x in localpix ]).reshape(-1, 1)

        # create TOD from map
        scansim = OpSimScan(distmap=distsig)
        scansim.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_satellite_hwpconst_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.mapdir, "madam_hwpconst")
        if self.comm.rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

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
        pars[ 'path_output' ] = madam_out

        madam = OpMadam(params=pars)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

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

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.totsamp, tothits)
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")

