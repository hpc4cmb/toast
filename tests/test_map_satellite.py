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
from toast.map.madam import *

from toast.mpirunner import MPITestCase


class MapSatelliteTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "map_satellite")
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


    def test_grad(self):
        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        cache.exec(self.data)

        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpixSimple(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        with open(os.path.join(self.outdir,"out_test_satellite_grad_info"), "w") as f:
            self.data.info(f)

        # make a binned map with madam
        madam_out = os.path.join(self.mapdir, "madam_grad")
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
                self.assertTrue(self.totsamp, tothits)

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
        pointing = OpPointingHpixSimple(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        with open(os.path.join(self.outdir,"out_test_satellite_noise_info"), "w") as f:
            self.data.info(f)

        # make a binned map with madam
        madam_out = os.path.join(self.mapdir, "madam_noise")
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

                # check that pixel rms makes sense given the
                # number of hits and the timestream rms

                tothits = np.sum(hits)
                self.assertTrue(self.totsamp, tothits)

                mask = (bins > -1.0e20)
                weighted = bins[mask] * np.sqrt(hits[mask])
                pixrms = np.std(weighted)
                todrms = self.NET * np.sqrt(self.rate)
                relerr = np.absolute(pixrms - todrms) / todrms
                self.assertTrue(relerr < 0.01)

        else:
            print("libmadam not available, skipping tests")
        
