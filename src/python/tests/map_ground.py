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
from scipy.constants import degree

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_noise import *
from ..tod.sim_det_map import *
from ..tod.sim_noise import *
from ..map import *


class MapGroundTest(MPITestCase):


    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "map_ground")
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

        # this is an unpolarized detector...
        self.epsilon = {
            'bore' : 1.0,
        }

        nside = 1024
        self.sim_nside = nside
        #self.totsamp = 400000
        self.totsamp = 100000
        self.map_nside = nside
        self.rate = 40.0
        self.site_lon = '-67:47:10'
        self.site_lat = '-22:57:30'
        self.site_alt = 5200.
        self.patch_lon = 15 * degree # '10.0'
        self.patch_lat = -60 * degree # '-70.0'
        self.patch_coord = 'C'
        self.throw = 4.
        self.scanrate = 1.0
        self.scan_accel = 0.1
        self.CES_start = None

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

        nflagged = 0

        for i in range(nobs):
            # create the TOD for this observation

            tod = TODGround(
                self.toastcomm.comm_group,
                self.dets, 
                self.totsamp, 
                firsttime=0.0,
                rate=self.rate,
                site_lon=self.site_lon,
                site_lat=self.site_lat,
                site_alt=self.site_alt,
                patch_lon=self.patch_lon,
                patch_lat=self.patch_lat,
                patch_coord=self.patch_coord,
                throw=self.throw,
                scanrate=self.scanrate,
                scan_accel=self.scan_accel,
                CES_start=self.CES_start)

            self.common_flag_mask = tod.TURNAROUND

            common_flags = tod.read_common_flags()
            nflagged += np.sum((common_flags & self.common_flag_mask) != 0)

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

        self.nflagged = nflagged

    def test_azel(self):
        start = MPI.Wtime()

        quats1 = []
        quats2 = []

        for ob in self.data.obs:
            tod = ob['tod']
            for d in tod.local_dets:
                quats1.append(tod.read_pntg(detector=d))
                quats2.append(tod.read_pntg(detector=d, azel=True))

        for i in range(10):
            if np.all(quats1[0][i] == quats2[0][i]):
                raise Exception('Horizontal and celestial pointing must be different')

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("Az/El test took {:.3f} s".format(elapsed))

    def test_grad(self):
        start = MPI.Wtime()

        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside, nest=True, common_flag_mask=self.common_flag_mask)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon)
        pointing.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_ground_grad_info"), "w")
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars[ 'nside_submap' ] = min(8, self.map_nside)
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = madam_out
        pars[ 'info' ] = 0

        madam = OpMadam(params=pars, name='grad', purge=True, common_flag_mask=self.common_flag_mask)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt

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
                nt.assert_equal(self.totsamp-self.nflagged, tothits)

                sig = grad.sigmap()
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")

    def test_noise(self):
        start = MPI.Wtime()

        # generate noise timestreams from the noise model
        nsig = OpSimNoise()
        nsig.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon)
        pointing.exec(self.data)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_ground_noise_info"), "w")
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars[ 'nside_submap' ] = min(8, self.map_nside)
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = madam_out
        pars[ 'info' ] = 0

        madam = OpMadam(params=pars, detweights=detweights, name='noise', common_flag_mask=self.common_flag_mask)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt

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
                nt.assert_equal(self.totsamp-self.nflagged, tothits)
                print("tothits = ", tothits)

                mask = (bins > -1.0e20)
                print("num good pix = ", len(mask))
                rthits = np.sqrt(hits[mask].astype(np.float64))
                print("rthits = ", rthits)
                print("bmap = ", bins[mask])
                weighted = bins[mask] * rthits
                print("weighted = ", weighted)
                pixrms = np.sqrt(np.mean(weighted**2))
                todrms = self.NET * np.sqrt(self.rate)
                relerr = np.absolute(pixrms - todrms) / todrms
                print("pixrms = ", pixrms)
                print("todrms = ", todrms)
                print("relerr = ", relerr)
                self.assertTrue(relerr < 0.02)

        else:
            print("libmadam not available, skipping tests")


    def test_scanmap(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, epsilon=self.epsilon)
        pointing.exec(self.data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # construct a sky gradient operator, just to get the signal
        # map- we are not going to use the operator on the data.
        grad = OpSimGradient(nside=self.sim_nside, nest=True, common_flag_mask=self.common_flag_mask)
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
            handle = open(os.path.join(self.outdir,"out_test_ground_scanmap_info"), "w")
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars[ 'nside_submap' ] = min(8, self.map_nside)
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = madam_out
        pars[ 'info' ] = 0

        madam = OpMadam(params=pars, name='scan', common_flag_mask=self.common_flag_mask)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt

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
                nt.assert_equal(self.totsamp-self.nflagged, tothits)
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")


    def test_hwpfast(self):
        start = MPI.Wtime()

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
            handle = open(os.path.join(self.outdir,"out_test_ground_hwpfast_info"), "w")
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars[ 'nside_submap' ] = min(8, self.map_nside)
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = madam_out
        pars[ 'info' ] = 0

        madam = OpMadam(params=pars, name='scan', common_flag_mask=self.common_flag_mask)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt

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
                nt.assert_equal(self.totsamp-self.nflagged, tothits)
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")


    def test_hwpconst(self):
        start = MPI.Wtime()

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
            handle = open(os.path.join(self.outdir,"out_test_ground_hwpconst_info"), "w")
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars[ 'nside_submap' ] = min(8, self.map_nside)
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'F'
        pars[ 'write_wcov' ] = 'F'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = madam_out
        pars[ 'info' ] = 0

        madam = OpMadam(params=pars, name='scan', common_flag_mask=self.common_flag_mask)
        if madam.available:
            madam.exec(self.data)
            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt
                
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
                nt.assert_equal(self.totsamp-self.nflagged, tothits)
                mask = (bins > -1.0e20)
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)

        else:
            print("libmadam not available, skipping tests")
