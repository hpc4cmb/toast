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
from ..tod.interval import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_detdata import *
from ..tod.sim_noise import *
from ..tod.sim_interval import *
from ..map import *


class BinnedTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
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

        self.totsamp = 2000000

        self.map_nside = 32
        self.map_npix = 12 * self.map_nside**2
        
        self.rate = 200.0
        self.spinperiod = 1.0
        self.spinangle = 30.0
        self.precperiod = 5.0
        self.precangle = 65.0

        self.hwprpm = 100

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

        # define some valid data intervals so that we can test flag handling
        # in the gaps

        nint = 4
        intsamp = self.totsamp // nint
        inttime = (intsamp - 1) / self.rate
        durtime = inttime * 0.85
        gaptime = inttime - durtime
        intrvls = regular_intervals(nint, 0, 0, self.rate, durtime, gaptime)
        self.validsamp = 0
        for it in intrvls:
            print(it.first," ",it.last," ",it.start," ",it.stop)
            self.validsamp += it.last - it.first + 1
        print(self.validsamp, " good samples")

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
            ob['intervals'] = intrvls
            ob['baselines'] = None
            ob['noise'] = nse

            self.data.obs.append(ob)


    def test_binned(self):
        start = MPI.Wtime()

        # flag data outside valid intervals
        gapflagger = OpFlagGaps(common_flag_name='comflag')
        gapflagger.exec(self.data)

        # generate noise timestreams from the noise model
        nsig = OpSimNoise()
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
        invnpp.data.fill(0.0)

        zmap = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=3, dtype=np.float64, submap=self.subnpix, local=localsm)
        zmap.data.fill(0.0)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=self.sim_npix, nnz=1, dtype=np.int64, submap=self.subnpix, local=localsm)
        hits.data.fill(0)

        # accumulate the inverse covariance and noise weighted map.  
        # Use detector weights based on the analytic NET.

        tod = self.data.obs[0]['tod']
        nse = self.data.obs[0]['noise']
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d)**2)

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits, zmap=zmap, name="noise", common_flag_name="comflag")
        build_invnpp.exec(self.data)

        invnpp.allreduce()
        hits.allreduce()
        zmap.allreduce()

        hits.write_healpix_fits(os.path.join(self.mapdir, "hits.fits"))
        invnpp.write_healpix_fits(os.path.join(self.mapdir, "invnpp.fits"))
        zmap.write_healpix_fits(os.path.join(self.mapdir, "zmap.fits"))

        # invert it
        covariance_invert(invnpp, 1.0e-3)

        invnpp.write_healpix_fits(os.path.join(self.mapdir, "npp.fits"))

        # compute the binned map, N_pp x Z

        covariance_apply(invnpp, zmap)
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
        pars[ 'pixlim_cross' ] = 1.0e-2
        pars[ 'pixlim_map' ] = 1.0e-3
        pars[ 'write_map' ] = 'F'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'T'
        pars[ 'write_wcov' ] = 'T'
        pars[ 'write_hits' ] = 'T'
        pars[ 'kfilter' ] = 'F'
        pars[ 'run_submap_test' ] = 'F'
        pars[ 'path_output' ] = madam_out

        madam = OpMadam(params=pars, detweights=detweights, name="noise",
            common_flag_name="comflag")

        if madam.available:
            madam.exec(self.data)

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt
                
                hitsfile = os.path.join(madam_out, 'madam_hmap.fits')
                hits = hp.read_map(hitsfile, nest=True)

                outfile = "{}.png".format(hitsfile)
                hp.mollview(hits, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                toastfile = os.path.join(self.mapdir, 'hits.fits')
                toasthits = hp.read_map(toastfile, nest=True)

                nt.assert_equal(hits, toasthits)

                tothits = np.sum(hits)
                nt.assert_equal(self.validsamp, tothits)

                covfile = os.path.join(madam_out, 'madam_wcov_inv.fits')
                cov = hp.read_map(covfile, nest=True, field=None)

                toastfile = os.path.join(self.mapdir, 'invnpp.fits')
                toastcov = hp.read_map(toastfile, nest=True, field=None)

                # for p in range(6):
                #     print("elem {} madam min/max = ".format(p), np.min(cov[p]), " / ", np.max(cov[p]))
                #     print("elem {} toast min/max = ".format(p), np.min(toastcov[p]), " / ", np.max(toastcov[p]))
                #     print("elem {} invNpp max diff = ".format(p), np.max(np.absolute(toastcov[p] - cov[p])))
                #     nt.assert_almost_equal(cov[p], toastcov[p])

                covfile = os.path.join(madam_out, 'madam_wcov.fits')
                cov = hp.read_map(covfile, nest=True, field=None)

                toastfile = os.path.join(self.mapdir, 'npp.fits')
                toastcov = hp.read_map(toastfile, nest=True, field=None)

                # for p in range(6):
                #     covdiff = toastcov[p] - cov[p]
                #     print("elem {} madam min/max = ".format(p), np.min(cov[p]), " / ", np.max(cov[p]))
                #     print("elem {} toast min/max = ".format(p), np.min(toastcov[p]), " / ", np.max(toastcov[p]))
                #     print("elem {} Npp max diff = ".format(p), np.max(np.absolute(covdiff[p])))
                #     print("elem {} Npp mean / rms diff = ".format(p), np.mean(covdiff[p]), " / ", np.std(covdiff[p]))
                #     print("elem {} Npp relative diff mean / rms = ".format(p), np.mean(np.absolute(covdiff[p]/cov[p])), " / ", np.std(np.absolute(covdiff[p]/cov[p])))
                #     nt.assert_almost_equal(cov[p], toastcov[p])

                binfile = os.path.join(madam_out, 'madam_bmap.fits')
                bins = hp.read_map(binfile, nest=True, field=None)
                mask = hp.mask_bad(bins[0])
                bins[0][mask] = 0.0
                mask = hp.mask_bad(bins[1])
                bins[1][mask] = 0.0
                mask = hp.mask_bad(bins[2])
                bins[2][mask] = 0.0

                outfile = "{}_I.png".format(binfile)
                hp.mollview(bins[0], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                outfile = "{}_Q.png".format(binfile)
                hp.mollview(bins[1], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                outfile = "{}_U.png".format(binfile)
                hp.mollview(bins[2], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                toastfile = os.path.join(self.mapdir, 'binned.fits')
                toastbins = hp.read_map(toastfile, nest=True, field=None)

                outfile = "{}_I.png".format(toastfile)
                hp.mollview(toastbins[0], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                outfile = "{}_Q.png".format(toastfile)
                hp.mollview(toastbins[1], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                outfile = "{}_U.png".format(toastfile)
                hp.mollview(toastbins[2], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # compare binned map to madam output

                diffmap = toastbins[0] - bins[0]
                mask = (bins[0] != 0)
                # print("toast/madam I diff mean / std = ", np.mean(diffmap[mask]), np.std(diffmap[mask]))
                # print("toast/madam I diff rel ratio min / max = ", np.min(diffmap[mask]/bins[0][mask]), " / ", np.max(diffmap[mask]/bins[0][mask]))
                outfile = "{}_diff_madam_I.png".format(toastfile)
                hp.mollview(diffmap, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                #nt.assert_almost_equal(bins[0][mask], binserial[0][mask], decimal=6)

                diffmap = toastbins[1] - bins[1]
                mask = (bins[1] != 0)
                # print("toast/madam Q diff mean / std = ", np.mean(diffmap[mask]), np.std(diffmap[mask]))
                # print("toast/madam Q diff rel ratio min / max = ", np.min(diffmap[mask]/bins[1][mask]), " / ", np.max(diffmap[mask]/bins[1][mask]))
                outfile = "{}_diff_madam_Q.png".format(toastfile)
                hp.mollview(diffmap, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                #nt.assert_almost_equal(bins[1][mask], binserial[1][mask], decimal=6)

                diffmap = toastbins[2] - bins[2]
                mask = (bins[2] != 0)
                # print("toast/madam U diff mean / std = ", np.mean(diffmap[mask]), np.std(diffmap[mask]))
                # print("toast/madam U diff rel ratio min / max = ", np.min(diffmap[mask]/bins[2][mask]), " / ", np.max(diffmap[mask]/bins[2][mask]))
                outfile = "{}_diff_madam_U.png".format(toastfile)
                hp.mollview(diffmap, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                #nt.assert_almost_equal(bins[2][mask], binserial[2][mask], decimal=6)


                # compute the binned map serially as a check

                zfile = os.path.join(self.mapdir, 'zmap.fits')
                ztoast = hp.read_map(zfile, nest=True, field=None)

                binserial = np.copy(ztoast)
                for p in range(self.map_npix):
                    binserial[0][p] = toastcov[0][p] * ztoast[0][p] + toastcov[1][p] * ztoast[1][p] + toastcov[2][p] * ztoast[2][p]
                    binserial[1][p] = toastcov[1][p] * ztoast[0][p] + toastcov[3][p] * ztoast[1][p] + toastcov[4][p] * ztoast[2][p]
                    binserial[2][p] = toastcov[2][p] * ztoast[0][p] + toastcov[4][p] * ztoast[1][p] + toastcov[5][p] * ztoast[2][p]

                toastfile = os.path.join(self.mapdir, 'binned_serial')
                outfile = "{}_I.png".format(toastfile)
                hp.mollview(binserial[0], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                outfile = "{}_Q.png".format(toastfile)
                hp.mollview(binserial[1], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                outfile = "{}_U.png".format(toastfile)
                hp.mollview(binserial[2], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # compare binned map to madam output

                diffmap = binserial[0] - bins[0]
                mask = (bins[0] != 0)
                # print("serial/madam I diff mean / std = ", np.mean(diffmap[mask]), np.std(diffmap[mask]))
                # print("serial/madam I diff rel ratio min / max = ", np.min(diffmap[mask]/bins[0][mask]), " / ", np.max(diffmap[mask]/bins[0][mask]))
                outfile = "{}_diff_madam_I.png".format(toastfile)
                hp.mollview(diffmap, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                nt.assert_almost_equal(bins[0][mask], binserial[0][mask], decimal=3)

                diffmap = binserial[1] - bins[1]
                mask = (bins[1] != 0)
                # print("serial/madam Q diff mean / std = ", np.mean(diffmap[mask]), np.std(diffmap[mask]))
                # print("serial/madam Q diff rel ratio min / max = ", np.min(diffmap[mask]/bins[1][mask]), " / ", np.max(diffmap[mask]/bins[1][mask]))
                outfile = "{}_diff_madam_Q.png".format(toastfile)
                hp.mollview(diffmap, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                nt.assert_almost_equal(bins[1][mask], binserial[1][mask], decimal=3)

                diffmap = binserial[2] - bins[2]
                mask = (bins[2] != 0)
                # print("serial/madam U diff mean / std = ", np.mean(diffmap[mask]), np.std(diffmap[mask]))
                # print("serial/madam U diff rel ratio min / max = ", np.min(diffmap[mask]/bins[2][mask]), " / ", np.max(diffmap[mask]/bins[2][mask]))
                outfile = "{}_diff_madam_U.png".format(toastfile)
                hp.mollview(diffmap, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                nt.assert_almost_equal(bins[2][mask], binserial[2][mask], decimal=3)


            stop = MPI.Wtime()
            elapsed = stop - start
            self.print_in_turns("Madam test took {:.3f} s".format(elapsed))
        else:
            print("libmadam not available, skipping tests")

        #self.assertTrue(False)

        return

