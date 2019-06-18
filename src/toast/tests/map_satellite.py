# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os
import shutil

import numpy as np
import numpy.testing as nt

import healpy as hp

from .. import qarray as qa

from ..tod import (
    TODSatellite,
    OpPointingHpix,
    AnalyticNoise,
    OpSimGradient,
    OpSimNoise,
    slew_precession_axis,
    satellite_scanning,
    OpSimScan,
)

from ..map import OpLocalPixels, OpMadam, DistPixels

from ._helpers import (
    create_outdir,
    create_distdata,
    boresight_focalplane,
    uniform_chunks,
)


class MapSatelliteTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # one detector with multiple chunks distributed among the processes.

        self.data = create_distdata(self.comm, obs_per_group=1)

        self.ndet = 1
        self.rate = 40.0

        # Create detectors with white noise
        self.NET = 7.0

        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = boresight_focalplane(
            self.ndet, samplerate=self.rate, net=self.NET, fknee=0.0
        )

        # Samples per observation
        self.totsamp = 4000000

        # Pixelization
        self.sim_nside = 64
        self.map_nside = 64

        # Scan strategy
        self.spinperiod = 10.0
        self.spinangle = 30.0
        self.precperiod = 50.0
        self.precangle = 65.0

        # in order to make sure that the noise realization is reproducible
        # at all concurrencies, we set the chunksize to something independent
        # of the number of ranks.

        # Chunks
        chunks = uniform_chunks(self.totsamp, nchunk=100)

        # Populate the single observation per group

        tod = TODSatellite(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=1,
            firsttime=0.0,
            rate=self.rate,
            spinperiod=self.spinperiod,
            spinangle=self.spinangle,
            precperiod=self.precperiod,
            precangle=self.precangle,
            sampsizes=chunks,
        )

        precquat = np.empty(4 * tod.local_samples[1], dtype=np.float64).reshape((-1, 4))

        slew_precession_axis(
            precquat, firstsamp=tod.local_samples[0], samplerate=self.rate, degday=1.0
        )

        tod.set_prec_axis(qprec=precquat)

        # add analytic noise model with white noise

        nse = AnalyticNoise(
            rate=drate,
            fmin=dfmin,
            detectors=dnames,
            fknee=dfknee,
            alpha=dalpha,
            NET=dnet,
        )

        self.data.obs[0]["tod"] = tod
        self.data.obs[0]["noise"] = nse

    def test_boresight_null(self):
        # verify that if all angles are zero, we get fixed pointing
        # along the x-axis.

        nsim = 1000

        zaxis = np.array([0, 0, 1], dtype=np.float64)

        borequat = np.empty(4 * nsim, dtype=np.float64).reshape((-1, 4))

        satellite_scanning(
            borequat,
            qprec=None,
            samplerate=100.0,
            spinperiod=1.0,
            spinangle=0.0,
            precperiod=20.0,
            precangle=0.0,
        )

        data = qa.rotate(borequat, np.tile(zaxis, nsim).reshape(-1, 3))

        np.testing.assert_almost_equal(
            data, np.tile(np.array([1.0, 0.0, 0.0]), nsim).reshape(-1, 3)
        )
        return

    def test_grad(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(
                os.path.join(self.outdir, "out_test_satellite_grad_info"), "w"
            )
        self.data.info(handle)
        if rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.outdir, "madam_grad")
        if rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

        pars = {}
        pars["kfirst"] = "F"
        pars["base_first"] = 1.0
        pars["fsample"] = self.rate
        pars["nside_map"] = self.map_nside
        pars["nside_cross"] = self.map_nside
        pars["nside_submap"] = 8
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(params=pars, name="grad", purge=True)
        if madam.available:
            madam.exec(self.data)

            if rank == 0:
                import matplotlib.pyplot as plt

                hitsfile = os.path.join(madam_out, "madam_hmap.fits")
                hits = hp.read_map(hitsfile, nest=True)

                outfile = "{}.png".format(hitsfile)
                hp.mollview(hits, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                binfile = os.path.join(madam_out, "madam_bmap.fits")
                bins = hp.read_map(binfile, nest=True)

                outfile = "{}.png".format(binfile)
                hp.mollview(bins, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.data.comm.ngroups * self.totsamp, tothits)

                sig = grad.sigmap()
                mask = bins > -1.0e20
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)
        else:
            print("libmadam not available, skipping tests")
        return

    def test_noise(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # generate noise timestreams from the noise model
        nsig = OpSimNoise()
        nsig.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(
                os.path.join(self.outdir, "out_test_satellite_noise_info"), "w"
            )
        self.data.info(handle)
        if rank == 0:
            handle.close()

        # For noise weighting in madam, we know we are using an analytic noise
        # and so we can use noise weights based on the NET.  This is instrument
        # specific.

        tod = self.data.obs[0]["tod"]
        nse = self.data.obs[0]["noise"]
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d) ** 2)

        # make a binned map with madam
        madam_out = os.path.join(self.outdir, "madam_noise")
        if rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

        pars = {}
        pars["kfirst"] = "F"
        pars["base_first"] = 1.0
        pars["fsample"] = self.rate
        pars["nside_map"] = self.map_nside
        pars["nside_cross"] = self.map_nside
        pars["nside_submap"] = 8
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(params=pars, detweights=detweights, name="noise")
        if madam.available:
            madam.exec(self.data)

            if rank == 0:
                import matplotlib.pyplot as plt

                hitsfile = os.path.join(madam_out, "madam_hmap.fits")
                hits = hp.read_map(hitsfile, nest=True)

                outfile = "{}.png".format(hitsfile)
                hp.mollview(hits, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                binfile = os.path.join(madam_out, "madam_bmap.fits")
                bins = hp.read_map(binfile, nest=True)

                outfile = "{}.png".format(binfile)
                hp.mollview(bins, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # check that pixel rms makes sense given the
                # number of hits and the timestream rms

                tothits = np.sum(hits)
                nt.assert_equal(self.data.comm.ngroups * self.totsamp, tothits)
                # print("tothits = ", tothits)

                mask = bins > -1.0e20
                # print("num good pix = ", len(mask))
                rthits = np.sqrt(hits[mask].astype(np.float64))
                # print("rthits = ", rthits)
                # print("bmap = ", bins[mask])
                weighted = bins[mask] * rthits
                pixrms = np.std(weighted)
                todrms = self.NET * np.sqrt(self.rate)
                relerr = np.absolute(pixrms - todrms) / todrms
                # print("pixrms = ", pixrms)
                # print("todrms = ", todrms)
                # print("relerr = ", relerr)
                self.assertTrue(relerr < 0.05)
        else:
            print("libmadam not available, skipping tests")
        return

    def test_scanmap(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
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
        localsm = np.unique(np.floor_divide(localpix, submapsize))

        # construct a distributed map which has the gradient
        npix = 12 * self.sim_nside * self.sim_nside
        distsig = DistPixels(
            comm=self.data.comm.comm_group,
            size=npix,
            nnz=1,
            dtype=np.float64,
            submap=submapsize,
            local=localsm,
        )

        lsub, lpix = distsig.global_to_local(localpix)

        distsig.data[lsub, lpix, :] = np.array([sig[x] for x in localpix]).reshape(
            -1, 1
        )

        # create TOD from map
        scansim = OpSimScan(distmap=distsig)
        scansim.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(
                os.path.join(self.outdir, "out_test_satellite_scanmap_info"), "w"
            )
        self.data.info(handle)
        if rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.outdir, "madam_scansim")
        if rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

        pars = {}
        pars["kfirst"] = "F"
        pars["base_first"] = 1.0
        pars["fsample"] = self.rate
        pars["nside_map"] = self.map_nside
        pars["nside_cross"] = self.map_nside
        pars["nside_submap"] = 8
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(params=pars, name="scan")
        if madam.available:
            madam.exec(self.data)

            if rank == 0:
                import matplotlib.pyplot as plt

                hitsfile = os.path.join(madam_out, "madam_hmap.fits")
                hits = hp.read_map(hitsfile, nest=True)

                outfile = "{}.png".format(hitsfile)
                hp.mollview(hits, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                binfile = os.path.join(madam_out, "madam_bmap.fits")
                bins = hp.read_map(binfile, nest=True)

                outfile = "{}.png".format(binfile)
                hp.mollview(bins, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.data.comm.ngroups * self.totsamp, tothits)
                mask = bins > -1.0e20
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)
        else:
            print("libmadam not available, skipping tests")
        return

    def test_hwpfast(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # make a pointing matrix with a HWP that rotates 2*PI every sample
        hwprate = self.rate * 60.0
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, hwprpm=hwprate)
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
        localsm = np.unique(np.floor_divide(localpix, submapsize))

        # construct a distributed map which has the gradient
        npix = 12 * self.sim_nside * self.sim_nside

        distsig = DistPixels(
            comm=self.data.comm.comm_group,
            size=npix,
            nnz=1,
            dtype=np.float64,
            submap=submapsize,
            local=localsm,
        )

        lsub, lpix = distsig.global_to_local(localpix)

        distsig.data[lsub, lpix, :] = np.array([sig[x] for x in localpix]).reshape(
            -1, 1
        )

        # create TOD from map
        scansim = OpSimScan(distmap=distsig)
        scansim.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(
                os.path.join(self.outdir, "out_test_satellite_hwpfast_info"), "w"
            )
        self.data.info(handle)
        if rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.outdir, "madam_hwpfast")
        if rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

        pars = {}
        pars["kfirst"] = "F"
        pars["base_first"] = 1.0
        pars["fsample"] = self.rate
        pars["nside_map"] = self.map_nside
        pars["nside_cross"] = self.map_nside
        pars["nside_submap"] = 8
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(params=pars, name="scan")
        if madam.available:
            madam.exec(self.data)

            if rank == 0:
                import matplotlib.pyplot as plt

                hitsfile = os.path.join(madam_out, "madam_hmap.fits")
                hits = hp.read_map(hitsfile, nest=True)

                outfile = "{}.png".format(hitsfile)
                hp.mollview(hits, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                binfile = os.path.join(madam_out, "madam_bmap.fits")
                bins = hp.read_map(binfile, nest=True)

                outfile = "{}.png".format(binfile)
                hp.mollview(bins, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.data.comm.ngroups * self.totsamp, tothits)
                mask = bins > -1.0e20
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)
        else:
            print("libmadam not available, skipping tests")
        return

    def test_hwpconst(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # make a pointing matrix with a HWP that is constant
        hwpstep = 2.0 * np.pi
        hwpsteptime = (self.totsamp / self.rate) / 60.0
        pointing = OpPointingHpix(
            nside=self.map_nside, nest=True, hwpstep=hwpstep, hwpsteptime=hwpsteptime
        )
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
        localsm = np.unique(np.floor_divide(localpix, submapsize))

        # construct a distributed map which has the gradient
        npix = 12 * self.sim_nside * self.sim_nside

        distsig = DistPixels(
            comm=self.data.comm.comm_group,
            size=npix,
            nnz=1,
            dtype=np.float64,
            submap=submapsize,
            local=localsm,
        )

        lsub, lpix = distsig.global_to_local(localpix)

        distsig.data[lsub, lpix, :] = np.array([sig[x] for x in localpix]).reshape(
            -1, 1
        )

        # create TOD from map
        scansim = OpSimScan(distmap=distsig)
        scansim.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(
                os.path.join(self.outdir, "out_test_satellite_hwpconst_info"), "w"
            )
        self.data.info(handle)
        if rank == 0:
            handle.close()

        # make a binned map with madam
        madam_out = os.path.join(self.outdir, "madam_hwpconst")
        if rank == 0:
            if os.path.isdir(madam_out):
                shutil.rmtree(madam_out)
            os.mkdir(madam_out)

        pars = {}
        pars["kfirst"] = "F"
        pars["base_first"] = 1.0
        pars["fsample"] = self.rate
        pars["nside_map"] = self.map_nside
        pars["nside_cross"] = self.map_nside
        pars["nside_submap"] = 8
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(params=pars, name="scan")
        if madam.available:
            madam.exec(self.data)

            if rank == 0:
                import matplotlib.pyplot as plt

                hitsfile = os.path.join(madam_out, "madam_hmap.fits")
                hits = hp.read_map(hitsfile, nest=True)

                outfile = "{}.png".format(hitsfile)
                hp.mollview(hits, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                binfile = os.path.join(madam_out, "madam_bmap.fits")
                bins = hp.read_map(binfile, nest=True)

                outfile = "{}.png".format(binfile)
                hp.mollview(bins, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                # compare binned map to input signal

                tothits = np.sum(hits)
                nt.assert_equal(self.data.comm.ngroups * self.totsamp, tothits)
                mask = bins > -1.0e20
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)
        else:
            print("libmadam not available, skipping tests")
        return
