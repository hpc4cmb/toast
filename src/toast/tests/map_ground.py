# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os
import shutil

import numpy as np
import numpy.testing as nt

import healpy as hp

from ..tod import (
    TODGround,
    OpPointingHpix,
    AnalyticNoise,
    OpSimGradient,
    OpSimNoise,
    OpSimScan,
)

from ..map import OpLocalPixels, OpMadam, DistPixels

from ._helpers import create_outdir, create_distdata, boresight_focalplane


class MapGroundTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # one detector per process and a single chunk.

        self.data = create_distdata(self.comm, obs_per_group=1)
        self.data_fast_hwp = create_distdata(self.comm, obs_per_group=1)
        self.data_const_hwp = create_distdata(self.comm, obs_per_group=1)

        self.ndet = self.data.comm.group_size
        self.rate = 20.0

        # Create detectors with white noise
        self.NET = 5.0

        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = boresight_focalplane(
            self.ndet, samplerate=self.rate, fknee=0.0, net=self.NET
        )

        # Samples per observation
        self.totsamp = 100000

        # Pixelization
        nside = 256
        self.sim_nside = nside
        self.map_nside = nside

        # Scan properties
        self.site_lon = "-67:47:10"
        self.site_lat = "-22:57:30"
        self.site_alt = 5200.0
        self.coord = "C"
        self.azmin = 45
        self.azmax = 55
        self.el = 60
        self.scanrate = 1.0
        self.scan_accel = 0.1
        self.CES_start = None

        # Populate the single observation per group

        common_args = [dquat, self.totsamp]
        common_kwargs = {
            "detranks": self.data.comm.group_size,
            "firsttime": 0.0,
            "rate": self.rate,
            "site_lon": self.site_lon,
            "site_lat": self.site_lat,
            "site_alt": self.site_alt,
            "azmin": self.azmin,
            "azmax": self.azmax,
            "el": self.el,
            "coord": self.coord,
            "scanrate": self.scanrate,
            "scan_accel": self.scan_accel,
            "CES_start": self.CES_start,
        }

        # No HWP
        tod_no_hwp = TODGround(self.data.comm.comm_group, *common_args, **common_kwargs)

        # CHWP
        hwprate = self.rate * 60.0
        tod_fast_hwp = TODGround(
            self.data_fast_hwp.comm.comm_group,
            *common_args,
            **common_kwargs,
            hwprpm=hwprate
        )

        # Stepped HWP
        hwpstep = 2.0 * np.pi
        hwpsteptime = (self.totsamp / self.rate) / 60.0
        tod_const_hwp = TODGround(
            self.data_const_hwp.comm.comm_group,
            *common_args,
            **common_kwargs,
            hwpstep=hwpstep,
            hwpsteptime=hwpsteptime
        )

        for data, tod in [
            (self.data, tod_no_hwp),
            (self.data_fast_hwp, tod_fast_hwp),
            (self.data_const_hwp, tod_const_hwp),
        ]:

            self.common_flag_mask = tod.TURNAROUND

            common_flags = tod.read_common_flags()

            # Number of flagged samples in each observation.  Only the first row
            # of the process grid needs to contribute, since all process columns
            # have identical common flags.
            nflagged = 0
            if (tod.grid_comm_col is None) or (tod.grid_comm_col.rank == 0):
                nflagged += np.sum((common_flags & self.common_flag_mask) != 0)

            # Number of flagged samples across all observations
            self.nflagged = None
            if self.comm is None:
                self.nflagged = nflagged
            else:
                self.nflagged = self.data.comm.comm_world.allreduce(nflagged)

            # add analytic noise model with white noise

            nse = AnalyticNoise(
                rate=drate,
                fmin=dfmin,
                detectors=dnames,
                fknee=dfknee,
                alpha=dalpha,
                NET=dnet,
            )

            data.obs[0]["tod"] = tod
            data.obs[0]["noise"] = nse
        return

    def test_azel(self):
        quats1 = []
        quats2 = []

        for ob in self.data.obs:
            tod = ob["tod"]
            for d in tod.local_dets:
                quats1.append(tod.read_pntg(detector=d))
                quats2.append(tod.read_pntg(detector=d, azel=True))

        for i in range(10):
            if np.all(quats1[0][i] == quats2[0][i]):
                raise Exception("Horizontal and celestial pointing must be different")
        return

    def test_grad(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # add simple sky gradient signal
        grad = OpSimGradient(
            nside=self.sim_nside, nest=True, common_flag_mask=self.common_flag_mask
        )
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_ground_grad_info"), "w")
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars["nside_submap"] = min(8, self.map_nside)
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(
            params=pars,
            name="grad",
            purge=False,
            common_flag_mask=self.common_flag_mask,
        )
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
                nt.assert_equal(
                    self.ndet
                    * ((self.data.comm.ngroups * self.totsamp) - self.nflagged),
                    tothits,
                )

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
            handle = open(os.path.join(self.outdir, "out_test_ground_noise_info"), "w")
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars["nside_submap"] = min(8, self.map_nside)
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(
            params=pars,
            detweights=detweights,
            name="noise",
            common_flag_mask=self.common_flag_mask,
        )
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
                nt.assert_equal(
                    self.ndet
                    * ((self.data.comm.ngroups * self.totsamp) - self.nflagged),
                    tothits,
                )

                mask = bins > -1.0e20
                # print("num good pix = ", len(mask))
                rthits = np.sqrt(hits[mask].astype(np.float64))
                # print("rthits = ", rthits)
                # print("bmap = ", bins[mask])
                weighted = bins[mask] * rthits
                # print("weighted = ", weighted)
                pixrms = np.sqrt(np.mean(weighted ** 2))
                todrms = self.NET * np.sqrt(self.rate)
                relerr = np.absolute(pixrms - todrms) / todrms
                # print("pixrms = ", pixrms)
                # print("todrms = ", todrms)
                # print("relerr = ", relerr)
                self.assertTrue(relerr < 0.03)
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
        grad = OpSimGradient(
            nside=self.sim_nside, nest=True, common_flag_mask=self.common_flag_mask
        )
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
                os.path.join(self.outdir, "out_test_ground_scanmap_info"), "w"
            )
        self.data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars["nside_submap"] = min(8, self.map_nside)
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(
            params=pars, name="scan", common_flag_mask=self.common_flag_mask
        )
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
                nt.assert_equal(
                    self.ndet
                    * ((self.data.comm.ngroups * self.totsamp) - self.nflagged),
                    tothits,
                )
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
        data = self.data_fast_hwp
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(data)

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
            comm=data.comm.comm_group,
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
        scansim.exec(data)

        handle = None
        if rank == 0:
            handle = open(
                os.path.join(self.outdir, "out_test_ground_hwpfast_info"), "w"
            )
        data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars["nside_submap"] = min(8, self.map_nside)
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(
            params=pars, name="scan", common_flag_mask=self.common_flag_mask
        )
        if madam.available:
            madam.exec(data)

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
                nt.assert_equal(
                    self.ndet * ((data.comm.ngroups * self.totsamp) - self.nflagged),
                    tothits,
                )
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
        data = self.data_const_hwp
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(data)

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(data)

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
            comm=data.comm.comm_group,
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
        scansim.exec(data)

        handle = None
        if rank == 0:
            handle = open(
                os.path.join(self.outdir, "out_test_ground_hwpconst_info"), "w"
            )
        data.info(handle, common_flag_mask=self.common_flag_mask)
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
        pars["nside_submap"] = min(8, self.map_nside)
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = madam_out
        pars["info"] = 0

        madam = OpMadam(
            params=pars, name="scan", common_flag_mask=self.common_flag_mask
        )
        if madam.available:
            madam.exec(data)

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
                nt.assert_equal(
                    self.ndet * ((data.comm.ngroups * self.totsamp) - self.nflagged),
                    tothits,
                )
                mask = bins > -1.0e20
                nt.assert_almost_equal(bins[mask], sig[mask], decimal=4)
        else:
            print("libmadam not available, skipping tests")
        return
