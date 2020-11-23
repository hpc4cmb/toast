# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from .mpi import MPITestCase
from toast.tests.mpi import MPITestCase

import glob
import os
import shutil

import healpy as hp
import numpy as np

from ..timing import gather_timers, GlobalTimers
from ..timing import dump as dump_timing
from ..tod import AnalyticNoise, OpSimNoise, Interval, OpCacheCopy, OpCacheInit
from ..map import DistPixels
from ..todmap import (
    TODHpixSpiral,
    OpSimGradient,
    OpPointingHpix,
    OpFilterBin,
    OpSimScan,
)

from .. import qarray as qa

from ._helpers import create_outdir, create_distdata, boresight_focalplane


class OpFilterBinTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.rank = 0
        if self.comm is not None:
            self.rank = self.comm.rank
        if self.rank == 0:
            for fname in glob.glob("{}/*".format(self.outdir)):
                try:
                    os.remove(fname)
                except OSError:
                    pass
        if self.comm is not None:
            self.comm.barrier()

        self.nobs = 4
        self.data = create_distdata(self.comm, obs_per_group=self.nobs)

        self.ndet = 4
        self.sigma = 1
        self.rate = 50.0
        self.net = self.sigma / np.sqrt(self.rate)
        self.alpha = 2
        self.fknee = 1e0

        # Create detectors
        (
            dnames,
            dquat,
            depsilon,
            drate,
            dnet,
            dfmin,
            dfknee,
            dalpha,
        ) = boresight_focalplane(
            self.ndet,
            samplerate=self.rate,
            fknee=self.fknee,
            alpha=self.alpha,
            net=self.net,
        )

        # Pixelization
        self.sim_nside = 8
        self.map_nside = 8
        self.pointingmode = "IQU"
        self.nnz = 3

        # Samples per observation
        self.npix = 12 * self.sim_nside ** 2
        self.ninterval = 4
        self.totsamp = self.ninterval * self.npix

        # Define intervals
        intervals = []
        interval_length = self.npix
        istart = 0
        while istart < self.totsamp:
            istop = istart + interval_length
            intervals.append(
                Interval(
                    start=istart / self.rate,
                    stop=istop / self.rate,
                    first=istart,
                    last=istop - 1,
                )
            )
            istart = istop

        # Construct an uncorrelated analytic noise model for the detectors

        noise = AnalyticNoise(
            rate=drate,
            fmin=dfmin,
            detectors=dnames,
            fknee=dfknee,
            alpha=dalpha,
            NET=dnet,
        )

        # Populate the observations

        for iobs in range(self.nobs):
            if iobs % 3 == 1:
                rot = qa.from_angles(np.pi / 2, 0, 0)
            if iobs % 3 == 2:
                rot = qa.from_angles(np.pi / 2, np.pi / 2, 0)
            else:
                rot = None

            tod = TODHpixSpiral(
                self.data.comm.comm_group,
                dquat,
                self.totsamp,
                detranks=self.data.comm.group_size,
                rate=self.rate,
                nside=self.sim_nside,
                rot=rot,
            )

            self.data.obs[iobs]["tod"] = tod
            self.data.obs[iobs]["noise"] = noise
            self.data.obs[iobs]["intervals"] = intervals

        self.npix = 12 * self.map_nside ** 2
        pix = np.arange(self.npix)
        pix = hp.reorder(pix, r2n=True)

        # Synthesize an input map
        self.lmax = 2 * self.sim_nside
        self.cl = np.ones([4, self.lmax + 1])
        self.cl[:, 0:2] = 0
        fwhm = np.radians(10)
        self.inmap = hp.synfast(
            self.cl,
            self.sim_nside,
            lmax=self.lmax,
            mmax=self.lmax,
            pol=True,
            pixwin=True,
            fwhm=np.radians(30),
            verbose=False,
        )
        self.inmap = hp.reorder(self.inmap, r2n=True)
        self.inmapfile = os.path.join(self.outdir, "input_map.fits")
        hp.write_map(self.inmapfile, self.inmap, overwrite=True, nest=True)

        return

    def test_filterbin(self):

        name = "testtod2"
        init = OpCacheInit(name=name, init_val=0)

        # make a simple pointing matrix
        pointing = OpPointingHpix(
            nside=self.map_nside, nest=True, mode=self.pointingmode
        )
        pointing.exec(self.data)

        # Scan the signal from a map
        distmap = DistPixels(self.data, nnz=self.nnz, dtype=np.float32)
        distmap.read_healpix_fits(self.inmapfile)

        scansim = OpSimScan(input_map=distmap, out=name)
        scansim.exec(self.data)

        # Add simulated noise
        opnoise = OpSimNoise(realization=0, out=name)
        opnoise.exec(self.data)

        # Copy the signal for comparison
        name_madam = name + "_copy"
        cachecopy = OpCacheCopy(name, name_madam)
        cachecopy.exec(self.data)

        # Run FilterBin

        gt = GlobalTimers.get()
        gt.start("OpMapMaker test")

        filterbin = OpFilterBin(
            nside=self.map_nside,
            nnz=self.nnz,
            name=name,
            outdir=self.outdir,
            outprefix="toast_test_",
            write_obs_matrix=True,
            ground_filter_order=None,
            poly_filter_order=3,
        )
        filterbin.exec(self.data)

        return
