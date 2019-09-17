# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# from .mpi import MPITestCase
from toast.tests.mpi import MPITestCase

import glob
import os
import shutil

import healpy as hp
import numpy as np

from ..tod import AnalyticNoise, OpSimNoise, Interval, OpCacheCopy
from ..todmap import TODHpixSpiral, OpSimGradient, OpPointingHpix, OpMadam, OpMapMaker

from ._helpers import create_outdir, create_distdata, boresight_focalplane


class OpMapMakerTest(MPITestCase):
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

        # One observation per group
        self.data = create_distdata(self.comm, obs_per_group=1)

        self.ndet = 4  # self.data.comm.group_size
        self.rate = 50.0

        # Create detectors with defaults
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = boresight_focalplane(
            self.ndet, samplerate=self.rate
        )

        # Pixelization
        self.sim_nside = 64
        self.map_nside = 64
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

        # Populate the observations

        tod = TODHpixSpiral(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=self.data.comm.group_size,
            rate=self.rate,
            nside=self.sim_nside,
        )

        # Construct an uncorrelated analytic noise model for the detectors

        noise = AnalyticNoise(
            rate=drate,
            fmin=dfmin,
            detectors=dnames,
            fknee=dfknee,
            alpha=dalpha,
            NET=dnet,
        )

        self.data.obs[0]["tod"] = tod
        self.data.obs[0]["noise"] = noise
        self.data.obs[0]["intervals"] = intervals

        # Write processing masks

        npix = 12 * self.map_nside ** 2
        pix = np.arange(npix)
        pix = hp.reorder(pix, r2n=True)

        self.binary_mask = np.logical_or(pix < npix * 0.45, pix > npix * 0.55)
        self.maskfile_binary = os.path.join(self.outdir, "binary_mask.fits")
        hp.write_map(
            self.maskfile_binary,
            self.binary_mask,
            dtype=np.float32,
            overwrite=True,
            nest=True,
        )

        self.smooth_mask = (np.abs(pix - npix / 2) / (npix / 2)) ** 0.5
        self.maskfile_smooth = os.path.join(self.outdir, "apodized_mask.fits")
        hp.write_map(
            self.maskfile_smooth,
            self.smooth_mask,
            dtype=np.float32,
            overwrite=True,
            nest=True,
        )

        return

    def test_mapmaker_gradient(self):

        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(
            nside=self.map_nside, nest=True, mode=self.pointingmode
        )
        pointing.exec(self.data)

        # Add simulated noise
        name = "grad"
        opnoise = OpSimNoise(realization=0, out=name)
        opnoise.exec(self.data)

        # Copy the signal to run with Madam
        name_madam = "grad_copy"
        cachecopy = OpCacheCopy(name, name_madam)
        cachecopy.exec(self.data)

        # Run TOAST mapmaker

        mapmaker = OpMapMaker(
            nside=self.map_nside,
            nnz=self.nnz,
            name=name,
            outdir=self.outdir,
            outprefix="toast_",
            baseline_length=1,
            maskfile=self.maskfile_binary,
            #weightmapfile=self.maskfile_smooth,
        )
        mapmaker.exec(self.data)

        # Run Madam

        pars = {}
        pars["kfirst"] = "T"
        pars["base_first"] = 1.0
        pars["fsample"] = self.rate
        pars["nside_map"] = self.map_nside
        pars["nside_cross"] = self.map_nside
        pars["nside_submap"] = min(8, self.map_nside)
        pars["write_map"] = "T"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["file_inmask"] = self.maskfile_binary
        pars["kfilter"] = "T"
        pars["path_output"] = self.outdir
        pars["file_root"] = "madam"
        pars["info"] = 3

        madam = OpMadam(params=pars, name=name_madam, flag_mask=1)
        if not madam.available:
            print("libmadam not available, skipping mapmaker comparison")
            return

        madam.exec(self.data)

        # Compare

        m0 = None
        if self.rank == 0:
            m0 = hp.read_map(os.path.join(self.outdir, "toast_binned.fits"))
        m1 = None
        if self.rank == 0:
            m1 = hp.read_map(os.path.join(self.outdir, "madam_bmap.fits"))

        failed = False
        if self.rank == 0:
            if not np.allclose(m0[self.binary_mask], m1[self.binary_mask]):
                print("TOAST mapmaker and Madam do not agree.")
                failed = True
        if self.comm is not None:
            failed = self.comm.bcast(failed, root=0)
        self.assertFalse(failed)

        return


if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
    if rank == 0:
        print("Running test")
    myTest = OpMapMakerTest()
    myTest.comm = comm
    myTest.setUp()
    myTest.test_mapmaker_gradient()
