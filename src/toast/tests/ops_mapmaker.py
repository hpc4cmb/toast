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
    OpMadam,
    OpMapMaker,
    OpSimScan,
)
from ..todmap.mapmaker import TemplateMatrix, OffsetTemplate, Signal
from .. import qarray as qa

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
        if self.comm is not None:
            self.comm.barrier()

        self.nobs = 1
        self.data = create_distdata(self.comm, obs_per_group=self.nobs)

        self.ndet = self.data.comm.group_size
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
        self.sim_nside = 32
        self.map_nside = 32
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

        # Write processing masks

        self.npix = 12 * self.map_nside ** 2
        pix = np.arange(self.npix)
        pix = hp.reorder(pix, r2n=True)

        self.binary_mask = np.logical_or(pix < self.npix * 0.45, pix > self.npix * 0.55)
        self.maskfile_binary = os.path.join(self.outdir, "binary_mask.fits")
        hp.write_map(
            self.maskfile_binary,
            self.binary_mask,
            dtype=np.float32,
            overwrite=True,
            nest=True,
        )

        self.smooth_mask = (np.abs(pix - self.npix / 2) / (self.npix / 2)) ** 0.5
        self.maskfile_smooth = os.path.join(self.outdir, "apodized_mask.fits")
        hp.write_map(
            self.maskfile_smooth,
            self.smooth_mask,
            dtype=np.float32,
            overwrite=True,
            nest=True,
        )

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

    """
    def test_project_signal_offsets(self):
        from .._libtoast import project_signal_offsets

        x = np.arange(1000, dtype=np.float64)
        todslice = slice(100, 200)
        amplitudes = np.zeros(100, dtype=np.float64)
        itemplate = 10
        testvalue = np.sum(x[todslice])
        project_signal_offsets(x, [todslice], amplitudes, [itemplate])
        print("testvalue =", testvalue, ", value = ", amplitudes[itemplate], flush=True)
        # Timing
        from ..mpi import MPI

        n = 1000000
        t1 = MPI.Wtime()
        for i in range(n):
            testvalue = np.sum(x[todslice])
        t2 = MPI.Wtime()
        todslices = []
        itemplates = []
        for i in range(n):
            todslices.append(todslice)
            itemplates.append(itemplate)
        project_signal_offsets(x, todslices, amplitudes, np.array(itemplates))
        t3 = MPI.Wtime()
        print("Time1 =", t2 - t1, ", Time2 =", t3 - t2, flush=True)
    """

    """
    def test_subharmonic_template(self):

        name = "testtod1"
        init = OpCacheInit(name=name, init_val=0)

        # make a simple pointing matrix
        pointing = OpPointingHpix(
            nside=self.map_nside, nest=True, mode=self.pointingmode
        )
        pointing.exec(self.data)

        # Scan the signal from a map
        distmap = DistPixels(
            comm=self.comm,
            size=self.npix,
            nnz=self.nnz,
            dtype=np.float32,
            submap=subnpix,
            local=localsm,
        )
        distmap.read_healpix_fits(self.inmapfile)
        scansim = OpSimScan(distmap=distmap, out=name)
        scansim.exec(self.data)

        # add a sharp gradient to one of the detectors
        for obs in self.data.obs:
            tod = obs["tod"]
            sig = tod.local_signal("d01", name)
            sig[:] += np.arange(sig.size)

        # Run TOAST mapmaker

        mapmaker = OpMapMaker(
            nside=self.map_nside,
            nnz=self.nnz,
            name=name,
            outdir=self.outdir,
            outprefix="toast_subharmonic_test_",
            baseline_length=None,
            # maskfile=self.maskfile_binary,
            # weightmapfile=self.maskfile_smooth,
            subharmonic_order=1,
        )
        mapmaker.exec(self.data)

        # Run the mapmaker again

        mapmaker = OpMapMaker(
            nside=self.map_nside,
            nnz=self.nnz,
            name=name,
            outdir=self.outdir,
            outprefix="toast_subharmonic_test2_",
            baseline_length=None,
            # maskfile=self.maskfile_binary,
            # weightmapfile=self.maskfile_smooth,
            subharmonic_order=1,
        )
        mapmaker.exec(self.data)

        # Compare

        m1 = None
        if self.rank == 0:
            m1 = hp.read_map(
                os.path.join(self.outdir, "toast_subharmonic_test2_destriped.fits"),
                None,
                nest=True,
            )

        failed = False
        if self.rank == 0:
            if not np.allclose(self.inmap[1], m1[1], atol=1e-6, rtol=1e-6):
                print("Input and output maps do not agree.")
                failed = True
        if self.comm is not None:
            failed = self.comm.bcast(failed, root=0)
        self.assertFalse(failed)

        return
    """

    """
    def test_offset_template(self):

        name = "testtod2"
        init = OpCacheInit(name=name, init_val=0)

        # Add simulated noise
        opnoise = OpSimNoise(realization=0, out=name)
        opnoise.exec(self.data)

        # Build detector weights
        all_detweights = []
        for obs in self.data.obs:
            detweights = {}
            tod = obs["tod"]
            for det in tod.local_dets:
                detweights[det] = 1
            all_detweights.append(detweights)

        # Create offset template
        step_time = 1
        step_size = int(step_time * self.rate)
        offset_template = OffsetTemplate(
            self.data,
            all_detweights,
            step_length=step_time,
            intervals="intervals",
            use_noise_prior=True,
        )

        # Create Signal
        signal = Signal(self.data, name=name)

        # Create template matrix
        templates = TemplateMatrix(self.data, self.comm, [offset_template])

        amplitudes1 = templates.zero_amplitudes()
        reference = templates.apply_transpose(signal)
        reference /= step_size
        amplitudes3 = templates.zero_amplitudes()
        templates.add_prior(reference, amplitudes3)
        print("amplitudes1:", amplitudes1)
        print("reference:", reference)
        print("amplitudes3:", amplitudes3)

        return

    """

    def test_mapmaker_madam(self):

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

        scansim = OpSimScan(distmap=distmap, out=name)
        scansim.exec(self.data)

        # Add simulated noise
        opnoise = OpSimNoise(realization=0, out=name)
        opnoise.exec(self.data)

        # Copy the signal to run with Madam
        name_madam = name + "_copy"
        cachecopy = OpCacheCopy(name, name_madam)
        cachecopy.exec(self.data)

        # Run TOAST mapmaker

        gt = GlobalTimers.get()
        gt.start("OpMapMaker test")

        mapmaker = OpMapMaker(
            nside=self.map_nside,
            nnz=self.nnz,
            name=name,
            outdir=self.outdir,
            outprefix="toast_test_",
            baseline_length=1,
            maskfile=self.maskfile_binary,
            # weightmapfile=self.maskfile_smooth,
            subharmonic_order=None,
            iter_max=100,
            use_noise_prior=True,
            precond_width=30,
        )
        mapmaker.exec(self.data)
        # User needs to set TOAST_FUNCTIME to see timing results
        mapmaker.report_timing()

        gt.stop_all()
        alltimers = gather_timers(comm=self.comm)
        if self.rank == 0:
            out = os.path.join(self.outdir, "timing")
            dump_timing(alltimers, out)
            print("Saved timers in {}".format(out))

        # Run the destriper again
        # mapmaker = OpMapMaker(
        #    nside=self.map_nside,
        #    nnz=self.nnz,
        #    name=name,
        #    outdir=self.outdir,
        #    outprefix="toast_test2_",
        #    baseline_length=1,
        #    maskfile=self.maskfile_binary,
        #    # weightmapfile=self.maskfile_smooth,
        #    subharmonic_order=None,
        #    iter_max=100,
        #    use_noise_prior=True,
        # )
        # mapmaker.exec(self.data)

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
        pars["write_wcov"] = "T"
        pars["write_hits"] = "T"
        pars["file_inmask"] = self.maskfile_binary
        pars["kfilter"] = "T"
        pars["path_output"] = self.outdir
        pars["file_root"] = "madam"
        pars["info"] = 3

        madam = OpMadam(
            params=pars,
            name=name_madam,
            name_out=name_madam,
            flag_mask=1,
            detweights=mapmaker.detweights[0],
        )
        if not madam.available:
            print("libmadam not available, skipping mapmaker comparison")
            return

        madam.exec(self.data)

        # DEBUG begin
        # import pdb
        # import matplotlib.pyplot as plt
        # tod = self.data.obs[0]["tod"]
        # sig1 = tod.local_signal("d00", name)
        # sig2 = tod.local_signal("d00", name_madam)
        # plt.plot(sig1, '.', label="TOAST mapmaker")
        # plt.plot(sig2, '.', label="Madam")
        # plt.plot(sig2 - sig1, '.', label="Madam - TOAST")
        # plt.legend(loc="best")
        # plt.savefig("test2.png")
        # pdb.set_trace()
        # DEBUG end

        # Compare

        m0 = None
        if self.rank == 0:
            m0 = hp.read_map(os.path.join(self.outdir, "toast_test_binned.fits"))
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
