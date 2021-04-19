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
import scipy.sparse

from ..timing import gather_timers, GlobalTimers
from ..timing import dump as dump_timing
from ..tod import (
    AnalyticNoise, OpSimNoise, Interval, OpCacheCopy, OpCacheInit,
)
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


def combine_observation_matrix(rootname):
    pattern = f"{rootname}.*.*.*.data.npy"
    datafiles = sorted(glob.glob(pattern))
    if len(datafiles) == 0:
        raise RuntimeError(f"No observation matrix files match '{pattern}''")

    all_data = []
    all_indices = []
    all_indptr = [0]

    current_row = 0
    current_offset = 0
    shape = None

    for datafile in datafiles:
        parts = datafile.split(".")
        row_start = int(parts[-5])
        row_stop = int(parts[-4])
        nrow_tot = int(parts[-3])
        if shape is None:
            shape = (nrow_tot, nrow_tot)
        elif shape[0] != nrow_tot:
            raise RuntimeError("Mismatch in shape")
        if current_row != row_start:
            all_indptr.append(np.zeros(row_start - current_row) + current_offset)
            current_row = row_start
        data = np.load(datafile)
        indices = np.load(datafile.replace(".data.", ".indices."))
        indptr = np.load(datafile.replace(".data.", ".indptr."))
        all_data.append(data)
        all_indices.append(indices)
        indptr += current_offset
        all_indptr.append(indptr[1:])
        current_row = row_stop
        current_offset = indptr[-1]

    if current_row != nrow_tot:
        all_indptr.append(np.zeros(nrow_tot - current_row) + current_offset)

    all_data = np.hstack(all_data)
    all_indices = np.hstack(all_indices)
    all_indptr = np.hstack(all_indptr)
    obs_matrix = scipy.sparse.csr_matrix((all_data, all_indices, all_indptr), shape)
    scipy.sparse.save_npz(rootname, obs_matrix)

    return


class OpFilterBinTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.rank = 0
        self.ntask = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.ntask = self.comm.size
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

        self.ndet = 4 * self.ntask
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
            pairs=True,
        )

        # Pixelization
        self.sim_nside = 8
        self.map_nside = 8
        self.nnz = 3
        self.pointingmode = "IQU"[: self.nnz]

        # Samples per observation
        self.npix = 12 * self.sim_nside ** 2
        self.ninterval = 4
        self.totsamp = self.ninterval * self.npix

        # Define intervals with gaps in between
        intervals = []
        interval_length = self.npix
        istart = 0
        while istart < self.totsamp:
            istop = istart + interval_length
            interval_start = istart + 100
            interval_stop = istop - 100
            intervals.append(
                Interval(
                    start=interval_start / self.rate,
                    stop=interval_stop / self.rate,
                    first=interval_start,
                    last=interval_stop - 1,
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
            elif iobs % 3 == 2:
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

            cflags = tod.local_common_flags()
            cflags[:] = 255
            for ival in tod.local_intervals(intervals):
                cflags[ival.first : ival.last + 1] = 0

            tod._az = (tod.local_times() % 100) / 100
            tod.scan_range = [0, 1, 0, 0]

            def read_boresight_az(self, local_start=0, n=None):
                if n is None:
                    n = self.local_samples[1] - local_start
                ind = slice(local_start, local_start + n)
                return tod._az[ind]

            tod.read_boresight_az = read_boresight_az.__get__(tod)

        self.npix = 12 * self.map_nside ** 2
        pix = np.arange(self.npix)
        pix = hp.reorder(pix, r2n=True)

        # Synthesize an input map
        self.lmax = 2 * self.sim_nside
        self.cl = np.ones([4, self.lmax + 1])
        self.cl[:, 0:2] = 0
        #self.cl[1:] = 0  # DEBUG
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

        # Force some gaps in the observing
        for obs in self.data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                pix = tod.cache.reference("pixels_" + det)
                n = 3 * self.map_nside ** 2
                pix[0:n] = pix[2 * n : n : -1]
                pix[3 * n : 4 * n] = pix[3 * n : 2 * n : -1]

        # Scan the signal from a map
        distmap = DistPixels(self.data, nnz=self.nnz, dtype=np.float32)
        distmap.read_healpix_fits(self.inmapfile)

        scansim = OpSimScan(input_map=distmap, out=name)
        scansim.exec(self.data)

        # Run FilterBin

        gt = GlobalTimers.get()
        gt.start("OpMapMaker test")

        outprefix = "toast_test_"

        filterbin = OpFilterBin(
            nside=self.map_nside,
            nnz=self.nnz,
            name=name,
            outdir=self.outdir,
            outprefix=outprefix,
            write_obs_matrix=True,
            ground_filter_order=3,
            poly_filter_order=20,
            zip_maps=True,
            common_flag_mask=255,
            write_binned=True,
            write_hits=True,
            write_wcov_inv=True,
            write_wcov=True,
        )
        filterbin.exec(self.data, self.comm)

        if self.rank == 0:

            # Test that we can replicate the filtering by applying
            # the observation matrix to the input map

            rootname = os.path.join(self.outdir, outprefix + "obs_matrix")
            combine_observation_matrix(rootname)
            fname = rootname + ".npz"
            obs_matrix = scipy.sparse.load_npz(fname)

            fname = os.path.join(self.outdir, outprefix + "filtered.fits.gz")
            outmap = hp.read_map(fname, None, nest=True)

            inmap = hp.read_map(self.inmapfile, None, nest=True)
            outmap_test = obs_matrix.dot(inmap.ravel()).reshape([self.nnz, -1])

            np.testing.assert_array_almost_equal(outmap, outmap_test)

        if self.comm is not None:
            self.comm.Barrier()

        return

    def test_filterbin_deproject(self):

        name = "testtod2"
        init = OpCacheInit(name=name, init_val=0)

        # make a simple pointing matrix
        pointing = OpPointingHpix(
            nside=self.map_nside, nest=True, mode=self.pointingmode
        )
        pointing.exec(self.data)

        # Force detector pairs to share pointing.
        for obs in self.data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                if det.endswith("A"):
                    pairdet = det[:-1] + "B"
                    pixels = tod.cache.reference("pixels_" + det)
                    pairpixels = tod.cache.reference("pixels_" + pairdet)
                    pairpixels[:] = pixels

        # Scan the signal from a map
        distmap = DistPixels(self.data, nnz=self.nnz, dtype=np.float32)
        distmap.read_healpix_fits(self.inmapfile)

        scansim = OpSimScan(input_map=distmap, out=name)
        scansim.exec(self.data)

        # Scramble the gains
        for obs in self.data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                if det.endswith("A"):
                    pairdet = det[:-1] + "B"
                    signal = tod.local_signal(det, name)
                    pairsignal = tod.local_signal(pairdet, name)
                    gain = 1 + np.random.randn() * 1e-2
                    signal *= gain
                    pairsignal /= gain

        # Sum/diff the signal
        weights_in = "weights"
        weights_out = "pairweights"
        signal_in = name
        signal_out = "pairtod"
        for obs in self.data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                if det.endswith("A"):
                    pairdet = det[:-1] + "B"
                    signal = tod.local_signal(det, signal_in)
                    pairsignal = tod.local_signal(pairdet, signal_in)
                    tod.cache.put(
                        signal_out + "_" + det,
                        0.5 * (signal + pairsignal),
                        replace=True,
                    )
                    tod.cache.put(
                        signal_out + "_" + pairdet,
                        0.5 * (signal - pairsignal),
                        replace=True,
                    )
                    weights = tod.cache.reference(weights_in + "_" + det)
                    pairweights = tod.cache.reference(weights_in + "_" + pairdet)
                    tod.cache.put(
                        weights_out + "_" + det,
                        0.5 * (weights + pairweights),
                        replace=True,
                    )
                    tod.cache.put(
                        weights_out + "_" + pairdet,
                        0.5 * (weights - pairweights),
                        replace=True,
                    )
                    pixels = tod.cache.reference("pixels_" + det)
                    pairpixels = tod.cache.reference("pixels_" + pairdet)

        # Make a deprojection template map
        dpmap_file = os.path.join(self.outdir, "deprojection_map.fits")
        if self.rank == 0:
            tmap = hp.read_map(self.inmapfile, nest=True)
            hp.write_map(dpmap_file, tmap, nest=True)
        if self.comm is not None:
            self.comm.Barrier()
            
        # Run FilterBin

        gt = GlobalTimers.get()
        gt.start("OpMapMaker test")

        outprefix = "toast_test2_"

        filterbin = OpFilterBin(
            nside=self.map_nside,
            nnz=self.nnz,
            name=signal_out,
            weights_name=weights_out,
            outdir=self.outdir,
            outprefix=outprefix,
            write_obs_matrix=True,
            ground_filter_order=3,
            poly_filter_order=20,
            zip_maps=True,
            common_flag_mask=255,
            write_binned=True,
            write_hits=True,
            write_wcov_inv=True,
            write_wcov=True,
            deproject_map=dpmap_file,
            deproject_pattern=".*B",
            deproject_nnz=1,
        )
        filterbin.exec(self.data, self.comm)

        if self.rank == 0:

            # Test that we can replicate the filtering by applying
            # the observation matrix to the input map

            rootname = os.path.join(self.outdir, outprefix + "obs_matrix")
            combine_observation_matrix(rootname)
            fname = rootname + ".npz"
            obs_matrix = scipy.sparse.load_npz(fname)

            fname = os.path.join(self.outdir, outprefix + "filtered.fits.gz")
            outmap = hp.read_map(fname, None, nest=True)

            inmap = hp.read_map(self.inmapfile, None, nest=True)
            outmap_test = obs_matrix.dot(inmap.ravel()).reshape([self.nnz, -1])

            # The gain fluctuations can change the overall calibration

            ratio = np.std(outmap[1:]) / np.std(outmap_test[1:])
            outmap_test *= ratio

            np.testing.assert_array_almost_equal(outmap, outmap_test)

        if self.comm is not None:
            self.comm.Barrier()

        return
