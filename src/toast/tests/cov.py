# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import numpy as np

import numpy.testing as nt

import healpy as hp

from ..tod import AnalyticNoise, OpSimNoise
from ..todmap import TODSatellite, OpPointingHpix, OpAccumDiag
from ..todmap.todmap_math import cov_accum_diag
from ..map import DistPixels, covariance_invert, covariance_rcond, covariance_multiply

from ._helpers import (
    create_outdir,
    create_distdata,
    boresight_focalplane,
    uniform_chunks,
)


class CovarianceTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # a fixed number of detectors and one chunk per process.

        self.data = create_distdata(self.comm, obs_per_group=1)

        self.ndet = 1
        self.rate = 40.0
        self.hwprpm = 50

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
        ) = boresight_focalplane(self.ndet, samplerate=self.rate, net=7.0)

        # Samples per observation
        self.totsamp = 240000

        # Pixelization

        self.sim_nside = 32
        self.sim_npix = 12 * self.sim_nside ** 2

        self.map_nside = 32
        self.map_npix = 12 * self.map_nside ** 2

        # Scan strategy

        self.spinperiod = 10.0
        self.spinangle = 30.0
        self.precperiod = 50.0
        self.precangle = 65.0

        # One chunk per process
        chunks = uniform_chunks(self.totsamp, nchunk=self.data.comm.group_size)

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
            hwprpm=self.hwprpm,
        )

        tod.set_prec_axis()

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

    def tearDown(self):
        del self.data

    def test_accum(self):
        nsm = 2
        npix = 3
        nnz = 4
        block = int(nnz * (nnz + 1) / 2)
        scale = 2.0
        nsamp = nsm * npix

        fake = DistPixels(
            None,
            comm=self.data.comm.comm_world,
            npix=nsm * npix,
            nnz=nnz,
            dtype=np.float64,
            npix_submap=npix,
            local_submaps=np.arange(nsm, dtype=np.int64),
        )
        check = fake.duplicate()

        hits = DistPixels(
            None,
            comm=self.data.comm.comm_world,
            npix=nsm * npix,
            nnz=1,
            dtype=np.int64,
            npix_submap=npix,
            local_submaps=np.arange(nsm, dtype=np.int64),
        )
        checkhits = hits.duplicate()

        invn = DistPixels(
            None,
            comm=self.data.comm.comm_world,
            npix=nsm * npix,
            nnz=block,
            dtype=np.float64,
            npix_submap=npix,
            local_submaps=np.arange(nsm, dtype=np.int64),
        )
        checkinvn = invn.duplicate()

        sm = np.zeros(nsamp, dtype=np.int64)
        pix = np.zeros(nsamp, dtype=np.int64)
        wt = np.zeros(nsamp * nnz, dtype=np.float64)

        for i in range(nsamp):
            sm[i] = i % nsm
            pix[i] = i % npix
            for k in range(nnz):
                wt[i * nnz + k] = float(k + 1)

        signal = np.random.normal(size=nsamp)
        #
        # print(
        #     nsm,
        #     npix,
        #     nnz,
        #     sm.dtype,
        #     pix.dtype,
        #     wt.dtype,
        #     scale,
        #     signal.dtype,
        #     invn.flatdata.dtype,
        #     hits.flatdata.dtype,
        #     fake.flatdata.dtype,
        #     flush=True,
        # )

        cov_accum_diag(
            nsm,
            npix,
            nnz,
            sm,
            pix,
            wt,
            scale,
            signal,
            invn.flatdata,
            hits.flatdata,
            fake.flatdata,
        )

        for i in range(nsamp):
            checkhits.data[sm[i], pix[i], 0] += 1
            off = 0
            for j in range(nnz):
                check.data[sm[i], pix[i], j] += scale * signal[i] * wt[i * nnz + j]
                for k in range(j, nnz):
                    checkinvn.data[sm[i], pix[i], off] += (
                        scale * wt[i * nnz + j] * wt[i * nnz + k]
                    )
                    off += 1

        # for i in range(nsamp):
        #     print("{}: {} {}".format(i, checkhits.data[sm[i], pix[i], 0], hits.data[sm[i], pix[i], 0]))
        #     off = 0
        #     for j in range(nnz):
        #         print("  {}:  {}  {}".format(j, check.data[sm[i], pix[i], j], fake.data[sm[i], pix[i], j]))
        #         for k in range(j, nnz):
        #             print("    {}:  {}  {}".format(off, checkinvn.data[sm[i], pix[i], off], invn.data[sm[i], pix[i], off]))
        #             off += 1

        nt.assert_equal(hits.data, checkhits.data)
        nt.assert_almost_equal(fake.data, check.data)
        nt.assert_almost_equal(invn.data, checkinvn.data)

        return

    def test_invert(self):
        nsm = 2
        npix = 3
        nnz = 4
        scale = 2.0
        nsamp = nsm * npix
        nelem = int(nnz * (nnz + 1) / 2)
        threshold = 1.0e-6

        invn = DistPixels(
            None,
            comm=self.data.comm.comm_world,
            npix=nsm * npix,
            nnz=nelem,
            dtype=np.float64,
            npix_submap=npix,
            local_submaps=np.arange(nsm, dtype=np.int64),
        )
        check = invn.duplicate()

        rowdata = 10.0 * np.arange(nnz, 0, -1)

        for i in range(nsm):
            for j in range(npix):
                off = 0
                for k in range(nnz):
                    for m in range(k, nnz):
                        invn.data[i, j, off] = rowdata[m - k]
                        check.data[i, j, off] = invn.data[i, j, off]
                        off += 1

        # invert twice
        covariance_invert(invn, threshold)
        covariance_invert(invn, threshold)

        # for i in range(nsm):
        #     for j in range(npix):
        #         off = 0
        #         print("sm {}, pix {}:".format(i, j))
        #         for k in range(nnz):
        #             for m in range(k, nnz):
        #                 print("  {} {}".format(fakedata[i,j,off], checkdata[i,j,off]))
        #                 off += 1

        nt.assert_almost_equal(invn.data, check.data)
        return

    def test_invnpp(self):
        op = OpSimNoise(realization=0)
        op.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode="IQU")
        pointing.exec(self.data)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(self.data, nnz=6, dtype=np.float64,)
        invnpp.data.fill(0.0)

        hits = DistPixels(self.data, nnz=1, dtype=np.int64,)
        hits.data.fill(0)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]["tod"]
        nse = self.data.obs[0]["noise"]
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d) ** 2)

        build_invnpp = OpAccumDiag(
            detweights=detweights, invnpp=invnpp, hits=hits, name="noise"
        )
        build_invnpp.exec(self.data)

        # for i in range(invnpp.data.shape[0]):
        #     for j in range(invnpp.data.shape[1]):
        #         print("sm {}, pix {}:  hits = {}".format(i, j, hits.data[i, j, 0]))
        #         for k in range(invnpp.data.shape[2]):
        #             print("  {}".format(invnpp.data[i, j, k]))

        invnpp.allreduce()
        hits.allreduce()

        # invert it
        check = invnpp.duplicate()
        covariance_invert(invnpp, 1.0e-14)
        covariance_invert(invnpp, 1.0e-14)

        # Matrices that failed the rcond test are set to zero
        nonzero = np.absolute(invnpp.data) > 1.0e-12
        if np.sum(nonzero) == 0:
            raise Exception("All matrices failed the rcond test.")

        nt.assert_almost_equal(invnpp.data[nonzero], check.data[nonzero])

        return

    def test_distpix_init(self):
        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode="IQU")
        pointing.exec(self.data)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(self.data, nnz=6, dtype=np.float64,)

        return

    def test_multiply(self):
        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode="IQU")
        pointing.exec(self.data)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(self.data, nnz=6, dtype=np.float64,)

        hits = DistPixels(self.data, nnz=1, dtype=np.int64,)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]["tod"]
        nse = self.data.obs[0]["noise"]
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d) ** 2)

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        invnpp.allreduce()
        hits.allreduce()

        # invert it
        check = invnpp.duplicate()
        covariance_invert(invnpp, 1.0e-3)

        # multiply the two
        covariance_multiply(check, invnpp)

        # check that the multiplied matrices are unit matrices
        nsubmap, npix, nblock = check.data.shape
        nnz = int(((np.sqrt(8 * nblock) - 1) / 2) + 0.5)
        for i in range(nsubmap):
            for j in range(npix):
                if np.all(invnpp.data[i, j] == 0):
                    # Matrix failed to invert
                    continue
                off = 0
                for k in range(nnz):
                    for m in range(k, nnz):
                        if k == m:
                            nt.assert_almost_equal(check.data[i, j, off], 1.0)
                        else:
                            nt.assert_almost_equal(check.data[i, j, off], 0.0)
                        off += 1

        return

    def test_fitsio(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True, mode="IQU")
        pointing.exec(self.data)

        # construct a distributed map to store the covariance and hits

        invnpp = DistPixels(self.data, nnz=6, dtype=np.float64,)

        rcond = DistPixels(self.data, nnz=1, dtype=np.float64,)

        hits = DistPixels(self.data, nnz=1, dtype=np.int64,)

        # accumulate the inverse covariance.  Use detector weights
        # based on the analytic NET.

        tod = self.data.obs[0]["tod"]
        nse = self.data.obs[0]["noise"]
        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0 / (self.rate * nse.NET(d) ** 2)

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits)
        build_invnpp.exec(self.data)

        # self.assertTrue(False)

        invnpp.allreduce()
        hits.allreduce()

        # invert it

        covariance_invert(invnpp, 1.0e-3)
        rcond = covariance_rcond(invnpp)

        check = invnpp.duplicate()
        checkhits = hits.duplicate()
        checkrcond = rcond.duplicate()

        # write this out...

        subsum = [
            np.sum(invnpp.data[x, :, :]) for x in range(len(invnpp.local_submaps))
        ]

        outfile = os.path.join(self.outdir, "covtest.fits")
        if rank == 0:
            if os.path.isfile(outfile):
                os.remove(outfile)

        hitfile = os.path.join(self.outdir, "covtest_hits.fits")
        if rank == 0:
            if os.path.isfile(hitfile):
                os.remove(hitfile)

        rcondfile = os.path.join(self.outdir, "covtest_rcond.fits")
        if rank == 0:
            if os.path.isfile(rcondfile):
                os.remove(rcondfile)

        invnpp.write_healpix_fits(outfile)
        rcond.write_healpix_fits(rcondfile)
        hits.write_healpix_fits(hitfile)

        invnpp.data.fill(0.0)
        invnpp.read_healpix_fits(outfile)

        diffdata = invnpp.duplicate()
        diffdata.data -= check.data

        difffile = os.path.join(self.outdir, "readwrite_diff.fits")
        diffdata.write_healpix_fits(difffile)

        if rank == 0:
            import matplotlib.pyplot as plt

            dat = hp.read_map(outfile)
            outfile = "{}.png".format(outfile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

            dat = hp.read_map(difffile)
            outfile = "{}.png".format(difffile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

            dat = hp.read_map(hitfile)
            outfile = "{}.png".format(hitfile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

            dat = hp.read_map(rcondfile)
            outfile = "{}.png".format(rcondfile)
            hp.mollview(dat, xsize=int(1600))
            plt.savefig(outfile)
            plt.close()

        rcond.data.fill(0.0)
        rcond.read_healpix_fits(rcondfile)

        nt.assert_almost_equal(rcond.data, checkrcond.data, decimal=6)
        # nt.assert_almost_equal(invnpp.data, checkdata, decimal=6)

        hits.data.fill(0)
        hits.read_healpix_fits(hitfile)

        nt.assert_equal(hits.data, checkhits.data)

        return
