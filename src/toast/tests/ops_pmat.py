# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import healpy as hp
import numpy as np

from .._libtoast import pointing_matrix_healpix
from ..healpix import HealpixPixels
from ..todmap import TODHpixSpiral, OpPointingHpix, OpLocalPixels
from .. import qarray as qa

from ._helpers import create_outdir, create_distdata, boresight_focalplane


class OpPointingHpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # one detector per process and a single chunk.  Data within an
        # observation is distributed by detector.

        self.data = create_distdata(self.comm, obs_per_group=1)
        self.ndet = self.data.comm.group_size

        # Create detectors with default properties
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = boresight_focalplane(
            self.ndet
        )

        # A small number of samples
        self.totsamp = 10

        # Populate the observations (one per group)

        tod = TODHpixSpiral(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=self.data.comm.group_size,
        )

        self.data.obs[0]["tod"] = tod

    def tearDown(self):
        del self.data

    def test_pointing_matrix_healpix2(self):
        nside = 64
        npix = 12 * nside ** 2
        hpix = HealpixPixels(64)
        nest = True
        phivec = np.radians([-360, -270, -180, -135, -90, -45, 0, 45, 90, 135, 180, 270, 360])
        nsamp = phivec.size
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        hwpang = np.zeros(nsamp)
        flags = np.zeros(nsamp, dtype=np.uint8)
        pixels = np.zeros(nsamp, dtype=np.int64)
        weights = np.zeros([nsamp, nnz], dtype=np.float64)
        theta = np.radians(135)
        psi = np.radians(135)
        quats = []
        xaxis, yaxis, zaxis = np.eye(3)
        for phi in phivec:
            phirot = qa.rotation(zaxis, phi)
            quats.append(qa.from_angles(theta, phi, psi))
        quats = np.vstack(quats)
        pointing_matrix_healpix(
            hpix,
            nest,
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            pixels,
            weights.reshape(-1),
        )
        failed = False
        bad = np.logical_or(pixels < 0, pixels > npix - 1)
        nbad = np.sum(bad)
        if nbad > 0:
            print("{} pixels are outside of the map. phi = {} deg".format(nbad, np.degrees(phivec[bad])))
            failed = True
        self.assertFalse(failed)
        return

    def test_pointing_matrix_healpix(self):
        nside = 64
        hpix = HealpixPixels(64)
        nest = True
        psivec = np.radians([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        # psivec = np.radians([-180, 180])
        nsamp = psivec.size
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        hwpang = np.zeros(nsamp)
        flags = np.zeros(nsamp, dtype=np.uint8)
        pixels = np.zeros(nsamp, dtype=np.int64)
        weights = np.zeros([nsamp, nnz], dtype=np.float64)
        pix = 49103
        theta, phi = hp.pix2ang(nside, pix, nest=nest)
        xaxis, yaxis, zaxis = np.eye(3)
        thetarot = qa.rotation(yaxis, theta)
        phirot = qa.rotation(zaxis, phi)
        pixrot = qa.mult(phirot, thetarot)
        quats = []
        for psi in psivec:
            psirot = qa.rotation(zaxis, psi)
            quats.append(qa.mult(pixrot, psirot))
        quats = np.vstack(quats)
        pointing_matrix_healpix(
            hpix,
            nest,
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            pixels,
            weights.reshape(-1),
        )
        weights_ref = []
        for quat in quats:
            theta, phi, psi = qa.to_angles(quat)
            weights_ref.append(np.array([1, np.cos(2 * psi), np.sin(2 * psi)]))
        weights_ref = np.vstack(weights_ref)
        failed = False
        for w1, w2, psi, quat in zip(weights_ref, weights, psivec, quats):
            # print("\npsi = {}, quat = {} : ".format(psi, quat), end="")
            if not np.allclose(w1, w2):
                print(
                    "Pointing weights do not agree: {} != {}".format(w1, w2), flush=True
                )
                failed = True
            else:
                # print("Pointing weights agree: {} == {}".format(w1, w2), flush=True)
                pass
        self.assertFalse(failed)
        return

    def test_hpix_simple(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        op = OpPointingHpix()
        op.exec(self.data)

        lc = OpLocalPixels()
        local = lc.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_hpix_simple_info"), "w")
        self.data.info(handle)
        if rank == 0:
            handle.close()
        return

    def test_hpix_hwpnull(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        op = OpPointingHpix(mode="IQU")
        op.exec(self.data)

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_hpix_hwpnull_info"), "w")
        self.data.info(handle)
        if rank == 0:
            handle.close()
        return
