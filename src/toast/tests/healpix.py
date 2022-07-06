# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np

from ..healpix import Pixels, ang2vec, vec2ang
from ..rng import random
from ._helpers import create_outdir
from .mpi import MPITestCase


class HealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 256
        self.eps32 = np.finfo(np.float32).eps
        self.eps64 = np.finfo(np.float64).eps

        theta = [
            0.0,
            self.eps64,
            self.eps32,
            np.radians(90.0) - self.eps32,
            np.radians(90.0) - self.eps64,
            np.radians(90.0),
            np.radians(90.0) + self.eps64,
            np.radians(90.0) + self.eps32,
            np.radians(180.0) - self.eps32,
            np.radians(180.0) - self.eps64,
            np.radians(180.0),
        ]
        phi = list()
        for pts in [0.0, 90.0, 180.0, 270.0, 360.0]:
            phi.append(np.radians(pts) - self.eps32)
            phi.append(np.radians(pts) - self.eps32)
            phi.append(np.radians(pts))
            phi.append(np.radians(pts) + self.eps32)
            phi.append(np.radians(pts) + self.eps32)

        self.extremes = list()
        self.extcompnest = list()
        self.extcompring = list()
        for th in theta:
            for ph in phi:
                self.extremes.append((th, ph))
                self.extcompnest.append(hp.ang2pix(self.nside, th, ph, nest=True))
                self.extcompring.append(hp.ang2pix(self.nside, th, ph, nest=False))

        self.nreg = 500
        self.regular = list()
        self.regcompnest = list()
        self.regcompring = list()
        for th in range(self.nreg):
            for ph in range(self.nreg):
                theta = np.radians(th * 180.0 / self.nreg)
                phi = np.radians(ph * 360.0 / self.nreg)
                self.regular.append((theta, phi))
                self.regcompnest.append(hp.ang2pix(self.nside, theta, phi, nest=True))
                self.regcompring.append(hp.ang2pix(self.nside, theta, phi, nest=False))

    def test_roundtrip(self):
        # Test roundtrip except for the pole, where input phi will not
        # match the zero values returned.
        theta = np.array([x[0] for x in self.regular if x[0] > 0])
        phi = np.array([x[1] for x in self.regular if x[0] > 0])
        vec = ang2vec(theta, phi)
        comptheta, compphi = vec2ang(vec)
        np.testing.assert_array_almost_equal(comptheta, theta)
        np.testing.assert_array_almost_equal(compphi, phi)

    def test_pix(self):
        theta = np.array([x[0] for x in self.regular])
        phi = np.array([x[1] for x in self.regular])
        hpix = Pixels(nside=self.nside)
        pixnest = hpix.ang2nest(theta, phi)
        pixring = hpix.ang2ring(theta, phi)
        # for th, ph, nst, hnst, rng, hrng in zip(
        #     theta, phi, pixnest, self.regcompnest, pixring, self.regcompring
        # ):
        #     if (nst != hnst) or (rng != hrng):
        #         print(th, ph, nst, hnst, rng, hrng, "  <-----", flush=True)
        #     else:
        #         print(th, ph, nst, hnst, rng, hrng, flush=True)
        np.testing.assert_equal(pixnest, self.regcompnest)
        np.testing.assert_equal(pixring, self.regcompring)
