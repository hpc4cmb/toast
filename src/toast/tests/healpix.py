# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np

from ..healpix import (
    ang2nest,
    ang2ring,
    ang2vec,
    degrade_nest,
    degrade_ring,
    nest2ring,
    ring2nest,
    upgrade_nest,
    upgrade_ring,
    vec2ang,
)
from .mpi import MPITestCase


class HealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]

    def create_hp_data(self, nside):
        eps32 = np.finfo(np.float32).eps
        eps64 = np.finfo(np.float64).eps

        theta = [
            0.0,
            eps64,
            eps32,
            np.radians(90.0) - eps32,
            np.radians(90.0) - eps64,
            np.radians(90.0),
            np.radians(90.0) + eps64,
            np.radians(90.0) + eps32,
            np.radians(180.0) - eps32,
            np.radians(180.0) - eps64,
            np.radians(180.0),
        ]
        phi = list()
        for pts in [0.0, 90.0, 180.0, 270.0, 360.0]:
            phi.append(np.radians(pts) - eps32)
            phi.append(np.radians(pts) - eps32)
            phi.append(np.radians(pts))
            phi.append(np.radians(pts) + eps32)
            phi.append(np.radians(pts) + eps32)

        extremes = list()
        extcompnest = list()
        extcompring = list()
        for th in theta:
            for ph in phi:
                extremes.append((th, ph))
                extcompnest.append(hp.ang2pix(nside, th, ph, nest=True))
                extcompring.append(hp.ang2pix(nside, th, ph, nest=False))

        nreg = 100
        regular = list()
        regcompnest = list()
        regcompring = list()
        for th in range(nreg):
            for ph in range(nreg):
                theta = np.radians(th * 180.0 / nreg)
                phi = np.radians(ph * 360.0 / nreg)
                regular.append((theta, phi))
                regcompnest.append(hp.ang2pix(nside, theta, phi, nest=True))
                regcompring.append(hp.ang2pix(nside, theta, phi, nest=False))
        return (regular, regcompnest, regcompring, extremes, extcompnest, extcompring)

    def test_ang_vec(self):
        # Test roundtrip except for the pole, where input phi will not
        # match the zero values returned.
        for nside in [1, 256, 16384]:
            (
                regular,
                regcompnest,
                regcompring,
                extremes,
                extcompnest,
                extcompring,
            ) = self.create_hp_data(nside)
            theta = np.array([x[0] for x in regular if x[0] > 0])
            phi = np.array([x[1] for x in regular if x[0] > 0])
            vec = ang2vec(theta, phi)
            comptheta, compphi = vec2ang(vec.reshape((-1, 3)))
            np.testing.assert_array_almost_equal(comptheta, theta)
            np.testing.assert_array_almost_equal(compphi, phi)

    def test_ang_pix(self):
        for nside in [1, 256, 16384]:
            (
                regular,
                regcompnest,
                regcompring,
                extremes,
                extcompnest,
                extcompring,
            ) = self.create_hp_data(nside)
            for ang, ring, nest in zip(regular, regcompring, regcompnest):
                pring = ang2ring(nside, ang[0], ang[1])
                if pring != ring:
                    self.assertTrue(False)
                pnest = ang2nest(nside, ang[0], ang[1])
                if pnest != nest:
                    self.assertTrue(False)
            for ang, ring, nest in zip(extremes, extcompring, extcompnest):
                pring = ang2ring(nside, ang[0], ang[1])
                if pring != ring:
                    self.assertTrue(False)
                pnest = ang2nest(nside, ang[0], ang[1])
                if pnest != nest:
                    self.assertTrue(False)

    def test_ring_nest(self):
        nsamp = 100
        for nside in [1, 256, 16384]:
            npix = 12 * nside**2
            in_pix = np.linspace(0, npix - 1, num=nsamp, endpoint=True, dtype=np.int64)
            nest_pix = hp.ring2nest(nside, in_pix)
            ring_pix = hp.nest2ring(nside, nest_pix)

            # Just check our use of healpy...
            np.testing.assert_equal(ring_pix, in_pix)

            # Now do the same conversion with internal tools
            nest = ring2nest(nside, ring_pix)
            ring = nest2ring(nside, nest_pix)
            np.testing.assert_equal(nest, nest_pix)
            np.testing.assert_equal(ring, ring_pix)

    def test_degrade_upgrade(self):
        nsamp = 100
        factor = 2
        for nside in [1, 256, 16384]:
            npix = 12 * nside**2
            ring_pix = np.linspace(
                0, npix - 1, num=nsamp, endpoint=True, dtype=np.int64
            )
            nest_pix = hp.ring2nest(nside, ring_pix)
            up_nside = nside * 2**factor

            up_nest = upgrade_nest(nside, factor, nest_pix)
            deg_nest = degrade_nest(up_nside, factor, up_nest)
            up_ring = upgrade_ring(nside, factor, ring_pix)
            deg_ring = degrade_ring(up_nside, factor, up_ring)

            np.testing.assert_equal(deg_ring, ring_pix)
            np.testing.assert_equal(deg_nest, nest_pix)

    def test_all_sphere(self):
        nside = 16384
        angperring = 10
        nring = 7
        thetainc = 0.98 * (np.pi / nring)
        phiinc = 0.98 * (2.0 * np.pi / angperring)
        n = nring * angperring
        theta = np.zeros(n)
        phi = np.zeros(n)
        i = 0
        for t in range(nring):
            for p in range(angperring):
                theta[i] = t * thetainc
                phi[i] = p * phiinc
                i += 1
        pix_ring = hp.ang2pix(nside, theta, phi, nest=False)
        pix_nest = hp.ang2pix(nside, theta, phi, nest=True)
        pix_r2n = hp.ring2nest(nside, pix_ring)
        pix_n2r = hp.nest2ring(nside, pix_nest)

        comp_ring = ang2ring(nside, theta, phi)
        comp_nest = ang2nest(nside, theta, phi)
        np.testing.assert_equal(comp_ring, pix_ring)
        np.testing.assert_equal(comp_nest, pix_nest)

        comp_r2n = ring2nest(nside, comp_ring)
        comp_n2r = nest2ring(nside, comp_nest)
        np.testing.assert_equal(comp_n2r, pix_n2r)
        np.testing.assert_equal(comp_r2n, pix_r2n)
