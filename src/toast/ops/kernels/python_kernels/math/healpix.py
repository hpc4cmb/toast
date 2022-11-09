# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

TWOINVPI = 0.63661977236758134308
MACHINE_EPSILON = np.finfo(np.float).eps


class HPIX_PYTHON:
    """
    Encapsulate the information from a hpix structure in a JAX compatible way
    This class can be converted into a pytree
    and has an efficient hash that lets it be cached
    """

    nside: np.int64
    npix: np.int64
    ncap: np.int64
    dnside: np.double
    twonside: np.int64
    fournside: np.int64
    nsideplusone: np.int64
    nsideminusone: np.int64
    halfnside: np.double
    tqnside: np.double
    factor: np.int64
    jr: np.array
    jq: np.array
    utab: np.array
    ctab: np.array

    def __init__(self, nside):
        self.nside = nside
        self.ncap = 2 * (nside * nside - nside)
        self.npix = 12 * nside * nside
        self.dnside = float(nside)
        self.twonside = 2 * nside
        self.fournside = 4 * nside
        self.nsideplusone = nside + 1
        self.halfnside = 0.5 * (self.dnside)
        self.tqnside = 0.75 * (self.dnside)
        self.nsideminusone = nside - 1

        self.factor = 0
        while nside != (1 << self.factor):
            self.factor += 1

        self.jr = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        self.jq = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])

        m = np.arange(stop=0x100)
        self.utab = (
            (m & 0x1)
            | ((m & 0x2) << 1)
            | ((m & 0x4) << 2)
            | ((m & 0x8) << 3)
            | ((m & 0x10) << 4)
            | ((m & 0x20) << 5)
            | ((m & 0x40) << 6)
            | ((m & 0x80) << 7)
        )
        self.ctab = (
            (m & 0x1)
            | ((m & 0x2) << 7)
            | ((m & 0x4) >> 1)
            | ((m & 0x8) << 6)
            | ((m & 0x10) >> 2)
            | ((m & 0x20) << 5)
            | ((m & 0x40) >> 3)
            | ((m & 0x80) << 4)
        )

    def xy2pix(self, x, y):
        return (
            self.utab[x & 0xFF]
            | (self.utab[(x >> 8) & 0xFF] << 16)
            | (self.utab[(x >> 16) & 0xFF] << 32)
            | (self.utab[(x >> 24) & 0xFF] << 48)
            | (self.utab[y & 0xFF] << 1)
            | (self.utab[(y >> 8) & 0xFF] << 17)
            | (self.utab[(y >> 16) & 0xFF] << 33)
            | (self.utab[(y >> 24) & 0xFF] << 49)
        )


def zphi2nest(hpix, phi, region, z, rtz):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (double)
        region (int)
        z (double)
        rtz (double)

    Returns:
        pix (int)
    """
    if np.abs(phi) < MACHINE_EPSILON:
        phi = 0.0

    tt = phi * TWOINVPI
    if phi < 0.0:
        tt += 4.0

    if np.abs(region) == 1:
        temp1 = hpix.halfnside + hpix.dnside * tt
        temp2 = hpix.tqnside * z

        jp = np.int64(temp1 - temp2)
        jm = np.int64(temp1 + temp2)

        ifp = jp >> hpix.factor
        ifm = jm >> hpix.factor

        if ifp == ifm:
            face = 4 if ifp == 4 else ifp + 4
        elif ifp < ifm:
            face = ifp
        else:
            face = ifm + 8

        x = jm & hpix.nsideminusone
        y = hpix.nsideminusone - (jp & hpix.nsideminusone)
    else:
        ntt = np.int64(tt)

        tp = tt - np.double(ntt)

        temp1 = hpix.dnside * rtz

        jp = np.int64(tp * temp1)
        jm = np.int64((1.0 - tp) * temp1)

        if jp >= hpix.nside:
            jp = hpix.nsideminusone

        if jm >= hpix.nside:
            jm = hpix.nsideminusone

        if z >= 0:
            face = ntt
            x = hpix.nsideminusone - jm
            y = hpix.nsideminusone - jp
        else:
            face = ntt + 8
            x = jp
            y = jm

    sipf = hpix.xy2pix(np.int64(x), np.int64(y))
    pix = np.int64(sipf) + (face << (2 * hpix.factor))
    return pix


def zphi2ring(hpix, phi, region, z, rtz, pix):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (double)
        region (int)
        z (double)
        rtz (double)

    Returns:
        pix (int)
    """
    if np.abs(phi) < MACHINE_EPSILON:
        phi = 0.0

    tt = phi * TWOINVPI
    if phi < 0.0:
        tt += 4.0

    if np.abs(region) == 1:
        temp1 = hpix.halfnside + hpix.dnside * tt
        temp2 = hpix.tqnside * z

        jp = np.int64(temp1 - temp2)
        jm = np.int64(temp1 + temp2)

        ir = hpix.nsideplusone + jp - jm
        kshift = 1 - (ir & 1)

        ip = (jp + jm - hpix.nside + kshift + 1) >> 1
        ip = ip % hpix.fournside

        pix = hpix.ncap + ((ir - 1) * hpix.fournside + ip)
    else:
        tp = tt - np.floor(tt)

        temp1 = hpix.dnside * rtz

        jp = np.int64(tp * temp1)
        jm = np.int64((1.0 - tp) * temp1)
        ir = jp + jm + 1
        ip = np.int64(tt * np.double(ir))
        longpart = np.int64(ip / (4 * ir))
        ip -= longpart

        pix = (
            (2 * ir * (ir - 1) + ip)
            if (region > 0)
            else (hpix.npix - 2 * ir * (ir + 1) + ip)
        )

    return pix


def vec2zphi(vec):
    """
    Args:
        vec (array, double) of size 3

    Returns:
        (phi, region, z, rtz)
        phi (double)
        region (int)
        z (double)
        rtz (double)
    """
    z = vec[2]
    za = np.abs(z)

    # region encodes BOTH the sign of Z and whether its
    # absolute value is greater than 2/3.
    itemps = 1 if (z > 0.0) else -1
    region = itemps if (za <= 2.0 / 3.0) else 2 * itemps

    work1 = 3.0 * (1.0 - za)
    rtz = np.sqrt(work1)
    phi = np.arctan2(vec[1], vec[0])

    return (phi, region, z, rtz)
