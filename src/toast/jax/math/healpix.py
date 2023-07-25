# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp
import numpy as np

TWOINVPI = 0.63661977236758134308
MACHINE_EPSILON = np.finfo(np.float64).eps


class HPIX_JAX:
    """
    JAX compatible HPIX structure
    This class has an efficient hash that lets it be cached
    """

    def __init__(self, nside):
        self.nside = nside
        self.ncap = 2 * (nside * nside - nside)
        self.npix = 12 * nside * nside
        self.dnside = float(nside)
        self.twonside = 2 * nside
        self.fournside = 4 * nside
        self.nsideplusone = nside + 1
        self.nsideminusone = nside - 1
        self.halfnside = 0.5 * self.dnside
        self.tqnside = 0.75 * self.dnside

        factor = 0
        while nside != (1 << factor):
            factor += 1
        self.factor = factor

        self.jr = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        self.jq = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])

        m = np.arange(start=0, stop=0x100)
        self.utab = jnp.array(
            (m & 0x1)
            | ((m & 0x2) << 1)
            | ((m & 0x4) << 2)
            | ((m & 0x8) << 3)
            | ((m & 0x10) << 4)
            | ((m & 0x20) << 5)
            | ((m & 0x40) << 6)
            | ((m & 0x80) << 7)
        )
        self.ctab = jnp.array(
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

    def __key(self):
        return self.nside

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, HPIX_JAX):
            return self.__key() == other.__key()
        return NotImplemented


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
    phi = jnp.where(jnp.abs(phi) < MACHINE_EPSILON, 0.0, phi)
    tt = jnp.where(phi >= 0.0, phi * TWOINVPI, phi * TWOINVPI + 4.0)

    # Then branch
    temp1 = hpix.halfnside + hpix.dnside * tt
    temp2 = hpix.tqnside * z

    jp = jnp.int64(temp1 - temp2)
    jm = jnp.int64(temp1 + temp2)

    ifp = jnp.right_shift(jp, hpix.factor)
    ifm = jnp.right_shift(jm, hpix.factor)

    face = jnp.where(
        ifp == ifm, jnp.where(ifp == 4, 4, ifp + 4), jnp.where(ifp < ifm, ifp, ifm + 8)
    )
    x = jm & hpix.nsideminusone
    y = hpix.nsideminusone - (jp & hpix.nsideminusone)

    # Else branch
    ntt = jnp.floor(tt)
    tp = tt - ntt
    temp1 = hpix.dnside * rtz

    jp = jnp.int64(tp * temp1)
    jm = jnp.int64((1.0 - tp) * temp1)

    jp = jnp.where(jp >= hpix.nside, hpix.nsideminusone, jp)
    jm = jnp.where(jm >= hpix.nside, hpix.nsideminusone, jm)

    face_else = jnp.where(z >= 0, ntt, ntt + 8)
    x_else = jnp.where(z >= 0, hpix.nsideminusone - jm, jp)
    y_else = jnp.where(z >= 0, hpix.nsideminusone - jp, jm)

    # Test
    if_cond = jnp.abs(region) == 1
    face = jnp.where(if_cond, face, face_else)
    x = jnp.where(if_cond, x, x_else)
    y = jnp.where(if_cond, y, y_else)

    sipf = hpix.xy2pix(x, y)
    return jnp.int64(sipf) + (jnp.int64(face) << (2 * hpix.factor))


def zphi2ring(hpix, phi, region, z, rtz):
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
    phi = jnp.where(jnp.abs(phi) < MACHINE_EPSILON, 0.0, phi)
    tt = phi * TWOINVPI
    tt = jnp.where(phi < 0.0, tt + 4.0, tt)

    # then branch
    def then_branch(tt, z):
        temp1 = hpix.halfnside + hpix.dnside * tt
        temp2 = hpix.tqnside * z

        jp = jnp.int64(temp1 - temp2)
        jm = jnp.int64(temp1 + temp2)

        ir = hpix.nsideplusone + jp - jm
        kshift = 1 - (ir & 1)

        ip = (jp + jm - hpix.nside + kshift + 1) >> 1
        ip = ip % hpix.fournside

        pix = hpix.ncap + ((ir - 1) * hpix.fournside + ip)
        return pix

    pix_then = then_branch(tt, z)

    # else branch
    def else_branch(tt, rtz):
        tp = tt - jnp.floor(tt)

        temp1 = hpix.dnside * rtz

        jp = jnp.int64(tp * temp1)
        jm = jnp.int64((1.0 - tp) * temp1)
        ir = jp + jm + 1
        ip = jnp.int64(tt * jnp.double(ir))
        longpart = jnp.int64(ip / (4 * ir))
        ip = ip - longpart

        pix_pos = 2 * ir * (ir - 1) + ip
        pix_neg = hpix.npix - 2 * ir * (ir + 1) + ip
        pix = jnp.where(region > 0, pix_pos, pix_neg)
        return pix

    pix_else = else_branch(tt, rtz)

    # test
    pix = jnp.where(jnp.abs(region) == 1, pix_then, pix_else)
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
    za = jnp.abs(z)

    # region encodes BOTH the sign of Z and whether its
    # absolute value is greater than 2/3.
    itemps = jnp.where(z > 0.0, 1, -1)
    region = jnp.where(za <= 2.0 / 3.0, itemps, 2 * itemps)

    tz = 3.0 * (1.0 - za)
    rtz = jnp.sqrt(tz)
    phi = jnp.arctan2(vec[1], vec[0])

    return (phi, region, z, rtz)
