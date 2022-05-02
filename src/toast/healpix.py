# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ._libtoast import (
    HealpixPixels,
    healpix_ang2vec,
    healpix_vec2ang,
    healpix_vecs2angpa,
)
from .timing import function_timer
from .utils import (
    AlignedF64,
    AlignedI64,
    Logger,
    ensure_buffer_f64,
    ensure_buffer_i64,
    object_ndim,
)


def ang2vec(theta, phi):
    """Convert spherical coordinates to a unit vector.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        theta (array_like): The spherical coordinate theta angles in
            radians.
        phi (array like): The spherical coordinate phi angles in radians.

    Returns:
        (array):  The array of output vectors.

    """
    intheta = ensure_buffer_f64(theta)
    inphi = ensure_buffer_f64(phi)
    ntheta = len(intheta)
    nphi = len(inphi)
    if ntheta != nphi:
        raise RuntimeError("theta / phi vectors must have the same length")
    vec = AlignedF64(3 * nphi)
    healpix_ang2vec(intheta, inphi, vec)
    if len(vec) == 3:
        if (object_ndim(theta) == 1) or (object_ndim(phi) == 1):
            return vec.array().reshape(1, 3)
        else:
            return vec.array()
    else:
        return vec.array().reshape((-1, 3))


def vec2ang(vec):
    """Convert unit vectors to spherical coordinates.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        vec (array_like):  The array of input vectors.

    Returns:
        (tuple): The (theta, phi) arrays in radians.

    """
    invec = ensure_buffer_f64(vec)
    n = len(invec) // 3
    theta = AlignedF64(n)
    phi = AlignedF64(n)
    healpix_vec2ang(invec, theta, phi)
    if len(vec) == 3:
        if object_ndim(vec) == 2:
            return (theta.array(), phi.array())
        else:
            return (theta[0], phi[0])
    else:
        return (theta.array(), phi.array())


def vecs2angpa(vec):
    """Convert direction / orientation unit vectors.

    The inputs are flat-packed pairs of direction and orientation unit
    vectors (6 float64 values total per sample).  The outputs are the
    theta, phi, and position angle of the location on the sphere.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.  The position angle is with respect
    to the local meridian at the point described by the theta / phi
    coordinates.

    Args:
        vec (array_like):  The array of packed input direction and
            orientation vectors.

    Returns:
        (tuple): The (theta, phi, pa) arrays in radians.

    """
    invec = ensure_buffer_f64(vec)
    n = len(invec) // 6
    theta = AlignedF64(n)
    phi = AlignedF64(n)
    pa = AlignedF64(n)
    healpix_vecs2angpa(invec, theta, phi, pa)
    if len(vec) == 6:
        if object_ndim(vec) == 2:
            return (theta.array(), phi.array(), pa.array())
        else:
            return (theta[0], phi[0], pa[0])
    else:
        return (theta.array(), phi.array(), pa.array())


class Pixels(object):
    """Class for HEALPix pixel operations at a fixed NSIDE.

    This provides a very thin wrapper around the internal HealpixPixels class.
    This wrapper mainly exists to provide methods that return a newly created
    output buffer (rather than requiring the user to pass in the buffer).  If
    you are happy pre-creating the output buffer then you can call methods
    of HealpixPixels directly.

    Args:
        nside (int): the map NSIDE.

    """

    def __init__(self, nside=1):
        self.hpix = HealpixPixels(nside)

    def __del__(self):
        if self.hpix is not None:
            del self.hpix

    def reset(self, nside):
        """Reset the class instance to use a new NSIDE value.

        Args:
            nside (int): the map NSIDE.

        Returns:
            None

        """
        self.hpix.reset(nside)
        return

    def ang2nest(self, theta, phi):
        """Convert spherical coordinates to pixels in NESTED ordering.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        Args:
            theta (array_like): Input spherical coordinate theta angles in
                radians.
            phi (array like): Input spherical coordinate phi angles in
                radians.

        Returns:
            (array): Output pixel indices.

        """
        intheta = ensure_buffer_f64(theta)
        inphi = ensure_buffer_f64(phi)
        ntheta = len(intheta)
        nphi = len(inphi)
        if ntheta != nphi:
            raise RuntimeError("theta / phi vectors must have the same length")
        pix = AlignedI64(nphi)
        self.hpix.ang2nest(intheta, inphi, pix)
        if nphi == 1:
            if (object_ndim(theta) == 1) or (object_ndim(phi) == 1):
                return pix.array()
            else:
                return pix[0]
        else:
            return pix.array()

    def ang2ring(self, theta, phi):
        """Convert spherical coordinates to pixels in RING ordering.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        Args:
            theta (array_like): Input spherical coordinate theta angles in
                radians.
            phi (array like): Input spherical coordinate phi angles in
                radians.

        Returns:
            (array): Output pixel indices.

        """
        intheta = ensure_buffer_f64(theta)
        inphi = ensure_buffer_f64(phi)
        ntheta = len(intheta)
        nphi = len(inphi)
        if ntheta != nphi:
            raise RuntimeError("theta / phi vectors must have the same length")
        pix = AlignedI64(nphi)
        self.hpix.ang2ring(intheta, inphi, pix)
        if nphi == 1:
            if (object_ndim(theta) == 1) or (object_ndim(phi) == 1):
                return pix.array()
            else:
                return pix[0]
        else:
            return pix.array()

    def vec2nest(self, vec):
        """Convert unit vectors to pixels in NESTED ordering.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        Args:
            vec (array_like): Input packed unit vectors.

        Returns:
            (array): Output pixel indices.

        """
        invec = ensure_buffer_f64(vec)
        n = len(invec) // 3
        pix = AlignedI64(n)
        self.hpix.vec2nest(invec, pix)
        if len(vec) == 3:
            if object_ndim(vec) == 2:
                return pix.array()
            else:
                return pix[0]
        else:
            return pix.array()

    def vec2ring(self, vec):
        """Convert unit vectors to pixels in RING ordering.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        Args:
            vec (array_like): Input packed unit vectors.

        Returns:
            (array): Output pixel indices.

        """
        invec = ensure_buffer_f64(vec)
        n = len(invec) // 3
        pix = AlignedI64(n)
        self.hpix.vec2ring(invec, pix)
        if len(vec) == 3:
            if object_ndim(vec) == 2:
                return pix.array()
            else:
                return pix[0]
        else:
            return pix.array()

    def ring2nest(self, ringpix):
        """Convert RING ordered pixel numbers into NESTED ordering.

        Args:
            in (array_like): Input pixel indices.

        Returns:
            (array): Output pixel indices.

        """
        inpix = ensure_buffer_i64(ringpix)
        n = len(inpix)
        out = AlignedI64(n)
        self.hpix.ring2nest(inpix, out)
        if n == 1:
            if object_ndim(ringpix) == 1:
                return out.array()
            else:
                return out[0]
        else:
            return out.array()

    def nest2ring(self, nestpix):
        """Convert NESTED ordered pixel numbers into RING ordering.

        Args:
            in (array_like): Input pixel indices.

        Returns:
            (array): Output pixel indices.

        """
        inpix = ensure_buffer_i64(nestpix)
        n = len(inpix)
        out = AlignedI64(n)
        self.hpix.ring2nest(inpix, out)
        if n == 1:
            if object_ndim(nestpix) == 1:
                return out.array()
            else:
                return out[0]
        else:
            return out.array()

    def degrade_ring(self, factor, inpix):
        """Degrade RING ordered pixel numbers.

        Each 'factor' is one division by two in the NSIDE resolution.  So
        a factor of '3' would divide the NSIDE value by 8.

        Args:
            factor (int):  The degrade factor.
            in (array_like): Input pixel indices.

        Returns:
            (array): Output pixel indices.

        """
        inp = ensure_buffer_i64(inpix)
        n = len(inp)
        out = AlignedI64(n)
        self.hpix.degrade_ring(factor, inp, out)
        if n == 1:
            if object_ndim(inpix) == 1:
                return out.array()
            else:
                return out[0]
        else:
            return out.array()

    def degrade_nest(self, factor, inpix):
        """Degrade NESTED ordered pixel numbers.

        Each 'factor' is one division by two in the NSIDE resolution.  So
        a factor of '3' would divide the NSIDE value by 8.

        Args:
            factor (int):  The degrade factor.
            in (array_like): Input pixel indices.

        Returns:
            (array): Output pixel indices.

        """
        inp = ensure_buffer_i64(inpix)
        n = len(inp)
        out = AlignedI64(n)
        self.hpix.degrade_nest(factor, inp, out)
        if n == 1:
            if object_ndim(inpix) == 1:
                return out.array()
            else:
                return out[0]
        else:
            return out.array()

    def upgrade_ring(self, factor, inpix):
        """Upgrade RING ordered pixel numbers.

        Each 'factor' is one multiplication by two in the NSIDE
        resolution.  So a factor of '3' would multiply the NSIDE value
        by 8.

        Args:
            factor (int):  The upgrade factor.
            inpix (array_like): Input pixel indices.

        Returns:
            (array): Output pixel indices.

        """
        inp = ensure_buffer_i64(inpix)
        n = len(inp)
        out = AlignedI64(n)
        self.hpix.upgrade_ring(factor, inp, out)
        if n == 1:
            if object_ndim(inpix) == 1:
                return out.array()
            else:
                return out[0]
        else:
            return out.array()

    def upgrade_nest(self, factor, inpix):
        """Upgrade RING ordered pixel numbers.

        Each 'factor' is one multiplication by two in the NSIDE
        resolution.  So a factor of '3' would multiply the NSIDE value
        by 8.

        Args:
            factor (int):  The upgrade factor.
            inpix (array_like): Input pixel indices.

        Returns:
            (array): Output pixel indices.

        """
        inp = ensure_buffer_i64(inpix)
        n = len(inp)
        out = AlignedI64(n)
        self.hpix.upgrade_nest(factor, inp, out)
        if n == 1:
            if object_ndim(inpix) == 1:
                return out.array()
            else:
                return out[0]
        else:
            return out.array()
