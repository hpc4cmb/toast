# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ._libtoast import (
    healpix_ang2nest,
    healpix_ang2ring,
    healpix_ang2vec,
    healpix_degrade_nest,
    healpix_degrade_ring,
    healpix_nest2ring,
    healpix_ring2nest,
    healpix_upgrade_nest,
    healpix_upgrade_ring,
    healpix_vec2ang,
    healpix_vec2nest,
    healpix_vec2ring,
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
    vec_array = vec.array()
    healpix_ang2vec(intheta, inphi, vec_array.reshape((nphi, 3)))
    if len(vec) == 3:
        if (object_ndim(theta) == 1) or (object_ndim(phi) == 1):
            return vec_array.reshape(1, 3)
        else:
            return vec_array
    else:
        return vec_array.reshape((-1, 3))


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
    theta_array = theta.array()
    phi = AlignedF64(n)
    phi_array = phi.array()
    healpix_vec2ang(invec.reshape((n, 3)), theta_array, phi_array)
    if len(vec) == 3:
        if object_ndim(vec) == 2:
            return (theta_array, phi_array)
        else:
            return (theta_array[0], phi_array[0])
    else:
        return (theta_array, phi_array)


def ang2nest(nside, theta, phi):
    """Convert spherical coordinates to pixels in NESTED ordering.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        nside (int): The NSIDE of the projection.
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
    pix_array = pix.array()
    healpix_ang2nest(nside, intheta, inphi, pix_array)
    if nphi == 1:
        if (object_ndim(theta) == 1) or (object_ndim(phi) == 1):
            return pix_array
        else:
            return pix_array[0]
    else:
        return pix_array


def ang2ring(nside, theta, phi):
    """Convert spherical coordinates to pixels in RING ordering.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        nside (int): The NSIDE of the projection.
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
    pix_array = pix.array()
    healpix_ang2ring(nside, intheta, inphi, pix_array)
    if nphi == 1:
        if (object_ndim(theta) == 1) or (object_ndim(phi) == 1):
            return pix_array
        else:
            return pix_array[0]
    else:
        return pix_array


def vec2nest(nside, vec):
    """Convert unit vectors to pixels in NESTED ordering.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        nside (int): The NSIDE of the projection.
        vec (array_like): Input packed unit vectors.

    Returns:
        (array): Output pixel indices.

    """
    invec = ensure_buffer_f64(vec)
    n = len(invec) // 3
    pix = AlignedI64(n)
    pix_array = pix.array()
    healpix_vec2nest(nside, invec.reshape((n, 3)), pix)
    if len(vec) == 3:
        if object_ndim(vec) == 2:
            return pix_array
        else:
            return pix_array[0]
    else:
        return pix_array


def vec2ring(nside, vec):
    """Convert unit vectors to pixels in RING ordering.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        nside (int): The NSIDE of the projection.
        vec (array_like): Input packed unit vectors.

    Returns:
        (array): Output pixel indices.

    """
    invec = ensure_buffer_f64(vec)
    n = len(invec) // 3
    pix = AlignedI64(n)
    pix_array = pix.array()
    healpix_vec2ring(nside, invec.reshape((n, 3)), pix_array)
    if len(vec) == 3:
        if object_ndim(vec) == 2:
            return pix_array
        else:
            return pix_array[0]
    else:
        return pix_array


def ring2nest(nside, ringpix):
    """Convert RING ordered pixel numbers into NESTED ordering.

    Args:
        nside (int): The NSIDE of the projection.
        in (array_like): Input pixel indices.

    Returns:
        (array): Output pixel indices.

    """
    inpix = ensure_buffer_i64(ringpix)
    n = len(inpix)
    out = AlignedI64(n)
    out_array = out.array()
    healpix_ring2nest(nside, inpix, out_array)
    if n == 1:
        if object_ndim(ringpix) == 1:
            return out_array
        else:
            return out_array[0]
    else:
        return out_array


def nest2ring(nside, nestpix):
    """Convert NESTED ordered pixel numbers into RING ordering.

    Args:
        nside (int): The NSIDE of the projection.
        in (array_like): Input pixel indices.

    Returns:
        (array): Output pixel indices.

    """
    inpix = ensure_buffer_i64(nestpix)
    n = len(inpix)
    out = AlignedI64(n)
    out_array = out.array()
    healpix_nest2ring(nside, inpix, out_array)
    if n == 1:
        if object_ndim(nestpix) == 1:
            return out_array
        else:
            return out_array[0]
    else:
        return out_array


def degrade_ring(nside, factor, inpix):
    """Degrade RING ordered pixel numbers.

    Each 'factor' is one division by two in the NSIDE resolution.  So
    a factor of '3' would divide the NSIDE value by 8.

    Args:
        nside (int): The NSIDE of the input projection.
        factor (int):  The degrade factor.
        in (array_like): Input pixel indices.

    Returns:
        (array): Output pixel indices.

    """
    if nside < 2**factor:
        msg = f"Cannot degrade NSIDE {nside} pixels by {factor} levels"
        raise RuntimeError(msg)
    inp = ensure_buffer_i64(inpix)
    n = len(inp)
    out = AlignedI64(n)
    out_array = out.array()
    healpix_degrade_ring(nside, factor, inp, out_array)
    if n == 1:
        if object_ndim(inpix) == 1:
            return out_array
        else:
            return out_array[0]
    else:
        return out_array


def degrade_nest(nside, factor, inpix):
    """Degrade NESTED ordered pixel numbers.

    Each 'factor' is one division by two in the NSIDE resolution.  So
    a factor of '3' would divide the NSIDE value by 8.

    Args:
        nside (int): The NSIDE of the input projection.
        factor (int):  The degrade factor.
        in (array_like): Input pixel indices.

    Returns:
        (array): Output pixel indices.

    """
    if nside < 2**factor:
        msg = f"Cannot degrade NSIDE {nside} pixels by {factor} levels"
        raise RuntimeError(msg)
    inp = ensure_buffer_i64(inpix)
    n = len(inp)
    out = AlignedI64(n)
    out_array = out.array()
    healpix_degrade_nest(nside, factor, inp, out_array)
    if n == 1:
        if object_ndim(inpix) == 1:
            return out_array
        else:
            return out_array[0]
    else:
        return out_array


def upgrade_ring(nside, factor, inpix):
    """Upgrade RING ordered pixel numbers.

    Each 'factor' is one multiplication by two in the NSIDE
    resolution.  So a factor of '3' would multiply the NSIDE value
    by 8.

    Args:
        nside (int): The NSIDE of the input projection.
        factor (int):  The upgrade factor.
        inpix (array_like): Input pixel indices.

    Returns:
        (array): Output pixel indices.

    """
    inp = ensure_buffer_i64(inpix)
    n = len(inp)
    out = AlignedI64(n)
    out_array = out.array()
    healpix_upgrade_ring(nside, factor, inp, out_array)
    if n == 1:
        if object_ndim(inpix) == 1:
            return out_array
        else:
            return out_array[0]
    else:
        return out_array


def upgrade_nest(nside, factor, inpix):
    """Upgrade RING ordered pixel numbers.

    Each 'factor' is one multiplication by two in the NSIDE
    resolution.  So a factor of '3' would multiply the NSIDE value
    by 8.

    Args:
        nside (int): The NSIDE of the input projection.
        factor (int):  The upgrade factor.
        inpix (array_like): Input pixel indices.

    Returns:
        (array): Output pixel indices.

    """
    inp = ensure_buffer_i64(inpix)
    n = len(inp)
    out = AlignedI64(n)
    out_array = out.array()
    healpix_upgrade_nest(nside, factor, inp, out_array)
    if n == 1:
        if object_ndim(inpix) == 1:
            return out_array
        else:
            return out_array[0]
    else:
        return out_array
