# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# This file provides a simplified interface to quaternion operations.

import numpy as np

from .utils import Logger, AlignedF64, ensure_buffer_f64, object_ndim

from ._libtoast import (
    qa_inv,
    qa_amplitude,
    qa_normalize,
    qa_rotate,
    qa_mult,
    qa_slerp,
    qa_exp,
    qa_ln,
    qa_pow,
    qa_from_axisangle,
    qa_to_axisangle,
    qa_to_rotmat,
    qa_from_rotmat,
    qa_from_vectors,
    qa_from_angles,
    qa_to_angles,
    qa_to_position,
    qa_from_position,
)


def inv(q):
    """Invert a quaternion array.

    Args:
        q (array_like):  The quaternion array to invert.

    Returns:
        (array):  The inverse.

    """
    qin = ensure_buffer_f64(q)
    out = AlignedF64(len(qin))
    out[:] = qin
    qa_inv(out)
    if len(out) == 4:
        if object_ndim(q) == 2:
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def amplitude(q):
    """Amplitude of a quaternion array

    Args:
        q (array_like):  The quaternion array.

    Returns:
        (array):  The array of amplitudes.

    """
    qin = ensure_buffer_f64(q)
    lamp = len(qin) // 4
    amp = AlignedF64(lamp)
    qa_amplitude(qin, amp)
    if len(amp) == 1:
        if object_ndim(q) == 2:
            return amp.array()
        else:
            return float(amp[0])
    else:
        return amp.array()


def norm(q):
    """Normalize quaternion array.

    Args:
        q (array_like):  The quaternion array.

    Returns:
        (array):  The normalized array.

    """
    qin = ensure_buffer_f64(q)
    lq = len(qin)
    nrm = AlignedF64(lq)
    qa_normalize(qin, nrm)
    if len(qin) == 4:
        if object_ndim(q) == 2:
            return nrm.array().reshape((1, 4))
        else:
            return nrm.array()
    else:
        return nrm.array().reshape((-1, 4))


def rotate(q, v):
    """Rotate vectors with quaternions.

    The number of quaternions and vectors should either be equal or one of
    the arrays should be a single quaternion or vector.

    Args:
        q (array_like):  The quaternion array.
        v (array_like):  The vector array.

    Returns:
        (array):  The rotated vectors.

    """
    qin = ensure_buffer_f64(q)
    nq = len(qin) // 4
    vin = ensure_buffer_f64(v)
    nv = len(vin) // 3
    nout = None
    if nq > nv:
        nout = nq
    else:
        nout = nv
    out = AlignedF64(3 * nout)
    qa_rotate(qin, vin, out)
    if len(out) == 3:
        if (object_ndim(q) == 2) or (object_ndim(v) == 2):
            return out.array().reshape(1, 3)
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 3))


def mult(p, q):
    """Multiply arrays of quaternions.

    The number of quaternions in the input arrays should either be equal or
    one.

    Args:
        p (array_like):  The first quaternion array.
        q (array_like):  The second quaternion array.

    Returns:
        (array):  The product.

    """
    pin = ensure_buffer_f64(p)
    qin = ensure_buffer_f64(q)
    out = None
    if len(pin) > len(qin):
        out = AlignedF64(len(pin))
    else:
        out = AlignedF64(len(qin))
    qa_mult(pin, qin, out)
    if len(out) == 4:
        if (object_ndim(p) == 2) or (object_ndim(q) == 2):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def slerp(targettime, time, q):
    """Spherical Linear Interpolation (SLERP) of a quaternion array.

    The input quaternions are specified at time stamps, and the output is
    interpolated to the target times.

    Args:
        targettime (array_like):  The output target times.
        time (array_like):  The input times.
        q (array_like):  The quaternion array.

    Returns:
        (array):  The interpolated quaternions.

    """
    tgt = ensure_buffer_f64(targettime)
    t = ensure_buffer_f64(time)
    qin = ensure_buffer_f64(q)
    log = Logger.get()
    if len(t) < 2:
        msg = "SLERP input times must have at least two values"
        log.error(msg)
        raise RuntimeError(msg)
    out = AlignedF64(4 * len(tgt))
    qa_slerp(t, tgt, qin, out)
    if len(out) == 4:
        if object_ndim(targettime) == 1:
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def exp(q):
    """Exponential of a quaternion array.

    Args:
        q (array_like):  The quaternion array.

    Returns:
        (array):  The result.

    """
    qin = ensure_buffer_f64(q)
    out = AlignedF64(len(qin))
    qa_exp(qin, out)
    if len(out) == 4:
        if object_ndim(q) == 2:
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def ln(q):
    """Natural logarithm of a quaternion array.

    Args:
        q (array_like):  The quaternion array.

    Returns:
        (array):  The result.

    """
    qin = ensure_buffer_f64(q)
    out = AlignedF64(len(qin))
    qa_ln(qin, out)
    if len(out) == 4:
        if object_ndim(q) == 2:
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def pow(q, pw):
    """Real power of a quaternion array.

    Args:
        q (array_like):  The quaternion array.
        pw (array_like):  The power.

    Returns:
        (array):  The result.

    """
    qin = ensure_buffer_f64(q)
    pwin = ensure_buffer_f64(pw)
    out = AlignedF64(len(qin))
    qa_pow(qin, pwin, out)
    if len(out) == 4:
        if (object_ndim(q) == 2) or (object_ndim(pw) == 1):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def rotation(axis, angle):
    """Create quaternions from axis / angle information.

    Args:
        axis (array_like):  The array of normalized axis vectors.
        angle (array_like):  The array of angles (in radians).

    Returns:
        (array):  The result.

    """
    axin = ensure_buffer_f64(axis)
    angin = ensure_buffer_f64(angle)
    out = AlignedF64(4 * len(angin))
    qa_from_axisangle(axin, angin, out)
    if len(out) == 4:
        if (object_ndim(axis) == 2) or (object_ndim(angle) == 1):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def to_axisangle(q):
    """Convert quaterions to axis / angle form.

    Args:
        q (array_like):  The input quaternions.

    Returns:
        (tuple):  The (axis, angle) results.

    """
    qin = ensure_buffer_f64(q)
    lq = len(qin) // 4
    ax = AlignedF64(3 * lq)
    ang = AlignedF64(lq)
    qa_to_axisangle(qin, ax, ang)
    if len(ax) == 3:
        if object_ndim(q) == 2:
            return (ax.array().reshape((1, 3)), ang.array())
        else:
            return (ax.array(), float(ang[0]))
    else:
        return (ax.array().reshape((-1, 3)), ang.array())


def to_rotmat(q):
    """Convert quaternions to rotation matrices.

    Args:
        q (array_like):  The input quaternions.

    Returns:
        (array):  The rotation matrices.

    """
    qin = ensure_buffer_f64(q)
    lq = len(qin) // 4
    out = AlignedF64(9 * lq)
    qa_to_rotmat(qin, out)
    if len(out) == 9:
        if object_ndim(q) == 2:
            return out.array().reshape((1, 3, 3))
        else:
            return out.array().reshape((3, 3))
    else:
        return out.array().reshape((-1, 3, 3))


def from_rotmat(rotmat):
    """Create quaternions from rotation matrices.

    Args:
        rotmat (array_like):  The input 3x3 rotation matrices.

    Returns:
        (array):  The quaternions.

    """
    rot = ensure_buffer_f64(rotmat)
    lr = len(rot) // 9
    out = AlignedF64(4 * lr)
    qa_from_rotmat(rot, out)
    if len(out) == 4:
        if object_ndim(rotmat) == 3:
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def from_vectors(v1, v2):
    """Create quaternions from pairs of vectors.

    Args:
        v1 (array_like):  The input starting vectors.
        v2 (array_like):  The input ending vectors.

    Returns:
        (array):  The quaternions.

    """
    v1in = ensure_buffer_f64(v1)
    v2in = ensure_buffer_f64(v2)
    lv = len(v1in) // 3
    out = AlignedF64(4 * lv)
    qa_from_vectors(v1in, v2in, out)
    if len(out) == 4:
        if (object_ndim(v1) == 2) or (object_ndim(v2) == 2):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def from_angles(theta, phi, pa, IAU=False):
    """Create quaternions from spherical coordinates.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.  The position angle is with respect
    to the local meridian at the point described by the theta / phi
    coordinates.

    Args:
        theta (array_like):  The input theta angles.
        phi (array_like):  The input phi vectors.
        pa (array_like):  The input position angle vectors.
        IAU (bool):  If True, use IAU convention.

    Returns:
        (array):  The quaternions.

    """
    thetain = ensure_buffer_f64(theta)
    phiin = ensure_buffer_f64(phi)
    pain = ensure_buffer_f64(pa)
    lt = len(thetain)
    out = AlignedF64(4 * lt)
    qa_from_angles(thetain, phiin, pain, out, IAU)
    if len(out) == 4:
        if (
            (object_ndim(theta) == 1)
            or (object_ndim(phi) == 1)
            or (object_ndim(pa) == 1)
        ):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def to_angles(q, IAU=False):
    """Convert quaternions to spherical coordinates and position angle.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.  The position angle is with respect
    to the local meridian at the point described by the theta / phi
    coordinates.

    Args:
        q (array_like):  The input quaternions.
        IAU (bool):  If True, use IAU convention.

    Returns:
        (tuple):  The (theta, phi, pa) arrays.

    """
    qin = ensure_buffer_f64(q)
    lq = len(qin) // 4
    theta = AlignedF64(lq)
    phi = AlignedF64(lq)
    pa = AlignedF64(lq)
    qa_to_angles(qin, theta, phi, pa, IAU)
    if len(qin) == 4:
        if object_ndim(q) == 2:
            return (theta.array(), phi.array(), pa.array())
        else:
            return (float(theta[0]), float(phi[0]), float(pa[0]))
    return (theta.array(), phi.array(), pa.array())


def from_position(theta, phi):
    """Create quaternions from spherical coordinates.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        theta (array_like):  The input theta angles.
        phi (array_like):  The input phi vectors.

    Returns:
        (array):  The quaternions.

    """
    thetain = ensure_buffer_f64(theta)
    phiin = ensure_buffer_f64(phi)
    lt = len(thetain)
    out = AlignedF64(4 * lt)
    qa_from_position(thetain, phiin, out)
    if len(out) == 4:
        if (object_ndim(theta) == 1) or (object_ndim(phi) == 1):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def to_position(q):
    """Convert quaternions to spherical coordinates.

    The theta angle is measured down from the North pole and phi is
    measured from the prime meridian.

    Args:
        q (array_like):  The input quaternions.

    Returns:
        (tuple):  The (theta, phi) arrays.

    """
    qin = ensure_buffer_f64(q)
    lq = len(qin) // 4
    theta = AlignedF64(lq)
    phi = AlignedF64(lq)
    qa_to_position(qin, theta, phi)
    if len(qin) == 4:
        if object_ndim(q) == 2:
            return (theta.array(), phi.array())
        else:
            return (float(theta[0]), float(phi[0]))
    return (theta.array(), phi.array())
