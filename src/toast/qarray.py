# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# This file provides a simplified interface to quaternion operations.

import numpy as np

from ._libtoast import (
    qa_amplitude,
    qa_exp,
    qa_from_angles,
    qa_from_axisangle,
    qa_from_iso,
    qa_from_position,
    qa_from_rotmat,
    qa_from_vectors,
    qa_inv,
    qa_ln,
    qa_mult,
    qa_normalize,
    qa_pow,
    qa_rotate,
    qa_slerp,
    qa_to_angles,
    qa_to_axisangle,
    qa_to_iso,
    qa_to_position,
    qa_to_rotmat,
)
from .utils import AlignedF64, Logger, ensure_buffer_f64, object_ndim

null_quat = np.array([0.0, 0.0, 0.0, 1.0])


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
    if np.shape(axis)[-1] != 3:
        raise RuntimeError("axis is not a 3D vector")
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


def from_axisangle(axis, angle):
    """Convert axis / angle arrays to quaternions.

    Args:
        axis (array_like):  An array of input unit vectors.
        angle (array_like):  The input rotation angles around each axis.

    Returns:
        (array):  The quaternion array results.

    """
    axin = ensure_buffer_f64(axis)
    angin = ensure_buffer_f64(angle)
    qout = AlignedF64(4 * len(angin))
    qa_from_axisangle(axin, angin, qout)
    if len(qout) == 4:
        if object_ndim(axis) == 2:
            return qout.array().reshape((1, 4))
        else:
            return qout.array()
    else:
        return qout.array().reshape((-1, 4))


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


def from_iso_angles(theta, phi, psi):
    """Create quaternions from ISO theta, phi, psi spherical coordinates.

    The input angles describe the ZYZ rotations used to build the quaternion.

    Args:
        theta (array):  Array or scalar theta values in radians
        phi (array):  Array or scalar phi values in radians
        psi (array):  Array or scalar psi values in radians

    Returns:
        (array):  The quaternions.

    """
    intheta = ensure_buffer_f64(theta)
    inphi = ensure_buffer_f64(phi)
    inpsi = ensure_buffer_f64(psi)
    lt = len(intheta)
    out = AlignedF64(4 * lt)
    qa_from_iso(intheta, inphi, inpsi, out)
    if len(out) == 4:
        if (
            (object_ndim(theta) == 1)
            or (object_ndim(phi) == 1)
            or (object_ndim(psi) == 1)
        ):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def from_lonlat_angles(lon, lat, psi):
    """Create quaternions from longitude, latitude, and psi spherical coordinates.

    Args:
        lon (array):  Array or scalar lon values in radians
        lat (array):  Array or scalar lat values in radians
        psi (array):  Array or scalar psi values in radians

    Returns:
        (array):  The quaternions.

    """
    inlon = ensure_buffer_f64(lon)
    inlat = ensure_buffer_f64(lat)
    inpsi = ensure_buffer_f64(psi)
    lt = len(inlon)
    out = AlignedF64(4 * lt)
    theta = AlignedF64(lt)
    theta[:] = 0.5 * np.pi - np.array(inlat)
    qa_from_iso(theta, inlon, inpsi, out)
    if len(out) == 4:
        if (
            (object_ndim(lon) == 1)
            or (object_ndim(lat) == 1)
            or (object_ndim(psi) == 1)
        ):
            return out.array().reshape((1, 4))
        else:
            return out.array()
    else:
        return out.array().reshape((-1, 4))


def to_iso_angles(q):
    """Convert quaternions to ISO theta, phi, psi spherical coordinates.

    Args:
        q (array_like):  The input quaternions.

    Returns:
        (tuple):  The theta, phi, psi angles in radians.

    """
    inq = ensure_buffer_f64(q)
    lq = len(inq) // 4
    theta = AlignedF64(lq)
    phi = AlignedF64(lq)
    psi = AlignedF64(lq)
    qa_to_iso(inq, theta, phi, psi)
    if len(inq) == 4:
        if object_ndim(q) == 2:
            return (theta.array(), phi.array(), psi.array())
        else:
            return (float(theta[0]), float(phi[0]), float(psi[0]))
    return (theta.array(), phi.array(), psi.array())


def to_lonlat_angles(q):
    """Convert quaternions to longitude, latitude, and psi spherical coordinates.

    Args:
        q (array_like):  The input quaternions.

    Returns:
        (tuple):  The longitude, latitude, psi angles in radians.

    """
    inq = ensure_buffer_f64(q)
    lq = len(inq) // 4
    lat = AlignedF64(lq)
    theta = AlignedF64(lq)
    phi = AlignedF64(lq)
    psi = AlignedF64(lq)
    qa_to_iso(inq, theta, phi, psi)
    lat[:] = 0.5 * np.pi - np.array(theta)
    if len(inq) == 4:
        if object_ndim(q) == 2:
            return (phi.array(), lat.array(), psi.array())
        else:
            return (float(phi[0]), float(lat[0]), float(psi[0]))
    return (phi.array(), lat.array(), psi.array())


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
    log = Logger.get()
    log.warning(
        "from_angles() is deprecated, Use from_iso_angles() or from_lonlat_angles() instead"
    )
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
    log = Logger.get()
    log.warning(
        "to_angles() is deprecated, Use to_iso_angles() or to_lonlat_angles() instead"
    )
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
    log = Logger.get()
    log.warning(
        "from_position() is deprecated, Use from_iso_angles() or from_lonlat_angles() instead"
    )
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
    log = Logger.get()
    log.warning(
        "to_position() is deprecated, Use to_iso_angles() or to_lonlat_angles() instead"
    )
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


# J2000 coordinate transforms

# RA, DEC to galactic coordinates

_coordmat_J2000radec2gal = None
_equ2gal = None


def equ2gal():
    """Return the equatorial to galactic coordinate transform quaternion."""
    global _coordmat_J2000radec2gal
    global _equ2gal
    if _coordmat_J2000radec2gal is None:
        _coordmat_J2000radec2gal = np.array(
            [
                -0.054875539726,
                -0.873437108010,
                -0.483834985808,
                0.494109453312,
                -0.444829589425,
                0.746982251810,
                -0.867666135858,
                -0.198076386122,
                0.455983795705,
            ]
        ).reshape([3, 3])
    if _equ2gal is None:
        _equ2gal = from_rotmat(_coordmat_J2000radec2gal)
    return _equ2gal


# RA, DEC to (geocentric) ecliptic coordinates

_coordmat_J2000radec2ecl = None
_equ2ecl = None


def equ2ecl():
    """Return the equatorial to ecliptic coordinate transform quaternion."""
    global _coordmat_J2000radec2ecl
    global _equ2ecl
    if _coordmat_J2000radec2ecl is None:
        _coordmat_J2000radec2ecl = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.917482062069182,
                0.397777155931914,
                0.0,
                -0.397777155931914,
                0.917482062069182,
            ]
        ).reshape([3, 3])
    if _equ2ecl is None:
        _equ2ecl = from_rotmat(coordmat_J2000radec2ecl)
    return _equ2ecl


# Ecliptic coordinates (geocentric) to galactic
# (use the same rotation as HEALPix, to avoid confusion)

_coordmat_J2000ecl2gal = None
_ecl2gal = None


def ecl2gal():
    """Return the ecliptic to galactic coordinate transform quaternion."""
    global _coordmat_J2000ecl2gal
    global _ecl2gal
    if _coordmat_J2000ecl2gal is None:
        _coordmat_J2000ecl2gal = np.array(
            [
                -0.054882486,
                -0.993821033,
                -0.096476249,
                0.494116468,
                -0.110993846,
                0.862281440,
                -0.867661702,
                -0.000346354,
                0.497154957,
            ]
        ).reshape([3, 3])
    if _ecl2gal is None:
        _ecl2gal = from_rotmat(_coordmat_J2000ecl2gal)
    return _ecl2gal
