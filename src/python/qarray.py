# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import collections

import numpy as np

from . import ctoast as ctoast
import timemory


def arraylist_dot(a, b):
    """Dot product of lists of arrays, returns a column array"""
    autotimer = timemory.auto_timer()
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=np.float64)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=np.float64)
    na = None
    ma = None
    if a.ndim == 1:
        na = 1
        ma = a.shape[0]
    else:
        na = a.shape[0]
        ma = a.shape[1]
    nb = None
    mb = None
    if b.ndim == 1:
        nb = 1
        mb = b.shape[0]
    else:
        nb = b.shape[0]
        mb = b.shape[1]

    if ma != mb:
        raise RuntimeError("vector elements of both arrays must "
                           "have the same length.")
    if na > 1 and nb > 1 and na != nb:
        raise RuntimeError("vector arrays must have length one or "
                           "matching lengths.")
    n = np.max([na, nb])

    aa = a
    if na != n:
        aa = np.tile(a, n)
    bb = b
    if nb != n:
        bb = np.tile(b, n)

    return ctoast.qarray_list_dot(
        n, ma, ma, aa.flatten().astype(np.float64, copy=False),
        bb.flatten().astype(np.float64, copy=False))


def inv(q):
    """Inverse of quaternion array q"""
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    ret = ctoast.qarray_inv(nq, q.flatten().astype(np.float64, copy=False))
    if q.ndim != 1:
        ret = ret.reshape((-1, 4))
    return ret


def amplitude(v):
    """Amplitude of a vector array"""
    autotimer = timemory.auto_timer()
    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype=np.float64)
    nv = None
    nm = None
    if v.ndim == 1:
        nv = 1
        nm = v.shape[0]
    else:
        nv = v.shape[0]
        nm = v.shape[1]
    ret = ctoast.qarray_amplitude(
        nv, nm, nm, v.flatten().astype(np.float64, copy=False))
    if v.ndim == 1:
        ret = ret[0]
    return ret


def norm(q):
    """Normalize quaternion array to unit quaternions"""
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    ret = ctoast.qarray_normalize(
        nq, 4, 4, q.flatten().astype(np.float64, copy=False))
    if q.ndim != 1:
        ret = ret.reshape((-1, 4))
    return ret


def rotate(q, v):
    """
    Use a quaternion or array of quaternions (q) to rotate a vector or
    array of vectors (v).
    """
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    nv = None
    if v.ndim == 1:
        nv = 1
    else:
        nv = v.shape[0]

    if nq > 1 and nv > 1 and nq != nv:
        raise RuntimeError("quaternion and vector arrays must have length "
                           "one or matching lengths.")

    ret = ctoast.qarray_rotate(
        nq, q.flatten().astype(np.float64, copy=False),
        nv, v.flatten().astype(np.float64, copy=False))
    if q.ndim != 1 or v.ndim != 1:
        ret = ret.reshape((-1, 3))
    return ret


def mult(p, q):
    """Multiply arrays of quaternions, see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    """
    autotimer = timemory.auto_timer()
    if not isinstance(p, np.ndarray):
        p = np.array(p, dtype=np.float64)
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    pn = None
    if p.ndim == 1:
        pn = 1
    else:
        pn = p.shape[0]

    if nq > 1 and pn > 1 and nq != pn:
        raise RuntimeError(
            "quaternion arrays must have length one or matching lengths.")

    ret = ctoast.qarray_mult(
        pn, p.flatten().astype(np.float64, copy=False),
        nq, q.flatten().astype(np.float64, copy=False))

    if p.ndim != 1 or q.ndim != 1:
        ret = ret.reshape((-1, 4))
    return ret


def slerp(targettime, time, q):
    """Slerp, q quaternion array interpolated from time to targettime"""
    autotimer = timemory.auto_timer()
    ttime = targettime
    if not isinstance(ttime, np.ndarray):
        ttime = np.array(ttime, dtype=np.float64, ndmin=1)
    if not isinstance(time, np.ndarray):
        time = np.array(time, dtype=np.float64)
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    n_time = len(time)
    n_targettime = len(ttime)
    nq = len(q.flatten())
    if nq != 4*n_time:
        raise RuntimeError("input quaternion and time arrays have different "
                           "numbers of elements")
    ret = ctoast.qarray_slerp(
        n_time, n_targettime, time, ttime,
        q.flatten().astype(np.float64, copy=False))

    if isinstance(targettime, collections.Iterable):
        ret = ret.reshape((-1, 4))
    return ret


def exp(q):
    """Exponential of a quaternion array"""
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    ret = ctoast.qarray_exp(nq, q.flatten().astype(np.float64, copy=False))
    if q.ndim == 1:
        return ret
    else:
        return ret.reshape((-1, 4))


def ln(q):
    """Natural logarithm of a quaternion array"""
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    ret = ctoast.qarray_ln(nq, q.flatten().astype(np.float64, copy=False))
    if q.ndim != 1:
        ret = ret.reshape((-1, 4))
    return ret


def pow(q, p):
    """Real power of a quaternion array"""
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    pn = None
    if isinstance(p, np.ndarray):
        pn = p.shape[0]
    else:
        pn = 1

    if nq > 1 and pn > 1 and nq != pn:
        raise RuntimeError("quaternion array and power must have length "
                           "one or matching lengths.")
    n = np.max([nq, pn])

    pp = None
    if isinstance(p, np.ndarray):
        if pn != n:
            pp = np.tile(p, n)
        else:
            pp = p
    else:
        pp = np.tile(p, n)

    qq = q
    if nq != n:
        qq = np.tile(q, n)

    ret = ctoast.qarray_pow(n, pp.flatten().astype(np.float64, copy=False),
                            qq.flatten().astype(np.float64, copy=False))
    if q.ndim != 1 or isinstance(p, collections.Iterable):
        ret = ret.reshape((-1, 4))

    return ret


def rotation(axis, angle):
    """Rotation quaternions of angles [rad] around axes [already normalized]"""
    autotimer = timemory.auto_timer()
    if not isinstance(axis, np.ndarray):
        axis = np.array(axis, dtype=np.float64)
    nax = None
    if axis.ndim == 1:
        nax = 1
    else:
        nax = axis.shape[0]
    nang = None
    if isinstance(angle, np.ndarray):
        nang = angle.shape[0]
    else:
        nang = 1

    if nax > 1 and nang > 1 and nax != nang:
        raise RuntimeError(
            "axis and angle arrays must have length one or matching lengths.")
    n = np.max([nax, nang])

    ax = axis
    if nax != n:
        ax = np.tile(axis, n)

    ang = None
    if isinstance(angle, np.ndarray):
        if nang != n:
            ang = np.tile(angle, n)
        else:
            ang = angle
    else:
        ang = np.tile(angle, n)

    ret = ctoast.qarray_from_axisangle(
        n, ax.flatten().astype(np.float64, copy=False),
        ang.flatten().astype(np.float64, copy=False))

    if axis.ndim != 1 or isinstance(angle, collections.Iterable):
        ret = ret.reshape((-1, 4))

    return ret


def to_axisangle(q):
    """To Axis Angle"""
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    (axis, angle) = ctoast.qarray_to_axisangle(
        nq, q.flatten().astype(np.float64, copy=False))

    if q.ndim == 1:
        angle = angle[0]
    else:
        axis = axis.reshape((-1, 3))

    return (axis, angle)


def to_rotmat(q):
    """Rotation matrix"""
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError('to_rotmat is not vectorized')
    return ctoast.qarray_to_rotmat(
        q.flatten().astype(np.float64, copy=False)).reshape((3, 3))


def from_rotmat(rotmat):
    autotimer = timemory.auto_timer()
    if not isinstance(rotmat, np.ndarray):
        rotmat = np.array(rotmat, dtype=np.float64)
    if rotmat.ndim != 2:
        raise ValueError('from_rotmat is not vectorized')
    return ctoast.qarray_from_rotmat(
        rotmat.flatten().astype(np.float64, copy=False))


def from_vectors(v1, v2):
    autotimer = timemory.auto_timer()
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1, dtype=np.float64)
    if not isinstance(v2, np.ndarray):
        v2 = np.array(v2, dtype=np.float64)
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError('from_vectors is not vectorized')
    return ctoast.qarray_from_vectors(
        v1.flatten().astype(np.float64, copy=False),
        v2.flatten().astype(np.float64, copy=False))


def from_angles(theta, phi, pa, IAU=False):
    autotimer = timemory.auto_timer()
    iterable = (isinstance(theta, collections.Iterable) or
                isinstance(phi, collections.Iterable) or
                isinstance(pa, collections.Iterable))

    if not isinstance(theta, np.ndarray):
        theta = np.array(theta, dtype=np.float64, ndmin=1)
    nt = theta.shape[0]

    if not isinstance(phi, np.ndarray):
        phi = np.array(phi, dtype=np.float64, ndmin=1)
    nph = phi.shape[0]

    if not isinstance(pa, np.ndarray):
        pa = np.array(pa, dtype=np.float64, ndmin=1)
    npa = pa.shape[0]

    if nt != nph or nt != npa or nph != npa:
        raise RuntimeError("all input angle arrays must have the same length")

    q = ctoast.qarray_from_angles(nt, theta, phi, pa, IAU)

    if iterable:
        q = q.reshape((-1, 4))
    else:
        q = q.flatten()
    return q

def to_angles(q, IAU=False):
    autotimer = timemory.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    theta, phi, psi = ctoast.qarray_to_angles(
        nq, q.flatten().astype(np.float64, copy=False), IAU)
    if q.ndim == 1:
        theta, phi, psi = (theta[0], phi[0], psi[0])
    return (theta, phi, psi)

def to_position(q):
    autotimer = timing.auto_timer()
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    theta, phi = ctoast.qarray_to_position(
        nq, q.flatten().astype(np.float64, copy=False))
    if q.ndim == 1:
        theta, phi = (theta[0], phi[0])
    return (theta, phi)
