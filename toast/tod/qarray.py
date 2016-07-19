# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from . import _qarray as qc


def arraylist_dot(a, b):
    '''Dot product of a lists of arrays, returns a column array'''
    na = 1
    if a.ndim > 1:
        na = a.shape[0]
    nb = 1
    if b.ndim > 1:
        nb = b.shape[0]
    if (na > 1) and (nb > 1) and (na != nb):
        raise RuntimeError("quaternion arrays must have length one or matching lengths.")
    n = np.max([na, nb])

    aa = a
    if a.ndim == 1:
        aa = np.tile(a, n).reshape((-1, a.shape[0]))
    bb = b
    if b.ndim == 1:
        bb = np.tile(b, n).reshape((-1, b.shape[0]))

    return qc.arraylist_dot(aa, bb)


def inv(q):
    """Inverse of quaternion array q"""
    qq = q
    if q.ndim == 1:
        qq = q.reshape((1, q.shape[0]))
    res = qc.inv(qq)
    if q.ndim == 1:
        return res.flatten()
    else:
        return res


def amplitude(v):
    vv = v
    if v.ndim == 1:
        vv = v.reshape((1, v.shape[0]))
    res = qc.amplitude(vv)
    if v.ndim == 1:
        return res.flatten()
    else:
        return res


def norm(q):
    """Normalize quaternion array q or array list to unit quaternions"""
    qq = q
    if q.ndim == 1:
        qq = q.reshape((1, q.shape[0]))
    res = qc.norm(qq)
    if q.ndim == 1:
        return res.flatten()
    else:
        return res


def rotate(q, v):
    """
    Use a quaternion or array of quaternions (q) to rotate a vector or 
    array of vectors (v).  If the dimensions of both q and v are 2, then
    they must have the same leading dimension.
    """
    nq = 1
    if q.ndim > 1:
        nq = q.shape[0]
    nv = 1
    if v.ndim > 1:
        nv = v.shape[0]
    if (nq > 1) and (nv > 1) and (nq != nv):
        raise RuntimeError("quaternion and vector arrays must have length one or matching lengths.")
    n = np.max([nq, nv])

    vv = v
    if v.ndim == 1:
        vv = np.tile(v, n).reshape((-1, 3))
    qq = q
    if q.ndim == 1:
        qq = np.tile(q, n).reshape((-1, 4))

    return qc.rotate(qq, vv)


def mult(p, q):
    """Multiply arrays of quaternions, see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    """
    qn = 1
    if q.ndim > 1:
        qn = q.shape[0]
    pn = 1
    if p.ndim > 1:
        pn = p.shape[0]
    if (qn > 1) and (pn > 1) and (qn != pn):
        raise RuntimeError("quaternion arrays must have length one or matching lengths.")
    n = np.max([qn, pn])

    pp = p
    if p.ndim == 1:
        pp = np.tile(p, n).reshape((-1, 4))
    qq = q
    if q.ndim == 1:
        qq = np.tile(q, n).reshape((-1, 4))

    return qc.mult(pp, qq)


def slerp(targettime, time, q):
    """Slerp, q quaternion array interpolated from time to targettime"""
    return qc.slerp(targettime, time, q)


def exp(q):
    """Exponential of a quaternion array"""
    qq = q
    if q.ndim == 1:
        qq = q.reshape((1, q.shape[0]))
    print(qq)
    return qc.exp(qq)


def ln(q):
    """Natural logarithm of a quaternion array"""
    qq = q
    if q.ndim == 1:
        qq = q.reshape((1, q.shape[0]))
    return qc.ln(qq)


def pow(q, p):
    """Real power of a quaternion array"""
    qq = q
    if q.ndim == 1:
        qq = q.reshape((1, q.shape[0]))
    pp = p
    if not isinstance(p, np.ndarray):
        pp = np.tile(p, qq.shape[0])
    return qc.pow(qq, pp)

    
def rotation(axis, angle):
    """Rotation quaternions of angles [rad] around axes [already normalized]"""
    nax = 1
    if axis.ndim > 1:
        nax = axis.shape[0]
    nang = 1
    if isinstance(angle, np.ndarray):
        # we have more than one angle
        nang = angle.shape[0]
    if (nax > 1) and (nang > 1) and (nax != nang):
        raise RuntimeError("axis and angle arrays must have length one or matching lengths.")
    n = np.max([nax, nang])

    ax = axis
    if axis.ndim == 1:
        ax = np.tile(axis, n).reshape((-1, 3))
    ang = angle
    if nang == 1:
        ang = np.tile(angle, n)

    res = qc.rotation(ax, ang)
    if n == 1:
        return res.flatten()
    else:
        return res


def to_axisangle(q):
    return qc.to_axisangle(q)


def to_rotmat(q):
    """Rotation matrix"""
    return qc.to_rotmat(q)
                                

def from_rotmat(rotmat):
    return qc.from_rotmat(rotmat)


def from_vectors(v1, v2):
    return qc.from_vectors(v1, v2)

