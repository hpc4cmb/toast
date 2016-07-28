# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from . import _qarray as qc


def inv(q):
    """Inverse of quaternion array q"""
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    return qc.inv(nq, q.flatten())


def amplitude(v):
    """Amplitude of a vector array"""
    nv = None
    nm = None
    if v.ndim == 1:
        nv = 1
        nm = v.shape[0]
    else:
        nv = v.shape[0]
        nm = v.shape[1]
    return qc.amplitude(nv, nm, v.flatten())


def norm(q):
    """Normalize quaternion array to unit quaternions"""
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    return qc.norm(nq, q.flatten())


def rotate(q, v):
    """
    Use a quaternion or array of quaternions (q) to rotate a vector or 
    array of vectors (v).  If the number of dimensions of both q and v 
    are 2, then they must have the same leading dimension.
    """
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

    if (nq > 1) and (nv > 1) and (nq != nv):
        raise RuntimeError("quaternion and vector arrays must have length one or matching lengths.")
    n = np.max([nq, nv])

    vv = v
    if nv != n:
        vv = np.tile(v, n)
    qq = q
    if nq != n:
        qq = np.tile(q, n)

    return qc.rotate(n, qq.flatten(), vv.flatten())


def mult(p, q):
    """Multiply arrays of quaternions, see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    """
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

    if (nq > 1) and (pn > 1) and (nq != pn):
        raise RuntimeError("quaternion arrays must have length one or matching lengths.")
    n = np.max([nq, pn])

    pp = p
    if pn != n:
        pp = np.tile(p, n)
    qq = q
    if nq != n:
        qq = np.tile(q, n)

    return qc.mult(n, pp.flatten(), qq.flatten())


def slerp(targettime, time, q):
    """Slerp, q quaternion array interpolated from time to targettime"""
    return qc.slerp(targettime, time, q.flatten())


def exp(q):
    """Exponential of a quaternion array"""
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]    
    return qc.exp(nq, q.flatten())


def ln(q):
    """Natural logarithm of a quaternion array"""
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]    
    return qc.ln(nq, q.flatten())


def pow(q, p):
    """Real power of a quaternion array"""
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

    if (nq > 1) and (pn > 1) and (nq != pn):
        raise RuntimeError("quaternion array and power must have length one or matching lengths.")
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

    return qc.pow(qq.flatten(), pp.flatten())

    
def rotation(axis, angle):
    """Rotation quaternions of angles [rad] around axes [already normalized]"""
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

    if (nax > 1) and (nang > 1) and (nax != nang):
        raise RuntimeError("axis and angle arrays must have length one or matching lengths.")
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

    return qc.rotation(ax.flatten(), ang.flatten())


def to_axisangle(q):
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    return qc.to_axisangle(nq, q.flatten())


def to_rotmat(q):
    """Rotation matrix"""
    return qc.to_rotmat(q.flatten())
                                

def from_rotmat(rotmat):
    return qc.from_rotmat(rotmat.flatten())


def from_vectors(v1, v2):
    return qc.from_vectors(v1.flatten(), v2.flatten())

