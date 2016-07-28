# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from . import _qarray as qc


def arraylist_dot(a, b):
    """Dot product of a lists of arrays, returns a column array"""
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
        raise RuntimeError("vector elements of both arrays must have the same length.")
    if (na > 1) and (nb > 1) and (na != nb):
        raise RuntimeError("vector arrays must have length one or matching lengths.")
    n = np.max([na, nb])

    aa = a
    if na != n:
        aa = np.tile(a, n)
    bb = b
    if nb != n:
        bb = np.tile(b, n)

    return qc.arraylist_dot(n, ma, ma, aa.flatten().astype(np.float64, copy=False), bb.flatten().astype(np.float64, copy=False))


def inv(q):
    """Inverse of quaternion array q"""
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    return qc.inv(nq, q.flatten().astype(np.float64, copy=False))


def amplitude(v):
    """Amplitude of a vector array"""
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
    return qc.amplitude(nv, nm, v.flatten().astype(np.float64, copy=False))


def norm(q):
    """Normalize quaternion array to unit quaternions"""
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    return qc.norm(nq, q.flatten().astype(np.float64, copy=False))


def rotate(q, v):
    """
    Use a quaternion or array of quaternions (q) to rotate a vector or 
    array of vectors (v).  If the number of dimensions of both q and v 
    are 2, then they must have the same leading dimension.
    """
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

    if (nq > 1) and (nv > 1) and (nq != nv):
        raise RuntimeError("quaternion and vector arrays must have length one or matching lengths.")
    n = np.max([nq, nv])

    vv = v
    if nv != n:
        vv = np.tile(v, n)
    qq = q
    if nq != n:
        qq = np.tile(q, n)

    return qc.rotate(n, qq.flatten().astype(np.float64, copy=False), vv.flatten().astype(np.float64, copy=False))


def mult(p, q):
    """Multiply arrays of quaternions, see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    """
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

    if (nq > 1) and (pn > 1) and (nq != pn):
        raise RuntimeError("quaternion arrays must have length one or matching lengths.")
    n = np.max([nq, pn])

    pp = p
    if pn != n:
        pp = np.tile(p, n)
    qq = q
    if nq != n:
        qq = np.tile(q, n)

    return qc.mult(n, pp.flatten().astype(np.float64, copy=False), qq.flatten().astype(np.float64, copy=False))


def slerp(targettime, time, q):
    """Slerp, q quaternion array interpolated from time to targettime"""
    if not isinstance(targettime, np.ndarray):
        targettime = np.array(targettime, dtype=np.float64)
    if not isinstance(time, np.ndarray):
        time = np.array(time, dtype=np.float64)
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    return qc.slerp(targettime, time, q.flatten().astype(np.float64, copy=False))


def exp(q):
    """Exponential of a quaternion array"""
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]    
    return qc.exp(nq, q.flatten().astype(np.float64, copy=False))


def ln(q):
    """Natural logarithm of a quaternion array"""
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]    
    return qc.ln(nq, q.flatten().astype(np.float64, copy=False))


def pow(q, p):
    """Real power of a quaternion array"""
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

    return qc.pow(qq.flatten().astype(np.float64, copy=False), pp.flatten().astype(np.float64, copy=False))

    
def rotation(axis, angle):
    """Rotation quaternions of angles [rad] around axes [already normalized]"""
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

    return qc.rotation(ax.flatten().astype(np.float64, copy=False), ang.flatten().astype(np.float64, copy=False))


def to_axisangle(q):
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    nq = None
    if q.ndim == 1:
        nq = 1
    else:
        nq = q.shape[0]
    return qc.to_axisangle(nq, q.flatten().astype(np.float64, copy=False))


def to_rotmat(q):
    """Rotation matrix"""
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype=np.float64)
    return qc.to_rotmat(q.flatten().astype(np.float64, copy=False))
                                

def from_rotmat(rotmat):
    if not isinstance(rotmat, np.ndarray):
        rotmat = np.array(rotmat, dtype=np.float64)
    return qc.from_rotmat(rotmat.flatten().astype(np.float64, copy=False))


def from_vectors(v1, v2):
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1, dtype=np.float64)
    if not isinstance(v2, np.ndarray):
        v2 = np.array(v2, dtype=np.float64)
    return qc.from_vectors(v1.flatten().astype(np.float64, copy=False), v2.flatten().astype(np.float64, copy=False))

