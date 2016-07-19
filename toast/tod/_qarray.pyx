
import numpy as np
cimport numpy as np

np.import_array()

f64 = np.float64
i64 = np.int64
i32 = np.int32

ctypedef np.float64_t f64_t
ctypedef np.uint64_t u64_t
ctypedef np.int32_t i32_t


cdef extern from "pytoast.h":
    void pytoast_qarraylist_dot(int n, int m, int d, const double* a, const double* b, double* dotprod)
    void pytoast_qinv(int n, double* q)
    void pytoast_qamplitude(int n, int m, int d, const double* v, double* l2)
    void pytoast_qnorm(int n, int m, int d, const double* q_in, double* q_out)
    void pytoast_qrotate(int n, const double* v, const double* q_in, double* v_out)
    void pytoast_qmult(int n, const double* p, const double* q, double* r)
    void pytoast_slerp(int n_time, int n_targettime, const double* time, const double* targettime, const double* q_in, double* q_interp)
    void pytoast_qexp(int n, const double* q_in, double* q_out)
    void pytoast_qln(int n, const double* q_in, double* q_out)
    void pytoast_qpow(int n, const double* p, const double* q_in, double* q_out)
    void pytoast_from_axisangle(int n, const double* axis, const double* angle, double* q_out)
    void pytoast_to_axisangle(const double* q, double* axis, double* angle)
    void pytoast_to_rotmat(const double* q, double* rotmat)
    void pytoast_from_rotmat(const double* rotmat, double* q)
    void pytoast_from_vectors(const double* vec1, const double* vec2, double* q)


def arraylist_dot(np.ndarray[f64_t, ndim=2] a, np.ndarray[f64_t, ndim=2] b):
    '''Dot product of a lists of arrays, returns a column array'''
    cdef int n = a.shape[0]
    cdef int m = a.shape[1]
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(n, dtype=f64)
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    pytoast_qarraylist_dot(n, m, m, <double*>a.data, <double*>b.data, <double*>out.data)
    return out


def inv(np.ndarray[f64_t, ndim=2] q):
    """Inverse of quaternion array q"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.copy(q)
    pytoast_qinv(n, <double*>out.data)
    return out


def amplitude(np.ndarray[f64_t, ndim=2] v):
    cdef int n = v.shape[0]
    cdef int m = v.shape[1]
    v = np.ascontiguousarray(v)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zero(n, dtype=f64)
    pytoast_qamplitude(n, m, m, <double*>v.data, <double*>out.data)
    return out


def norm(np.ndarray[f64_t, ndim=2] q):
    """Normalize quaternion array q or array list to unit quaternions"""
    cdef int n = q.shape[0]
    cdef int m = q.shape[1]
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.zeros_like(q)
    pytoast_qnorm(n, m, m, <double*>q.data, <double*>out.data)
    return out


def rotate(np.ndarray[f64_t, ndim=2] q, np.ndarray[f64_t, ndim=2] v):
    """Rotate vector or array of vectors v by quaternion q"""
    cdef int n = q.shape[0]
    q = np.ascontiguousarray(q)
    v = np.ascontiguousarray(v)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.zeros_like(v)
    pytoast_qrotate(n, <double*>v.data, <double*>q.data, <double*>out.data)
    return out


def mult(np.ndarray[f64_t, ndim=2] p, np.ndarray[f64_t, ndim=2] q):
    """Multiply arrays of quaternions, see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    """
    cdef int n = q.shape[0]
    p = np.ascontiguousarray(p)
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.zeros_like(q)
    pytoast_qmult(n, <double*>p.data, <double*>q.data, <double*>out.data)
    return out


def slerp(np.ndarray[f64_t, ndim=1] targettime, np.ndarray[f64_t, ndim=1] time, np.ndarray[f64_t, ndim=2] q):
    """Slerp, q quaternion array interpolated from time to targettime"""
    cdef int ntime = time.shape[0]
    cdef int ntarget = targettime.shape[0]
    time = np.ascontiguousarray(time)
    targettime = np.ascontiguousarray(targettime)
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.zeros((ntarget, 4), dtype=f64)
    pytoast_slerp(ntime, ntarget, <double*>time.data, <double*>targettime.data, <double*>q.data, <double*>out.data)
    return out


def exp(np.ndarray[f64_t, ndim=2] q):
    """Exponential of a quaternion array"""
    cdef int n = q.shape[0]
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.zeros_like(q)
    pytoast_qexp(n, <double*>q.data, <double*>out.data)
    return out


def ln(np.ndarray[f64_t, ndim=2] q):
    """Natural logarithm of a quaternion array"""
    cdef int n = q.shape[0]
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.copy(q)
    pytoast_qln(n, <double*>q.data, <double*>out.data)
    return out


def pow(np.ndarray[f64_t, ndim=2] q, np.ndarray[f64_t, ndim=1] p):
    """Real power of a quaternion array"""
    cdef int n = q.shape[0]
    q = np.ascontiguousarray(q)
    p = np.ascontiguousarray(p)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.copy(q)
    pytoast_qpow(n, <double*>p.data, <double*>q.data, <double*>out.data)
    return out
    

def rotation(np.ndarray[f64_t, ndim=2] axis, np.ndarray[f64_t, ndim=1] angle):
    """Rotation quaternions of angles [rad] around axes [already normalized]"""
    cdef int n = axis.shape[0]
    axis = np.ascontiguousarray(axis)
    angle = np.ascontiguousarray(angle)
    cdef np.ndarray[f64_t, ndim=2, mode='c'] out = np.zeros((n,4), dtype=f64)
    pytoast_from_axisangle(n, <double*>axis.data, <double*>angle.data, <double*>out.data)
    return out


def to_axisangle(np.ndarray[f64_t, ndim=1] q):
    cdef f64_t angle
    cdef np.ndarray[f64_t, ndim=1, mode='c'] axis = np.zeros(3, dtype=f64)
    q = np.ascontiguousarray(q)
    pytoast_to_axisangle(<double*>q.data, <double*>axis.data, <double*>&angle)
    return axis, angle


def to_rotmat(np.ndarray[f64_t, ndim=1] q):
    """Rotation matrix"""
    cdef np.ndarray[f64_t, ndim=2, mode='c'] rot = np.zeros((3,3), dtype=f64)
    q = np.ascontiguousarray(q)
    pytoast_to_rotmat(<double*>q.data, <double*>rot.data)
    return rot


def from_rotmat(np.ndarray[f64_t, ndim=2] rotmat):
    cdef np.ndarray[f64_t, ndim=1, mode='c'] q = np.zeros(4, dtype=f64)
    rotmat = np.ascontiguousarray(rotmat)
    pytoast_from_rotmat(<double*>rotmat.data, <double*>q.data)
    return q


def from_vectors(np.ndarray[f64_t, ndim=1] v1, np.ndarray[f64_t, ndim=1] v2):
    cdef np.ndarray[f64_t, ndim=1, mode='c'] q = np.zeros(4, dtype=f64)
    v1 = np.ascontiguousarray(v1)
    v2 = np.ascontiguousarray(v2)
    pytoast_from_vectors(<double*>v1.data, <double*>v2.data, <double*>q.data)
    return q

