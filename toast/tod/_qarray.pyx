
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
    void pytoast_to_axisangle(int n, const double* q, double* axis, double* angle)
    void pytoast_to_rotmat(const double* q, double* rotmat)
    void pytoast_from_rotmat(const double* rotmat, double* q)
    void pytoast_from_vectors(const double* vec1, const double* vec2, double* q)


def inv(int n, np.ndarray[f64_t, ndim=1] q):
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.copy(q)
    pytoast_qinv(n, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 4))


def amplitude(int n, int m, np.ndarray[f64_t, ndim=1] v):
    v = np.ascontiguousarray(v)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(n, dtype=f64)
    pytoast_qamplitude(n, m, m, <double*>v.data, <double*>out.data)
    return out


def norm(int n, np.ndarray[f64_t, ndim=1] q):
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(4*n, dtype=f64)
    pytoast_qnorm(n, 4, 4, <double*>q.data, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 4))


def rotate(int n, np.ndarray[f64_t, ndim=1] q, np.ndarray[f64_t, ndim=1] v):
    q = np.ascontiguousarray(q)
    v = np.ascontiguousarray(v)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(3*n, dtype=f64)
    pytoast_qrotate(n, <double*>v.data, <double*>q.data, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 3))


def mult(int n, np.ndarray[f64_t, ndim=1] p, np.ndarray[f64_t, ndim=1] q):
    p = np.ascontiguousarray(p)
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(4*n, dtype=f64)
    pytoast_qmult(n, <double*>p.data, <double*>q.data, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 4))


def slerp(np.ndarray[f64_t, ndim=1] targettime, np.ndarray[f64_t, ndim=1] time, np.ndarray[f64_t, ndim=1] q):
    cdef int ntime = time.shape[0]
    cdef int ntarget = targettime.shape[0]
    time = np.ascontiguousarray(time)
    targettime = np.ascontiguousarray(targettime)
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(ntarget*4, dtype=f64)
    pytoast_slerp(ntime, ntarget, <double*>time.data, <double*>targettime.data, <double*>q.data, <double*>out.data)
    if ntarget == 1:
        return out
    else:
        return out.reshape((ntarget, 4))


def exp(int n, np.ndarray[f64_t, ndim=1] q):
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(4*n, dtype=f64)
    pytoast_qexp(n, <double*>q.data, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 4))


def ln(int n, np.ndarray[f64_t, ndim=1] q):
    q = np.ascontiguousarray(q)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(4*n, dtype=f64)
    pytoast_qln(n, <double*>q.data, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 4))


def pow(np.ndarray[f64_t, ndim=1] q, np.ndarray[f64_t, ndim=1] p):
    cdef int n = p.shape[0]
    q = np.ascontiguousarray(q)
    p = np.ascontiguousarray(p)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(n*4, dtype=f64)
    pytoast_qpow(n, <double*>p.data, <double*>q.data, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 4))
    

def rotation(np.ndarray[f64_t, ndim=1] axis, np.ndarray[f64_t, ndim=1] angle):
    cdef int n = angle.shape[0]
    axis = np.ascontiguousarray(axis)
    angle = np.ascontiguousarray(angle)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] out = np.zeros(n*4, dtype=f64)
    pytoast_from_axisangle(n, <double*>axis.data, <double*>angle.data, <double*>out.data)
    if n == 1:
        return out
    else:
        return out.reshape((n, 4))


def to_axisangle(int n, np.ndarray[f64_t, ndim=1] q):
    cdef np.ndarray[f64_t, ndim=1, mode='c'] angle = np.zeros(n, dtype=f64)
    cdef np.ndarray[f64_t, ndim=1, mode='c'] axis = np.zeros(3*n, dtype=f64)
    q = np.ascontiguousarray(q)
    pytoast_to_axisangle(n, <double*>q.data, <double*>axis.data, <double*>angle.data)
    if n == 1:
        return axis, angle[0]
    else:
        return axis.reshape((n, 3)), angle


def to_rotmat(np.ndarray[f64_t, ndim=1] q):
    cdef np.ndarray[f64_t, ndim=1, mode='c'] rot = np.zeros(9, dtype=f64)
    q = np.ascontiguousarray(q)
    pytoast_to_rotmat(<double*>q.data, <double*>rot.data)
    return rot.reshape((3,3))


def from_rotmat(np.ndarray[f64_t, ndim=1] rotmat):
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

