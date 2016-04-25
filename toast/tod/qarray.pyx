
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
    void pytoast_qarraylist_dot(const int n, const int m, const double* a, const double* b, double* dotprod)
    void pytoast_qinv(const int n, double* q)
    void pytoast_qamplitude(const int n, const int m, const double* v, double* l2)
    void pytoast_qnorm(const int n, const int m, const double* q_in, double* q_out)
    void pytoast_qnorm_inplace(const int n, const int m, double* q)
    void pytoast_qrotate(const int n, const double* v, const double* q_in, double* v_out)
    void pytoast_qmult(const int n, const double* p, const double* q, double* r)
    void pytoast_nlerp(const int n_time, const double* targettime, const double* time, const double* q_in, double* q_interp)
    void pytoast_slerp(const int n_time, const double* targettime, const double* time, const double* q_in, double* q_interp)
    void pytoast_compute_t(const int n_time, const double* targettime, const double* time, double* t_matrix)
    void pytoast_qexp(const int n, const double* q_in, double* q_out)
    void pytoast_qln(const int n, const double* q_in, double* q_out)
    void pytoast_qpow(const int n, const double* p, const double* q_in, double* q_out)
    void pytoast_from_axisangle(const int n, const double* axis, const double* angle, double* q_out)
    void pytoast_to_axisangle(const double* q, double* axis, double* angle)
    void pytoast_to_rotmat(const double* q, double* rotmat)
    void pytoast_from_rotmat(const double* rotmat, double* q)
    void pytoast_from_vectors(const double* vec1, const double* vec2, double* q)


def arraylist_dot(np.ndarray[f64_t, ndim=2] a, np.ndarray[f64_t, ndim=2] b):
    '''Dot product of a lists of arrays, returns a column array'''
    cdef int n = a.shape[0]
    cdef int m = a.shape[1]
    cdef np.ndarray[f64_t, ndim=1] out = np.zeros(n, dtype=f64)
    pytoast_qarraylist_dot(n, m, <double*>a.data, <double*>b.data, <double*>out.data)
    return out


def inv(np.ndarray[f64_t, ndim=2] q):
    """Inverse of quaternion array q"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.copy(q)
    pytoast_qinv(n, <double*>out.data)
    return out


def amplitude(np.ndarray[f64_t, ndim=2] v):
    cdef int n = v.shape[0]
    cdef int m = v.shape[1]
    cdef np.ndarray[f64_t, ndim=1] out = np.zeros(n, dtype=f64)
    pytoast_qamplitude(n, m, <double*>v.data, <double*>out.data)
    return out


def norm(np.ndarray[f64_t, ndim=2] q):
    """Normalize quaternion array q or array list to unit quaternions"""
    cdef int n = q.shape[0]
    cdef int m = q.shape[1]
    cdef np.ndarray[f64_t, ndim=2] out = np.copy(q)
    pytoast_qnorm(n, m, <double*>q.data, <double*>out.data)
    return out


def norm_inplace(np.ndarray[f64_t, ndim=2] q):
    """Normalize quaternion array q or array list to unit quaternions"""
    cdef int n = q.shape[0]
    cdef int m = q.shape[1]
    pytoast_qnorm_inplace(n, m, <double*>q.data)
    return


def rotate(np.ndarray[f64_t, ndim=2] q, np.ndarray[f64_t, ndim=2] v):
    """Rotate vector or array of vectors v by quaternion q"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.zeros_like(v)
    pytoast_qrotate(n, <double*>v.data, <double*>q.data, <double*>out.data)
    return out


def mult(np.ndarray[f64_t, ndim=2] p, np.ndarray[f64_t, ndim=2] q):
    """Multiply arrays of quaternions,
    see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    """
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.zeros_like(q)
    pytoast_qmult(n, <double*>p.data, <double*>q.data, <double*>out.data)
    return out


def nlerp(np.ndarray[f64_t, ndim=1] targettime, np.ndarray[f64_t, ndim=1] time, np.ndarray[f64_t, ndim=2] q):
    """Nlerp, q quaternion array interpolated from time to targettime"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.zeros_like(q)
    pytoast_nlerp(n, <double*>targettime.data, <double*>time.data, <double*>q.data, <double*>out.data)
    return out


def slerp(np.ndarray[f64_t, ndim=1] targettime, np.ndarray[f64_t, ndim=1] time, np.ndarray[f64_t, ndim=2] q):
    """Slerp, q quaternion array interpolated from time to targettime"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.zeros_like(q)
    pytoast_slerp(n, <double*>targettime.data, <double*>time.data, <double*>q.data, <double*>out.data)
    return out


def compute_t(np.ndarray[f64_t, ndim=1] targettime, np.ndarray[f64_t, ndim=1] time):
    cdef int n = time.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.zeros_like(time)
    pytoast_compute_t(n, <double*>targettime.data, <double*>time.data, <double*>out.data)
    return out


def exp(np.ndarray[f64_t, ndim=2] q):
    """Exponential of a quaternion array"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.copy(q)
    pytoast_qexp(n, <double*>q.data, <double*>out.data)
    return out


def ln(np.ndarray[f64_t, ndim=2] q):
    """Natural logarithm of a quaternion array"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.copy(q)
    pytoast_qln(n, <double*>q.data, <double*>out.data)
    return out


def pow(np.ndarray[f64_t, ndim=2] q, np.ndarray[f64_t, ndim=1] p):
    """Real power of a quaternion array"""
    cdef int n = q.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.copy(q)
    pytoast_qpow(n, <double*>p.data, <double*>q.data, <double*>out.data)
    return out
    

def rotation(np.ndarray[f64_t, ndim=2] axis, np.ndarray[f64_t, ndim=1] angle):
    """Rotation quaternions of angles [rad] around axes [already normalized]"""
    cdef int n = axis.shape[0]
    cdef np.ndarray[f64_t, ndim=2] out = np.zeros((n,4), dtype=f64)
    pytoast_from_axisangle(n, <double*>axis.data, <double*>angle.data, <double*>out.data)
    return out


def to_axisangle(np.ndarray[f64_t, ndim=1] q):
    cdef f64_t angle
    cdef np.ndarray[f64_t, ndim=1] axis = np.zeros(3, dtype=f64)
    pytoast_to_axisangle(<double*>q.data, <double*>axis.data, <double*>&angle)
    return axis, angle


def to_rotmat(np.ndarray[f64_t, ndim=1] q):
    """Rotation matrix"""
    cdef np.ndarray[f64_t, ndim=2] rot = np.zeros((3,3), dtype=f64)
    pytoast_to_rotmat(<double*>q.data, <double*>rot.data)
    return rot
                                

def from_rotmat(np.ndarray[f64_t, ndim=2] rotmat):
    cdef np.ndarray[f64_t, ndim=1] q = np.zeros(4, dtype=f64)
    pytoast_from_rotmat(<double*>rotmat.data, <double*>q.data)
    return q


def from_vectors(np.ndarray[f64_t, ndim=1] v1, np.ndarray[f64_t, ndim=1] v2):
    cdef np.ndarray[f64_t, ndim=1] q = np.zeros(4, dtype=f64)
    pytoast_from_vectors(<double*>v1.data, <double*>v2.data, <double*>q.data)
    return q

