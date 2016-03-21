
import numpy as np
cimport numpy as np

f64 = np.float64
i64 = np.int64
i32 = np.int32

ctypedef np.float64_t f64_t
ctypedef np.int64_t i64_t
ctypedef np.int32_t i32_t


def accumulate_inverse_covariance(
        np.ndarray[f64_t, ndim=3] data,
        np.ndarray[i64_t, ndim=1] submap_indx,
        np.ndarray[i64_t, ndim=1] pix_indx,
        np.ndarray[f64_t, ndim=2] weights,
        f64_t scale,
        np.ndarray[i64_t, ndim=3] hits
    ):
    '''
    For a vector of pointing weights, build and accumulate the upper triangle
    of the diagonal inverse pixel covariance.
    '''
    cdef i64_t nsamp = weights.shape[0]
    cdef i64_t nnz = weights.shape[1]
    cdef i64_t i
    cdef i64_t elem
    cdef i64_t alt
    cdef i64_t off
    cdef i32_t do_hits = 1

    if submap_indx.shape[0] != nsamp:
        raise RuntimeError("submap index list does not have same length as weights")
    if pix_indx.shape[0] != nsamp:
        raise RuntimeError("pixel index list does not have same length as weights")
    if (hits.shape[0] != data.shape[0]) or (hits.shape[0] != data.shape[0]):
        do_hits = 0

    for i in range(nsamp):
        off = 0
        if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
            for elem in range(nnz):
                for alt in range(elem, nnz):
                    data[submap_indx[i], pix_indx[i], off] += scale * weights[i,elem] * weights[i,alt]
                    off += 1

    if do_hits > 0:
        for i in range(nsamp):
            if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
                hits[submap_indx[i], pix_indx[i]] += 1

    return
