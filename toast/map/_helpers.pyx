
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack

from libc.stdlib cimport malloc, free
from libc.string cimport memset

f64 = np.float64
i64 = np.int64
i32 = np.int32

ctypedef np.float64_t f64_t
ctypedef np.int64_t i64_t
ctypedef np.int32_t i32_t


def _accumulate_inverse_covariance(
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
    cdef i64_t nblock = int(nnz * (nnz+1) / 2)

    if data.shape[2] != nblock:
        raise RuntimeError("inverse covariance does not have correct shape for NNZ from weights")
    if submap_indx.shape[0] != nsamp:
        raise RuntimeError("submap index list does not have same length as weights")
    if pix_indx.shape[0] != nsamp:
        raise RuntimeError("pixel index list does not have same length as weights")
    if (hits.shape[0] != data.shape[0]) or (hits.shape[0] != data.shape[0]):
        do_hits = 0

    for i in range(nsamp):
        if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
            off = 0
            for elem in range(nnz):
                for alt in range(elem, nnz):
                    data[submap_indx[i], pix_indx[i], off] += scale * weights[i,elem] * weights[i,alt]
                    off += 1

    if do_hits > 0:
        for i in range(nsamp):
            if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
                hits[submap_indx[i], pix_indx[i]] += 1

    return


def _invert_covariance(np.ndarray[f64_t, ndim=3] data, f64_t threshold):
    cdef i64_t nsubmap = data.shape[0]
    cdef i64_t npix = data.shape[1]
    cdef i64_t nblock = data.shape[2]
    cdef i64_t nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
    cdef i64_t i
    cdef i64_t j
    cdef i64_t k
    cdef i64_t m
    cdef i64_t off

    cdef double * fdata = <double*>malloc(nnz*nnz*sizeof(double))
    cdef double * work = <double*>malloc(3*nnz*sizeof(double))
    cdef int * iwork = <int*>malloc(nnz*sizeof(int))
    cdef int fnnz = nnz
    cdef double norm
    cdef double rcond
    cdef int info
    cdef double inverse
    cdef char uplo = 'L'

    if nnz == 1:
        # shortcut
        for i in range(nsubmap):
            for j in range(npix):
                data[i,j,:] = 1.0 / data[i,j,:]
    else:
        for i in range(nsubmap):
            for j in range(npix):
                # copy to fortran compatible buffer
                off = 0
                memset(fdata, 0, nnz*nnz*sizeof(f64_t))
                for k in range(nnz):
                    for m in range(k, nnz):
                        fdata[k*nnz+m] = data[i,j,off]
                        off += 1

                # factor and check condition number
                norm = fdata[0]
                info = 0
                cython_lapack.dpotrf(&uplo, &fnnz, fdata, &fnnz, &info)

                if info == 0:
                    # cholesky worked, compute condition number
                    cython_lapack.dpocon(&uplo, &fnnz, fdata, &fnnz, &norm, &rcond, work, iwork, &info)
                    if info == 0:
                        # compare to threshold
                        if rcond >= threshold:
                            # invert
                            cython_lapack.dpotri(&uplo, &fnnz, fdata, &fnnz, &info)
                            if info == 0:
                                off = 0
                                for k in range(nnz):
                                    for m in range(k, nnz):
                                        data[i,j,off] = fdata[k*nnz+m]
                                        off += 1
    free(fdata)
    free(work)
    free(iwork)
    return

