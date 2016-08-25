
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas

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


def _accumulate_noiseweighted(
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
                if ( data[i,j,0] != 0 ):
                    data[i,j,0] = 1.0 / data[i,j,0]
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
                        else:
                            info = 1

                if info != 0:
                    data[i,j,:] = 0

    free(fdata)
    free(work)
    free(iwork)
    return


def _multiply_covariance(np.ndarray[f64_t, ndim=3] data1, np.ndarray[f64_t, ndim=3] data2):
    cdef i64_t nsubmap = data1.shape[0]
    cdef i64_t npix = data1.shape[1]
    cdef i64_t nblock = data1.shape[2]
    cdef i64_t nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
    cdef i64_t i
    cdef i64_t j
    cdef i64_t k
    cdef i64_t m
    cdef i64_t off

    cdef double * fdata1 = <double*>malloc(nnz*nnz*sizeof(double))
    cdef double * fdata2 = <double*>malloc(nnz*nnz*sizeof(double))
    cdef double * fdata3 = <double*>malloc(nnz*nnz*sizeof(double))
    cdef int fnnz = nnz
    cdef double fone = 1
    cdef double fzero = 0
    cdef char side = 'L'
    cdef char uplo = 'L'

    if nnz == 1:
        # shortcut
        for i in range(nsubmap):
            for j in range(npix):
                data1[i,j,0] *= data2[i,j,0]
    else:
        for i in range(nsubmap):
            for j in range(npix):
                # copy to fortran compatible buffer
                memset(fdata1, 0, nnz*nnz*sizeof(f64_t))
                memset(fdata2, 0, nnz*nnz*sizeof(f64_t))
                memset(fdata3, 0, nnz*nnz*sizeof(f64_t))
                off = 0
                for k in range(nnz):
                    for m in range(k, nnz):
                        fdata1[k*nnz+m] = data1[i,j,off]
                        fdata2[k*nnz+m] = data2[i,j,off]
                        if k != m:
                            # Second argument to dsymm must be full
                            fdata2[m*nnz+k] = data2[i,j,off]
                        off += 1
                        
                cython_blas.dsymm(&side, &uplo, &fnnz, &fnnz, &fone, fdata1, &fnnz, fdata2, &fnnz, &fzero, fdata3, &fnnz)

                off = 0
                for k in range(nnz):
                    for m in range(k, nnz):
                        data1[i,j,off] = fdata3[k*nnz+m]
                        off += 1
    free(fdata1)
    free(fdata2)
    free(fdata3)
    return


def _cond_covariance(np.ndarray[f64_t, ndim=3] data, np.ndarray[f64_t, ndim=3] cond):
    cdef i64_t nsubmap = data.shape[0]
    cdef i64_t npix = data.shape[1]
    cdef i64_t nblock = data.shape[2]
    cdef i64_t nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
    cdef i64_t i
    cdef i64_t j
    cdef i64_t k
    cdef i64_t m
    cdef i64_t off

    if cond.shape[2] != 1:
        raise RuntimeError("condition number map should have one non-zero per pixel")
    if cond.shape[1] != npix:
        raise RuntimeError("condition number map should have the same number of pixels as covariance")
    if cond.shape[0] != nsubmap:
        raise RuntimeError("condition number map should have the same number of submaps as covariance")

    cdef int lwork = (nnz + 2) * nnz
    cdef double * fdata = <double*>malloc(nnz*nnz*sizeof(double))
    cdef double * evals = <double*>malloc(nnz*sizeof(double))
    cdef double * work = <double*>malloc(lwork*sizeof(double))
    cdef int fnnz = nnz
    cdef double norm
    cdef double rcond
    cdef int info
    cdef double inverse
    cdef char uplo = 'L'
    cdef char jobz = 'V'
    cdef double emin
    cdef double emax

    if nnz == 1:
        # shortcut
        for i in range(nsubmap):
            for j in range(npix):
                cond[i,j,0] = 1.0
    else:
        for i in range(nsubmap):
            for j in range(npix):
                # copy to fortran compatible buffer
                off = 0
                memset(fdata, 0, nnz*nnz*sizeof(f64_t))
                memset(evals, 0, nnz*sizeof(f64_t))
                for k in range(nnz):
                    for m in range(k, nnz):
                        fdata[k*nnz+m] = data[i,j,off]
                        if k != m:
                            fdata[m*nnz+k] = data[i,j,off]
                        off += 1

                # eigendecomposition

                cython_lapack.dsyev(&jobz, &uplo, &fnnz, fdata, &fnnz, evals, work, &lwork, &info)

                if info == 0:
                    emin = 1.0e100
                    emax = 0
                    for t in range(nnz):
                        if evals[t] > emax:
                            emax = evals[t]
                        if evals[t] < emin:
                            emin = evals[t]
                    cond[i,j,0] = emin / emax
                else:
                    cond[i,j,0] = 0.0

    free(fdata)
    free(work)
    free(evals)
    return
