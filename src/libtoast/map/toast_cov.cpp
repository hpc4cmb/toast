/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_map_internal.hpp>


// void toast::map::accumulate_diagonal ( int64_t nsamp
//         np.ndarray[f64_t, ndim=3] zmap, 
//         int do_zmap, 
//         np.ndarray[i64_t, ndim=3] hits, 
//         int do_hits, 
//         np.ndarray[f64_t, ndim=3] invnpp, 
//         int do_invnpp,
//         np.ndarray[f64_t, ndim=1] signal,
//         np.ndarray[i64_t, ndim=1] submap_indx, 
//         np.ndarray[i64_t, ndim=1] pix_indx, 
//         np.ndarray[f64_t, ndim=2] weights, 
//         f64_t scale
//     ):
//     '''
//     For a vector of pointing weights, build and accumulate the diagonal
//     inverse noise covariance, the hit map, and the noise weighted map.
//     '''
//     cdef i64_t nsamp = weights.shape[0]
//     cdef i64_t nnz = weights.shape[1]
//     cdef i64_t nblock = int(nnz * (nnz+1) / 2)
//     cdef i64_t i
//     cdef i64_t elem
//     cdef i64_t alt
//     cdef i64_t off
//     cdef f64_t zsig = 0

//     if submap_indx.shape[0] != nsamp:
//         raise RuntimeError("submap index list does not have same length as weights")
//     if pix_indx.shape[0] != nsamp:
//         raise RuntimeError("pixel index list does not have same length as weights")

//     if (do_zmap != 0) and (zmap.shape[2] != nnz):
//         raise RuntimeError("noise weighted map does not have same NNZ as weights")

//     if (do_hits != 0) and (hits.shape[2] != 1):
//         raise RuntimeError("hit map does not have NNZ of one")

//     if (do_invnpp != 0) and (invnpp.shape[2] != nblock):
//         raise RuntimeError("inverse covariance does not have correct shape for NNZ from weights")

//     # Here we repeat code slightly, so that we can do a single loop over
//     # samples.

//     if (do_zmap != 0) and (do_hits != 0) and (do_invnpp != 0):

//         for i in range(nsamp):
//             if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
//                 hits[submap_indx[i], pix_indx[i]] += 1
//                 zsig = scale * signal[i]
//                 off = 0
//                 for elem in range(nnz):
//                     zmap[submap_indx[i], pix_indx[i], elem] += zsig * weights[i,elem]
//                     for alt in range(elem, nnz):
//                         invnpp[submap_indx[i], pix_indx[i], off] += scale * weights[i,elem] * weights[i,alt]
//                         off += 1

//     elif (do_zmap != 0) and (do_hits != 0):

//         for i in range(nsamp):
//             if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
//                 hits[submap_indx[i], pix_indx[i]] += 1
//                 zsig = scale * signal[i]
//                 for elem in range(nnz):
//                     zmap[submap_indx[i], pix_indx[i], elem] += zsig * weights[i,elem]

//     elif (do_hits != 0) and (do_invnpp != 0):

//         for i in range(nsamp):
//             if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
//                 hits[submap_indx[i], pix_indx[i]] += 1
//                 off = 0
//                 for elem in range(nnz):
//                     for alt in range(elem, nnz):
//                         invnpp[submap_indx[i], pix_indx[i], off] += scale * weights[i,elem] * weights[i,alt]
//                         off += 1
    
//     elif (do_zmap != 0) and (do_invnpp != 0):

//         for i in range(nsamp):
//             if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
//                 zsig = scale * signal[i]
//                 off = 0
//                 for elem in range(nnz):
//                     zmap[submap_indx[i], pix_indx[i], elem] += zsig * weights[i,elem]
//                     for alt in range(elem, nnz):
//                         invnpp[submap_indx[i], pix_indx[i], off] += scale * weights[i,elem] * weights[i,alt]
//                         off += 1
    
//     elif (do_zmap != 0):

//         for i in range(nsamp):
//             if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
//                 zsig = scale * signal[i]
//                 for elem in range(nnz):
//                     zmap[submap_indx[i], pix_indx[i], elem] += zsig * weights[i,elem]

//     elif (do_hits != 0):

//         for i in range(nsamp):
//             if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
//                 hits[submap_indx[i], pix_indx[i]] += 1

//     elif (do_invnpp != 0):

//         for i in range(nsamp):
//             if (submap_indx[i] >= 0) and (pix_indx[i] >= 0):
//                 off = 0
//                 for elem in range(nnz):
//                     for alt in range(elem, nnz):
//                         invnpp[submap_indx[i], pix_indx[i], off] += scale * weights[i,elem] * weights[i,alt]
//                         off += 1
//     return


// def _eigendecompose_covariance(np.ndarray[f64_t, ndim=3] data, np.ndarray[f64_t, ndim=3] cond, f64_t threshold, i32_t do_invert, i32_t do_rcond):
//     cdef i64_t nsubmap = data.shape[0]
//     cdef i64_t npix = data.shape[1]
//     cdef i64_t nblock = data.shape[2]
//     cdef i64_t nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
//     cdef i64_t i
//     cdef i64_t j
//     cdef i64_t k
//     cdef i64_t m
//     cdef i64_t off

//     if do_rcond != 0:
//         if cond.shape[2] != 1:
//             raise RuntimeError("condition number map should have one non-zero per pixel")
//         if cond.shape[1] != npix:
//             raise RuntimeError("condition number map should have the same number of pixels as covariance")
//         if cond.shape[0] != nsubmap:
//             raise RuntimeError("condition number map should have the same number of submaps as covariance")

//     if (do_invert == 0) and (do_rcond == 0):
//         # nothing to do!
//         return

//     cdef double * fdata = <double*>malloc(nnz*nnz*sizeof(double))
//     cdef double * ftemp = <double*>malloc(nnz*nnz*sizeof(double))
//     cdef double * finv = <double*>malloc(nnz*nnz*sizeof(double))
//     cdef double * evals = <double*>malloc(nnz*sizeof(double))
    
//     # we assume a large value here, since the work space needed
//     # will still be small.
//     cdef int NB = 256
    
//     cdef int lwork = NB * 2 + nnz
//     cdef double * work = <double*>malloc(lwork*sizeof(double))
//     cdef int fnnz = nnz

//     cdef double emin
//     cdef double emax
//     cdef double rcond
    
//     cdef int info
//     cdef double fzero = 0.0
//     cdef double fone = 1.0
    
//     cdef char jobz_vec = 'V'
//     cdef char jobz_val = 'N'
//     cdef char uplo = 'L'
//     cdef char transN = 'N'
//     cdef char transT = 'T'

//     if nnz == 1:
//         # shortcut
//         if do_invert == 0:
//             cond[:,:,:] = data[:,:,:]
//         else:
//             if do_rcond == 0:
//                 for i in range(nsubmap):
//                     for j in range(npix):
//                         if data[i,j,0] != 0:
//                             data[i,j,0] = 1.0 / data[i,j,0]
//             else: 
//                 for i in range(nsubmap):
//                     for j in range(npix):
//                         cond[i,j,0] = data[i,j,0]
//                         if data[i,j,0] != 0:
//                             data[i,j,0] = 1.0 / data[i,j,0]
//     else:
//         for i in range(nsubmap):
//             for j in range(npix):
//                 # copy to fortran compatible buffer
//                 off = 0
//                 memset(fdata, 0, nnz*nnz*sizeof(f64_t))
//                 for k in range(nnz):
//                     for m in range(k, nnz):
//                         fdata[k*nnz+m] = data[i,j,off]
//                         off += 1

//                 # eigendecompose
//                 if do_invert == 0:
//                     cython_lapack.dsyev(&jobz_val, &uplo, &fnnz, fdata, &fnnz, evals, work, &lwork, &info)
//                 else:
//                     cython_lapack.dsyev(&jobz_vec, &uplo, &fnnz, fdata, &fnnz, evals, work, &lwork, &info)

//                 rcond = 0.0

//                 if info == 0:
//                     # it worked, compute condition number
//                     emin = 1.0e100
//                     emax = 0.0
//                     for k in range(nnz):
//                         if evals[k] < emin:
//                             emin = evals[k]
//                         if evals[k] > emax:
//                             emax = evals[k]
//                     if emax > 0.0:
//                         rcond = emin / emax

//                     # compare to threshold
//                     if rcond >= threshold:
//                         if do_invert != 0:
//                             for k in range(nnz):
//                                 evals[k] = 1.0 / evals[k]
//                                 for m in range(nnz):
//                                     ftemp[k*nnz+m] = evals[k] * fdata[k*nnz+m]
//                             cython_blas.dgemm(&transN, &transT, &fnnz, &fnnz, &fnnz, &fone, ftemp, &fnnz, fdata, &fnnz, &fzero, finv, &fnnz)

//                             off = 0
//                             for k in range(nnz):
//                                 for m in range(k, nnz):
//                                     data[i,j,off] = finv[k*nnz+m]
//                                     off += 1
//                     else:
//                         # reject this pixel
//                         rcond = 0.0
//                         info = 1

//                 if do_invert != 0:
//                     if info != 0:
//                         data[i,j,:] = 0.0
//                 if do_rcond != 0:
//                     cond[i,j,0] = rcond

//     free(fdata)
//     free(ftemp)
//     free(finv)
//     free(evals)
//     free(work)
//     return


// def _multiply_covariance(np.ndarray[f64_t, ndim=3] data1, np.ndarray[f64_t, ndim=3] data2):
//     cdef i64_t nsubmap = data1.shape[0]
//     cdef i64_t npix = data1.shape[1]
//     cdef i64_t nblock = data1.shape[2]
//     cdef i64_t nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
//     cdef i64_t i
//     cdef i64_t j
//     cdef i64_t k
//     cdef i64_t m
//     cdef i64_t off

//     cdef double * fdata1 = <double*>malloc(nnz*nnz*sizeof(double))
//     cdef double * fdata2 = <double*>malloc(nnz*nnz*sizeof(double))
//     cdef double * fdata3 = <double*>malloc(nnz*nnz*sizeof(double))
//     cdef int fnnz = nnz
//     cdef double fone = 1
//     cdef double fzero = 0
//     cdef char side = 'L'
//     cdef char uplo = 'L'

//     if nnz == 1:
//         # shortcut
//         for i in range(nsubmap):
//             for j in range(npix):
//                 data1[i,j,0] *= data2[i,j,0]
//     else:
//         for i in range(nsubmap):
//             for j in range(npix):
//                 # copy to fortran compatible buffer
//                 memset(fdata1, 0, nnz*nnz*sizeof(f64_t))
//                 memset(fdata2, 0, nnz*nnz*sizeof(f64_t))
//                 memset(fdata3, 0, nnz*nnz*sizeof(f64_t))
//                 off = 0
//                 for k in range(nnz):
//                     for m in range(k, nnz):
//                         fdata1[k*nnz+m] = data1[i,j,off]
//                         fdata2[k*nnz+m] = data2[i,j,off]
//                         if k != m:
//                             # Second argument to dsymm must be full
//                             fdata2[m*nnz+k] = data2[i,j,off]
//                         off += 1
                        
//                 cython_blas.dsymm(&side, &uplo, &fnnz, &fnnz, &fone, fdata1, &fnnz, fdata2, &fnnz, &fzero, fdata3, &fnnz)

//                 off = 0
//                 for k in range(nnz):
//                     for m in range(k, nnz):
//                         data1[i,j,off] = fdata3[k*nnz+m]
//                         off += 1
//     free(fdata1)
//     free(fdata2)
//     free(fdata3)
//     return


// def _apply_covariance(np.ndarray[f64_t, ndim=3] cov, np.ndarray[f64_t, ndim=3] mdata):
//     cdef i64_t nsubmap = cov.shape[0]
//     cdef i64_t npix = cov.shape[1]
//     cdef i64_t nblock = cov.shape[2]
//     cdef i64_t nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
//     cdef i64_t i
//     cdef i64_t j
//     cdef i64_t k
//     cdef i64_t m
//     cdef i64_t x
//     cdef np.ndarray[f64_t, ndim=1] tempval = np.zeros(nnz, dtype=f64)

//     # we do this manually now, but could use dsymv if needed...
//     for i in range(nsubmap):
//         for j in range(npix):
//             x = 0
//             tempval.fill(0.0)
//             for k in range(nnz):
//                 for m in range(k, nnz):
//                     tempval[k] += cov[i,j,x] * mdata[i,j,m]
//                     if m != k:
//                         tempval[m] += cov[i,j,x] * mdata[i,j,k]
//                     x += 1
//             for k in range(nnz):
//                 mdata[i,j,k] = tempval[k]
//     return


// def _cond_covariance(np.ndarray[f64_t, ndim=3] data, np.ndarray[f64_t, ndim=3] cond):
//     cdef i64_t nsubmap = data.shape[0]
//     cdef i64_t npix = data.shape[1]
//     cdef i64_t nblock = data.shape[2]
//     cdef i64_t nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )
//     cdef i64_t i
//     cdef i64_t j
//     cdef i64_t k
//     cdef i64_t m
//     cdef i64_t off

//     if cond.shape[2] != 1:
//         raise RuntimeError("condition number map should have one non-zero per pixel")
//     if cond.shape[1] != npix:
//         raise RuntimeError("condition number map should have the same number of pixels as covariance")
//     if cond.shape[0] != nsubmap:
//         raise RuntimeError("condition number map should have the same number of submaps as covariance")

//     cdef int lwork = (nnz + 2) * nnz
//     cdef double * fdata = <double*>malloc(nnz*nnz*sizeof(double))
//     cdef double * evals = <double*>malloc(nnz*sizeof(double))
//     cdef double * work = <double*>malloc(lwork*sizeof(double))
//     cdef int fnnz = nnz
//     cdef double norm
//     cdef double rcond
//     cdef int info
//     cdef double inverse
//     cdef char uplo = 'L'
//     cdef char jobz = 'N'
//     cdef double emin
//     cdef double emax

//     if nnz == 1:
//         # shortcut
//         for i in range(nsubmap):
//             for j in range(npix):
//                 cond[i,j,0] = 1.0
//     else:
//         for i in range(nsubmap):
//             for j in range(npix):
//                 # copy to fortran compatible buffer
//                 off = 0
//                 memset(fdata, 0, nnz*nnz*sizeof(f64_t))
//                 memset(evals, 0, nnz*sizeof(f64_t))
//                 for k in range(nnz):
//                     for m in range(k, nnz):
//                         fdata[k*nnz+m] = data[i,j,off]
//                         if k != m:
//                             fdata[m*nnz+k] = data[i,j,off]
//                         off += 1

//                 # eigendecomposition

//                 cython_lapack.dsyev(&jobz, &uplo, &fnnz, fdata, &fnnz, evals, work, &lwork, &info)

//                 if info == 0:
//                     emin = 1.0e100
//                     emax = 0
//                     for t in range(nnz):
//                         if evals[t] > emax:
//                             emax = evals[t]
//                         if evals[t] < emin:
//                             emin = evals[t]
//                     if emax > 0:
//                         cond[i,j,0] = emin / emax
//                     else:
//                         cond[i,j,0] = 0.0
//                 else:
//                     cond[i,j,0] = 0.0

//     free(fdata)
//     free(work)
//     free(evals)
//     return



