
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_linearalgebra.hpp>
#include <toast/map_cov.hpp>

#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP


void toast::cov_accum_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                           int64_t nsamp,
                           int64_t const * indx_submap,
                           int64_t const * indx_pix, double const * weights,
                           double scale, double const * signal, double * zdata,
                           int64_t * hits, double * invnpp) {
    const int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (size_t i = 0; i < nsamp; ++i) {
            const int64_t isubmap = indx_submap[i] * subsize;
            const int64_t ipix = indx_pix[i];
            if ((isubmap < 0) || (ipix < 0)) continue;

            const int64_t hpx = isubmap + ipix;
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            const int64_t zpx = hpx * nnz;
            const int64_t ipx = hpx * block;

            const double scaled_signal = scale * signal[i];
            double * zpointer = zdata + zpx;
            const double * wpointer = weights + i * nnz;
            double * covpointer = invnpp + ipx;
            for (size_t j = 0; j < nnz; ++j, ++zpointer, ++wpointer) {
                *zpointer += *wpointer * scaled_signal;
                const double scaled_weight = *wpointer * scale;
                const double * wpointer2 = wpointer;
                for (size_t k = j; k < nnz; ++k, ++wpointer2, ++covpointer) {
                    *covpointer += *wpointer2 * scaled_weight;
                }
            }

            hits[hpx] += 1;
        }
    }

    return;
}

void toast::cov_accum_diag_hits(int64_t nsub, int64_t subsize, int64_t nnz,
                                int64_t nsamp,
                                int64_t const * indx_submap,
                                int64_t const * indx_pix, int64_t * hits) {
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (size_t i = 0; i < nsamp; ++i) {
            if ((indx_submap[i] < 0) || (indx_pix[i] < 0)) continue;

            const int64_t hpx = (indx_submap[i] * subsize) + indx_pix[i];
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            hits[hpx] += 1;
        }
    }

    return;
}

void toast::cov_accum_diag_invnpp(int64_t nsub, int64_t subsize, int64_t nnz,
                                  int64_t nsamp,
                                  int64_t const * indx_submap,
                                  int64_t const * indx_pix,
                                  double const * weights,
                                  double scale,
                                  double * invnpp) {
    const int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (size_t i = 0; i < nsamp; ++i) {
            const int64_t isubmap = indx_submap[i] * subsize;
            const int64_t ipix = indx_pix[i];
            if ((isubmap < 0) || (ipix < 0)) continue;

            const int64_t hpx = isubmap + ipix;
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            const int64_t ipx = hpx * block;

            const double * wpointer = weights + i * nnz;
            double * covpointer = invnpp + ipx;
            for (size_t j = 0; j < nnz; ++j, ++wpointer) {
                const double scaled_weight = *wpointer * scale;
                const double * wpointer2 = wpointer;
                for (size_t k = j; k < nnz; ++k, ++wpointer2, ++covpointer) {
                    *covpointer += *wpointer2 * scaled_weight;
                }
            }

            // std::cout << "Accum to local pixel " << hpx << " with scale " << scale <<
            // ":" << std::endl;
            // for (size_t j = 0; j < nnz; ++j) {
            //     std::cout << " " << weights[i * nnz + j];
            // }
            // std::cout << std::endl;
            // for (size_t j = 0; j < block; ++j) {
            //     std::cout << " " << invnpp[ipx + j];
            // }
            // std::cout << std::endl;
        }
    }

    return;
}

void toast::cov_accum_diag_invnpp_hits(int64_t nsub, int64_t subsize, int64_t nnz,
                                       int64_t nsamp,
                                       int64_t const * indx_submap,
                                       int64_t const * indx_pix,
                                       double const * weights,
                                       double scale, int64_t * hits,
                                       double * invnpp) {
    const int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (size_t i = 0; i < nsamp; ++i) {
            const int64_t isubmap = indx_submap[i] * subsize;
            const int64_t ipix = indx_pix[i];
            if ((isubmap < 0) || (ipix < 0)) continue;

            const int64_t hpx = isubmap + ipix;
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            const int64_t ipx = hpx * block;

            const double * wpointer = weights + i * nnz;
            double * covpointer = invnpp + ipx;
            for (size_t j = 0; j < nnz; ++j, ++wpointer) {
                const double scaled_weight = *wpointer * scale;
                const double * wpointer2 = wpointer;
                for (size_t k = j; k < nnz; ++k, ++wpointer2, ++covpointer) {
                    *covpointer += *wpointer2 * scaled_weight;
                }
            }

            hits[hpx] += 1;
        }
    }

    return;
}

void toast::cov_accum_zmap(int64_t nsub, int64_t subsize, int64_t nnz,
                           int64_t nsamp,
                           int64_t const * indx_submap,
                           int64_t const * indx_pix, double const * weights,
                           double scale, double const * signal,
                           double * zdata) {
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (int64_t i = 0; i < nsamp; ++i) {
            const int64_t isubmap = indx_submap[i] * subsize;
            const int64_t ipix = indx_pix[i];
            if ((isubmap < 0) || (ipix < 0)) continue;

            const int64_t hpx = isubmap + ipix;
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            const int64_t zpx = hpx * nnz;

            const double scaled_signal = scale * signal[i];
            double * zpointer = zdata + zpx;
            const double * wpointer = weights + i * nnz;
            for (int64_t j = 0; j < nnz; ++j, ++zpointer, ++wpointer) {
                *zpointer += *wpointer * scaled_signal;
            }
        }
    }

    return;
}

void toast::cov_eigendecompose_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                                    double * data, double * cond,
                                    double threshold, bool invert) {
    if (nnz == 1) {
        // shortcut for NNZ == 1
        if (!invert) {
            // Not much point in calling this!
            if (cond != NULL) {
                for (int64_t i = 0; i < nsub; ++i) {
                    for (int64_t j = 0; j < subsize; ++j) {
                        cond[i * subsize + j] = 1.0;
                    }
                }
            }
        } else {
            if (cond != NULL) {
                for (int64_t i = 0; i < nsub; ++i) {
                    for (int64_t j = 0; j < subsize; ++j) {
                        int64_t dpx = (i * subsize) + j;
                        cond[dpx] = 1.0;
                        if (data[dpx] != 0) {
                            data[dpx] = 1.0 / data[dpx];
                        }
                    }
                }
            } else {
                for (int64_t i = 0; i < nsub; ++i) {
                    for (int64_t j = 0; j < subsize; ++j) {
                        int64_t dpx = (i * subsize) + j;
                        if (data[dpx] != 0) {
                            data[dpx] = 1.0 / data[dpx];
                        }
                    }
                }
            }
        }
    }
    else
    {
        // problem size parameters
        int fnnz = (int)nnz;
        int batchNumber = nsub * subsize;
        int64_t blockSize = (int64_t)(nnz * (nnz + 1) / 2);

        // solver parameters
        toast::LinearAlgebra linearAlgebra;
        char jobz = (invert) ? 'V' : 'N';
        char uplo = 'L';
        char transN = 'N';
        char transT = 'T';
        
        // fdata = data (reordering)
        toast::AlignedVector <double> fdata_batch(batchNumber * nnz * nnz);
        #pragma omp parallel for
        for(int64_t batchid = 0; batchid < batchNumber; batchid++)
        {
            int offset = 0;
            for (int64_t k = 0; k < nnz; k++)
            {
                // zero half matrix
                for (int64_t m = 0; m < k; m++)
                {
                    fdata_batch[batchid*nnz*nnz + k*nnz + m] = 0.;
                }
                // copies other half matrix
                for (int64_t m = k; m < nnz; m++)
                {
                    fdata_batch[batchid*nnz*nnz + k*nnz + m] = data[batchid*blockSize + offset];
                    offset += 1;
                }
            }
        }

        // compute eigenvalues of fdata (stored in evals)
        // and, potentially, eigenvectors (which are then stored in fdata)
        int info;
        toast::AlignedVector <double> evals_batch(batchNumber * nnz);
        linearAlgebra.syev_batched(jobz, uplo, fnnz, fdata_batch.data(), fnnz, evals_batch.data(), &info, batchNumber);

        // compute condition number as the ratio of the eigenvalues
        // and sets ftemp = fdata / evals
        toast::AlignedVector <double> rcond_batch(batchNumber);
        toast::AlignedVector <double> ftemp_batch(batchNumber * nnz * nnz);
        #pragma omp parallel for
        for(int64_t batchid = 0; batchid < batchNumber; batchid++)
        {
            double emin = 1.0e100;
            double emax = 0.0;
            for (int64_t k = batchid*nnz; k < (batchid+1)*nnz; k++)
            {
                // computes the maximum and minimum eigenvalues
                if (evals_batch[k] < emin) emin = evals_batch[k];
                if (evals_batch[k] > emax) emax = evals_batch[k];
                // ftemp = fdata / evals (eigenvectors divided by eigenvalues)
                for (int64_t m = 0; m < nnz; m++)
                {
                    ftemp_batch[k * nnz + m] = fdata_batch[k * nnz + m] / evals_batch[k];
                }
            }
            // stores the condition number
            rcond_batch[batchid] = (emax > 0.0) ? (emin / emax) : 0.;
        }

        // finv = ftemp x fdata
        toast::AlignedVector <double> finv_batch(batchNumber * nnz * nnz);
        if(invert)
        {
            double fzero = 0.0;
            double fone = 1.0;
            linearAlgebra.gemm_batched(transN, transT, fnnz, fnnz,
                                       fnnz, fone, ftemp_batch.data(),
                                       fnnz, fdata_batch.data(), fnnz,
                                       fzero, finv_batch.data(), fnnz, batchNumber);
        }

        // data = finv
        #pragma omp parallel for
        for(int64_t batchid = 0; batchid < batchNumber; batchid++)
        {
            // did the computation of finv succeed
            const bool success = (info == 0) and (rcond_batch[batchid] >= threshold);
            // stores result in data
            if(invert)
            {
                if(success)
                {
                    // data = finv (reordering)
                    int offset = 0;
                    for (int64_t k = 0; k < nnz; k++)
                    {
                        for (int64_t m = k; m < nnz; m++)
                        {
                            data[batchid*blockSize + offset] = finv_batch[batchid*nnz*nnz + k*nnz + m];
                            offset += 1;
                        }
                    }
                }
                else
                {
                    // data = 0.
                    for (int64_t k = batchid*blockSize; k < (batchid+1)*blockSize; k++)
                    {
                        data[k] = 0.;
                    }
                }
            }
            // if we have an output parameter for the condition number
            if (cond != NULL)
            {
                cond[batchid] = (success) ? rcond_batch[batchid] : 0.;
            }
        }
    }
}

void toast::cov_mult_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                          double * data1, double const * data2) {
    if (nnz == 1) {
        // shortcut for NNZ == 1
        int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
        for (int64_t i = 0; i < nsub; ++i) {
            for (int64_t j = 0; j < subsize; ++j) {
                int64_t px = (i * subsize * block) + (j * block);
                data1[px] *= data2[px];
            }
        }
    } else {
        // Even if the actual BLAS/LAPACK library is threaded, these are very
        // small matrices.  So instead we divide up the map data across threads
        // and each thread does some large number of small eigenvalue problems.

        #pragma \
        omp parallel default(none) shared(nsub, subsize, nnz, data1, data2)
        {
            // thread-private variables
            toast::LinearAlgebra linearAlgebra;

            int fnnz = (int)nnz;

            double fzero = 0.0;
            double fone = 1.0;

            char uplo = 'L';
            char side = 'L';

            int64_t block = (int64_t)(nnz * (nnz + 1) / 2);

            int64_t off;

            toast::AlignedVector <double> fdata1(nnz * nnz);
            toast::AlignedVector <double> fdata2(nnz * nnz);
            toast::AlignedVector <double> fdata3(nnz * nnz);

            // Here we "unroll" the loop over submaps and pixels within each
            // submap.
            // This allows us to distribute the total pixels across all
            // threads.

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < (nsub * subsize); ++i) {
                int64_t px = i * block;

                // copy to fortran buffer

                std::fill(fdata1.begin(), fdata1.end(), 0);
                std::fill(fdata2.begin(), fdata2.end(), 0);
                std::fill(fdata3.begin(), fdata3.end(), 0);

                off = 0;
                for (int64_t k = 0; k < nnz; ++k) {
                    for (int64_t m = k; m < nnz; ++m) {
                        fdata1[k * nnz + m] = data1[px + off];
                        fdata2[k * nnz + m] = data2[px + off];
                        if (k != m) {
                            // Second argument to dsymm must be full
                            fdata2[m * nnz + k] = data2[px + off];
                        }
                        off += 1;
                    }
                }

                linearAlgebra.symm(side, uplo, fnnz, fnnz, fone,
                                   fdata1.data(), fnnz, fdata2.data(), fnnz,
                                   fzero, fdata3.data(), fnnz);

                off = 0;
                for (int64_t k = 0; k < nnz; ++k) {
                    for (int64_t m = k; m < nnz; ++m) {
                        data1[px + off] = fdata3[k * nnz + m];
                        off += 1;
                    }
                }
            }
        }
    }

    return;
}

void toast::cov_apply_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                           double const * mat, double * vec) {
    int64_t i, j, k;
    int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
    int64_t mpx;
    int64_t vpx;

    if (nnz == 1) {
        // shortcut for NNZ == 1

        for (i = 0; i < nsub; ++i) {
            for (j = 0; j < subsize; ++j) {
                mpx = (i * subsize * block) + (j * block);
                vpx = (i * subsize * nnz) + (j * nnz);
                vec[vpx] *= mat[mpx];
            }
        }
    } else {
        // We do this manually now, but could use dsymv if needed...
        // Since this is just multiply / add operations, the overhead of
        // threading
        // is likely more than the savings.

        int64_t m;
        int64_t off;

        toast::AlignedVector <double> temp(nnz);

        for (i = 0; i < nsub; ++i) {
            for (j = 0; j < subsize; ++j) {
                mpx = (i * subsize * block) + (j * block);
                vpx = (i * subsize * nnz) + (j * nnz);

                std::fill(temp.begin(), temp.end(), 0);

                off = 0;
                for (k = 0; k < nnz; ++k) {
                    for (m = k; m < nnz; ++m) {
                        temp[k] += mat[mpx + off] * vec[vpx + m];
                        if (m != k) {
                            temp[m] += mat[mpx + off] * vec[vpx + k];
                        }
                        off++;
                    }
                }

                for (k = 0; k < nnz; ++k) {
                    vec[vpx + k] = temp[k];
                }
            }
        }
    }

    return;
}
