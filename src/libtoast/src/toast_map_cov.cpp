
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
                           int64_t * hits, double * invnpp)
{
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
                                int64_t const * indx_pix, int64_t * hits)
{
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
                                  double * invnpp)
{
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
                                       double * invnpp)
{
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
                           double * zdata)
{
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
                                    double threshold, bool invert)
{
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
        } else   {
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
            } else   {
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
    } else   {
        // problem size parameters
        int batchNumber = nsub * subsize;
        int64_t blockSize = (int64_t)(nnz * (nnz + 1) / 2);

        // solver parameters
        char jobz = (invert) ? 'V' : 'N';
        char uplo = 'L';
        char transN = 'N';
        char transT = 'T';

        // fdata = data (reordering)
        toast::AlignedVector <double> fdata_batch(batchNumber * nnz * nnz);
#pragma omp parallel for schedule(static)
        for (int64_t batchid = 0; batchid < batchNumber; batchid++) {
            int offset = 0;
            for (int64_t k = 0; k < nnz; k++) {
                // zero half matrix
                for (int64_t m = 0; m < k; m++) {
                    fdata_batch[batchid * nnz * nnz + k * nnz + m] = 0.;
                }

                // copies other half matrix
                for (int64_t m = k; m < nnz; m++) {
                    fdata_batch[batchid * nnz * nnz + k * nnz +
                                m] = data[batchid * blockSize + offset];
                    offset += 1;
                }
            }
        }

        // compute eigenvalues of fdata (stored in evals)
        // and, potentially, eigenvectors (which are then stored in fdata)
        toast::AlignedVector <int> info_batch(batchNumber);
        toast::AlignedVector <double> evals_batch(batchNumber * nnz);
        toast::LinearAlgebra::syev_batched(jobz, uplo, nnz,
                                           fdata_batch.data(), nnz,
                                           evals_batch.data(), info_batch.data(),
                                           batchNumber);

        // compute condition number as the ratio of the eigenvalues
        // and sets ftemp = fdata / evals
        toast::AlignedVector <double> rcond_batch(batchNumber);
        toast::AlignedVector <double> ftemp_batch(batchNumber * nnz * nnz);
#pragma omp parallel for schedule(static)
        for (int64_t batchid = 0; batchid < batchNumber; batchid++) {
            double emin = 1.0e100;
            double emax = 0.0;
            for (int64_t k = batchid * nnz; k < (batchid + 1) * nnz; k++) {
                // computes the maximum and minimum eigenvalues
                if (evals_batch[k] < emin) emin = evals_batch[k];
                if (evals_batch[k] > emax) emax = evals_batch[k];

                // ftemp = fdata / evals (eigenvectors divided by eigenvalues)
                for (int64_t m = 0; m < nnz; m++) {
                    ftemp_batch[k * nnz + m] = fdata_batch[k * nnz + m] /
                                               evals_batch[k];
                }
            }

            // stores the condition number
            rcond_batch[batchid] = (emax > 0.0) ? (emin / emax) : 0.;
        }

        // finv = ftemp x fdata
        toast::AlignedVector <double> finv_batch(batchNumber * nnz * nnz);
        if (invert) {
            double fzero = 0.0;
            double fone = 1.0;
            toast::LinearAlgebra::gemm_batched(transN, transT, nnz, nnz,
                                               nnz, fone, ftemp_batch.data(),
                                               nnz, fdata_batch.data(), nnz,
                                               fzero,
                                               finv_batch.data(), nnz, batchNumber);
        }

        // data = finv
#pragma omp parallel for schedule(static)
        for (int64_t batchid = 0; batchid < batchNumber; batchid++) {
            // did the computation of finv succeed
            const bool success =
                (info_batch[batchid] == 0) and (rcond_batch[batchid] >= threshold);

            // stores result in data
            if (invert) {
                if (success) {
                    // data = finv (reordering)
                    int offset = 0;
                    for (int64_t k = 0; k < nnz; k++) {
                        for (int64_t m = k; m < nnz; m++) {
                            data[batchid * blockSize +
                                 offset] =
                                finv_batch[batchid * nnz * nnz + k * nnz + m];
                            offset += 1;
                        }
                    }
                } else   {
                    // data = 0.
                    for (int64_t k = batchid * blockSize; k < (batchid + 1) * blockSize;
                         k++) {
                        data[k] = 0.;
                    }
                }
            }

            // if we have an output parameter for the condition number
            if (cond != NULL) {
                cond[batchid] = (success) ? rcond_batch[batchid] : 0.;
            }
        }
    }
}

void toast::cov_mult_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                          double * data1, double const * data2)
{
    if (nnz == 1) {
        // shortcut for NNZ == 1
        int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
        for (int64_t i = 0; i < nsub; ++i) {
            for (int64_t j = 0; j < subsize; ++j) {
                int64_t px = (i * subsize * block) + (j * block);
                data1[px] *= data2[px];
            }
        }
    } else   {
        // problem size
        int batchNumber = nsub * subsize;
        int64_t blockSize = nnz * (nnz + 1) / 2;

        // temporary buffer for the data
        toast::AlignedVector <double> fdata1(batchNumber * nnz * nnz);
        toast::AlignedVector <double> fdata2(batchNumber * nnz * nnz);
        toast::AlignedVector <double> fdata3(batchNumber * nnz * nnz);

        // copy data to buffers
#pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < batchNumber; b++) {
            // zero out data
            std::fill(fdata1.begin() + b * (nnz * nnz),
                      fdata1.begin() + (b + 1) * (nnz * nnz), 0.0);
            std::fill(fdata2.begin() + b * (nnz * nnz),
                      fdata2.begin() + (b + 1) * (nnz * nnz), 0.0);

            // copies inputs and reshape them for the upcoming computation
            int64_t offset1 = 0;
            for (int64_t k = 0; k < nnz; k++) {
                for (int64_t m = k; m < nnz; m++) {
                    fdata1[k * nnz + m] = data1[b * blockSize + offset1];
                    fdata2[k * nnz + m] = data2[b * blockSize + offset1];
                    if (k != m) {
                        // Second argument to dsymm must be full
                        fdata2[m * nnz + k] = data2[b * blockSize + offset1];
                    }
                    offset1 += 1;
                }
            }
        }

        // batched symmetric matrix product
        const double fzero = 0.0;
        const double fone = 1.0;
        const char uplo = 'L';
        const char side = 'L';
        toast::LinearAlgebra::symm_batched(side, uplo, nnz, nnz, fone,
                                           fdata1.data(), nnz, fdata2.data(), nnz,
                                           fzero, fdata3.data(), nnz, batchNumber);

        // copy data back from buffer
#pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < batchNumber; b++) {
            int64_t offset2 = 0;
            for (int64_t k = 0; k < nnz; k++) {
                for (int64_t m = k; m < nnz; m++) {
                    data1[b * blockSize + offset2] = fdata3[k * nnz + m];
                    offset2 += 1;
                }
            }
        }
    }

    return;
}

void toast::cov_apply_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                           double const * mat, double * vec)
{
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
    } else   {
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
