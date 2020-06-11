
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <string.h>
#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP
#include <toast/sys_utils.hpp>
#include <toast/math_lapack.hpp>
#include <toast/tod_filter.hpp>


void toast::filter_polynomial(int64_t order, size_t n, uint8_t * flags,
                              std::vector <double *> const & signals, size_t nscan,
                              int64_t const * starts, int64_t const * stops) {
    // Process the signals, one subscan at a time.  There is only one
    // flag vector, because the flags must be identical to apply the
    // same template matrix.

    if (order < 0) return;

    int nsignal = signals.size();
    int norder = order + 1;

    char upper = 'U';
    char lower = 'L';
    char notrans = 'N';
    char trans = 'T';
    double fzero = 0.0;
    double fone = 1.0;

    #pragma                                                  \
    omp parallel default(none)                               \
    shared(order, signals, flags, n, nsignal, starts, stops, \
    nscan, norder, upper, lower, notrans, trans, fzero, fone)
    {
        #pragma omp for schedule(static)
        for (size_t iscan = 0; iscan < nscan; ++iscan) {
            int64_t start = starts[iscan];
            int64_t stop = stops[iscan];
            if (start < 0) start = 0;
            if (stop > n - 1) stop = n - 1;
            if (stop < start) continue;
            int scanlen = stop - start + 1;

            int ngood = 0;
            for (size_t i = 0; i < scanlen; ++i) {
                if (flags[start + i] == 0) ngood++;
            }
            if (ngood == 0) continue;

            // Build the full template matrix used to clean the signal.
            // We subtract the template value even from flagged samples to
            // support point source masking etc.

            toast::AlignedVector <double> full_templates(scanlen * norder);

            double dx = 2. / scanlen;
            double xstart = 0.5 * dx - 1;
            double * current, * last, * lastlast;

            for (size_t iorder = 0; iorder < norder; ++iorder) {
                current = &full_templates[iorder * scanlen];
                if (iorder == 0) {
                    #pragma omp simd
                    for (size_t i = 0; i < scanlen; ++i) current[i] = 1;
                } else if (iorder == 1) {
                    #pragma omp simd
                    for (size_t i = 0; i < scanlen; ++i) {
                        const double x = xstart + i * dx;
                        current[i] = x;
                    }
                } else {
                    last = &full_templates[(iorder - 1) * scanlen];
                    lastlast = &full_templates[(iorder - 2) * scanlen];
                    double orderinv = 1. / iorder;
                    #pragma omp simd
                    for (size_t i = 0; i < scanlen; ++i) {
                        const double x = xstart + i * dx;
                        current[i] =
                            ((2 * iorder - 1) * x * last[i] - (iorder - 1) *
                             lastlast[i]) *
                            orderinv;
                    }
                }
            }

            // Assemble the flagged template matrix used in the linear
            // regression

            toast::AlignedVector <double> masked_templates(ngood * norder);

            for (size_t iorder = 0; iorder < norder; ++iorder) {
                size_t offset = iorder * ngood;
                current = &full_templates[iorder * scanlen];
                for (size_t i = 0; i < scanlen; ++i) {
                    if (flags[start + i] != 0) continue;
                    masked_templates[offset++] = current[i];
                }
            }

            // Square the template matrix for A^T.A

            toast::AlignedVector <double> invcov(norder * norder);
            toast::lapack_syrk(&upper, &trans, &norder, &ngood, &fone,
                               masked_templates.data(), &ngood, &fzero, invcov.data(),
                               &norder);

            // Project the signals against the templates

            toast::AlignedVector <double> masked_signals(ngood * nsignal);

            for (size_t isignal = 0; isignal < nsignal; ++isignal) {
                size_t offset = isignal * ngood;
                double * signal = signals[isignal] + start;
                for (int64_t i = 0; i < scanlen; ++i) {
                    if (flags[start + i] != 0) continue;
                    masked_signals[offset++] = signal[i];
                }
            }

            toast::AlignedVector <double> proj(norder * nsignal);

            toast::lapack_gemm(&trans, &notrans, &norder, &nsignal, &ngood,
                               &fone, masked_templates.data(), &ngood,
                               masked_signals.data(), &ngood,
                               &fzero, proj.data(), &norder);

            // Symmetrize the covariance matrix, dgells is written for
            // generic matrices

            for (size_t row = 0; row < norder; ++row) {
                for (size_t col = row + 1; col < norder; ++col) {
                    invcov[col + row * norder] = invcov[row + col * norder];
                }
            }

            // Fit the templates against the data.  DGELSS uses SVD to
            // minimize the norm of the difference and the solution
            // vector.

            toast::AlignedVector <double> singular_values(norder);
            int rank, info;
            double rcond_limit = 1e-3;
            int lwork = std::max(10 * (norder + nsignal), 1000000);
            toast::AlignedVector <double> work(lwork);

            // DGELSS will overwrite proj with the fitting
            // coefficients.  invcov is overwritten with
            // singular vectors.
            toast::lapack_dgelss(&norder, &norder, &nsignal,
                                 invcov.data(), &norder,
                                 proj.data(), &norder,
                                 singular_values.data(), &rcond_limit,
                                 &rank, work.data(), &lwork, &info);

            for (int iorder = 0; iorder < norder; ++iorder) {
                double * temp = &full_templates[iorder * scanlen];
                for (int isignal = 0; isignal < nsignal; ++isignal) {
                    double * signal = &signals[isignal][start];
                    double amp = proj[iorder + isignal * norder];
                    if (toast::is_aligned(signal) && toast::is_aligned(temp)) {
                        #pragma omp simd
                        for (size_t i = 0; i < scanlen; ++i) signal[i] -= amp * temp[i];
                    } else {
                        for (size_t i = 0; i < scanlen; ++i) signal[i] -= amp * temp[i];
                    }
                }
            }
        }
    }

    return;
}

void toast::bin_templates(double * signal, double * templates,
                          uint8_t * good, double * invcov, double * proj,
                          size_t nsample, size_t ntemplate) {
    for (size_t row = 0; row < ntemplate; row++) {
        proj[row] = 0;
        for (size_t col = 0; col < ntemplate; col++) {
            invcov[ntemplate * row + col] = 0;
        }
    }

#pragma omp parallel for \
    schedule(static) default(none) shared(proj, templates, signal, good, ntemplate, nsample)
    for (size_t row = 0; row < ntemplate; ++row) {
        double * ptemplate = templates + row * nsample;
        for (size_t i = 0; i < nsample; ++i) {
            proj[row] += ptemplate[i] * signal[i] * good[i];
        }
    }

#pragma omp parallel \
    default(none) shared(invcov, templates, signal, good, ntemplate, nsample)
    {
        int nthread = 1;
        int id_thread = 0;
        #ifdef _OPENMP
        nthread = omp_get_num_threads();
        id_thread = omp_get_thread_num();
        #endif // ifdef _OPENMP

        int worker = -1;
        for (size_t row = 0; row < ntemplate; row++) {
            for (size_t col = row; col < ntemplate; ++col) {
                ++worker;
                if (worker % nthread == id_thread) {
                    double * rowtemplate = templates + row * nsample;
                    double * coltemplate = templates + col * nsample;
                    double * pcov = invcov + ntemplate * row + col;
                    for (size_t i = 0; i < nsample; ++i) {
                        *pcov += rowtemplate[i] * coltemplate[i] * good[i];
                    }
                    invcov[ntemplate * col + row] = *pcov;
                }
            }
        }
    }

    return;
}

void toast::chebyshev(double * x, double * templates, size_t start_order,
                      size_t stop_order, size_t nsample) {
    // order == 0
    if ((start_order == 0) && (stop_order > 0)) {
        for (size_t i = 0; i < nsample; ++i) templates[i] = 1;
    }

    // order == 1
    if ((start_order <= 1) && (stop_order > 1)) {
        memcpy(templates + (1 - start_order) * nsample, x, nsample * sizeof(double));
    }

    // Calculate the hierarchy of polynomials, one buffer length
    // at a time
    const size_t buflen = 1000;
    size_t nbuf = nsample / buflen + 1;

#pragma omp parallel for schedule(static) default(none) \
    shared(x, templates, start_order, stop_order, nsample, nbuf)
    for (size_t ibuf = 0; ibuf < nbuf; ++ibuf) {
        size_t istart = ibuf * buflen;
        size_t istop = istart + buflen;
        if (istop > nsample) istop = nsample;
        if (istop <= istart) continue;
        size_t n = istop - istart;
        size_t nbyte = n * sizeof(double);

        // Initialize to order = 1
        std::vector <double> val(n);
        memcpy(val.data(), x + istart, nbyte);
        std::vector <double> prev(n, 1);
        std::vector <double> next(n);

        for (size_t order = 2; order < stop_order; ++order) {
            // Evaluate current order and store in val
            for (size_t i = 0; i < n; ++i) {
                next[i] = 2 * x[istart + i] * val[i] - prev[i];
            }
            memcpy(prev.data(), val.data(), nbyte);
            memcpy(val.data(), next.data(), nbyte);
            if (order >= start_order) {
                memcpy(templates + istart + (order - start_order) * nsample,
                       val.data(), nbyte);
            }
        }
    }

    return;
}

void toast::add_templates(double * signal, double * templates, double * coeff,
                          size_t nsample, size_t ntemplate) {
    const size_t buflen = 1000;
    size_t nbuf = nsample / buflen + 1;

#pragma omp parallel for \
    schedule(static) default(none) shared(signal, templates, coeff, nsample, ntemplate, nbuf)
    for (size_t ibuf = 0; ibuf < nbuf; ++ibuf) {
        size_t istart = ibuf * buflen;
        size_t istop = istart + buflen;
        if (istop > nsample) istop = nsample;
        if (istop <= istart) continue;
        size_t n = istop - istart;
        for (size_t itemplate = 0; itemplate < ntemplate; ++itemplate) {
            double * ptemplate = templates + itemplate * nsample;
            double c = coeff[itemplate];
            for (size_t i = istart; i < istop; ++i) {
                signal[i] += c * ptemplate[i];
            }
        }
    }

    return;
}
