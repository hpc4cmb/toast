
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <string.h>
#include <algorithm>

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

void toast::bin_proj(double * signal, double * templates,
                     uint8_t * good, double * proj,
                     size_t nsample, size_t ntemplate) {
    for (size_t row = 0; row < ntemplate; row++) {
        proj[row] = 0;
    }

#pragma omp parallel for \
    schedule(static) default(none) shared(proj, templates, signal, good, ntemplate, nsample)
    for (size_t row = 0; row < ntemplate; ++row) {
        double * ptemplate = templates + row * nsample;
        for (size_t i = 0; i < nsample; ++i) {
            proj[row] += ptemplate[i] * signal[i] * good[i];
        }
    }

    return;
}

void toast::bin_invcov(double * templates, uint8_t * good, double * invcov,
                       size_t nsample, size_t ntemplate) {
    for (size_t row = 0; row < ntemplate; row++) {
        for (size_t col = 0; col < ntemplate; col++) {
            invcov[ntemplate * row + col] = 0;
        }
    }

#pragma omp parallel \
    default(none) shared(invcov, templates, good, ntemplate, nsample)
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
        std::copy(x, x + nsample, templates + (1 - start_order) * nsample);
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

        // Initialize to order = 1
        std::vector <double> val(n);
        std::copy(x + istart, x + istart + n, val.data());
        std::vector <double> prev(n);
        std::fill(prev.begin(), prev.end(), 1.0);
        std::vector <double> next(n);

        for (size_t order = 2; order < stop_order; ++order) {
            // Evaluate current order and store in val
            for (size_t i = 0; i < n; ++i) {
                next[i] = 2 * x[istart + i] * val[i] - prev[i];
            }
            std::copy(val.data(), val.data() + n, prev.data());
            std::copy(next.data(), next.data() + n, val.data());
            if (order >= start_order) {
                std::copy(
                    val.data(),
                    val.data() + n,
                    templates + istart + (order - start_order) * nsample
                    );
            }
        }
    }

    return;
}

void toast::legendre(double * x, double * templates, size_t start_order,
                     size_t stop_order, size_t nsample) {
    // order == 0
    double norm = 1. / sqrt(2);
    if ((start_order == 0) && (stop_order > 0)) {
        for (size_t i = 0; i < nsample; ++i) templates[i] = norm;
    }

    // order == 1
    norm = 1. / sqrt(2. / 3.);
    if ((start_order <= 1) && (stop_order > 1)) {
        double * ptemplates = templates + (1 - start_order) * nsample;
        for (size_t i = 0; i < nsample; ++i) ptemplates[i] = norm * x[i];
    }

    // Calculate the hierarchy of polynomials, one buffer length
    // at a time to allow for parallelization
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

        // Initialize to order = 1
        std::vector <double> val(n);
        std::copy(x + istart, x + istart + n, val.data());
        std::vector <double> prev(n);
        std::fill(prev.begin(), prev.end(), 1.0);
        std::vector <double> next(n);

        for (size_t order = 2; order < stop_order; ++order) {
            // Evaluate current order and store in val
            double orderinv = 1. / order;
            for (size_t i = 0; i < n; ++i) {
                next[i] =
                    ((2 * order - 1) * x[istart + i] * val[i] - (order - 1) *
                     prev[i]) * orderinv;
            }
            std::copy(val.data(), val.data() + n, prev.data());
            std::copy(next.data(), next.data() + n, val.data());
            if (order >= start_order) {
                double * ptemplates = templates + istart + (order - start_order) *
                                      nsample;
                std::copy(val.data(), val.data() + n, ptemplates);

                // Normalize for better condition number
                double norm = 1. / sqrt(2. / (2. * order + 1.));
                for (size_t i = 0; i < n; ++i) {
                    ptemplates[i] *= norm;
                }
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

void toast::filter_poly2D_solve(
    int64_t nsample, int32_t ndet, int32_t ngroup, int32_t nmode,
    int32_t const * det_group, double const * templates, uint8_t const * masks,
    double const * signals, double * coeff
    ) {
    // For each sample, solve for the regression coefficients.
    // The templates are flat packed across (detectors, modes).
    // The mask is flat packed across (samples, detectors).
    // The signals are flat packed across (samples, detectors).
    // The coefficients are flat packed across (samples, groups, modes).

    #pragma \
    omp parallel default(shared)
    {
        // These are all thread-private
        toast::AlignedVector <double> rhs(nmode);
        toast::AlignedVector <double> A(nmode * nmode);
        toast::AlignedVector <double> singular_values(nmode);

        int inmode = (int)nmode;
        int rank;
        int info;
        int one = 1;
        double rcond_limit = 1e-3;
        int lwork = std::max(5 * inmode, 1000000);
        toast::AlignedVector <double> work(lwork);

        #pragma omp for schedule(static)
        for (int64_t isamp = 0; isamp < nsample; ++isamp) {
            // For this sample...
            for (int32_t igroup = 0; igroup < ngroup; ++igroup) {
                // For this group of detectors...
                // Zero out solve buffers
                std::fill(rhs.begin(), rhs.end(), 0.0);
                std::fill(A.begin(), A.end(), 0.0);

                // Accumulate the RHS and design matrix one detector at a time.  Imagine
                // we have 2 detectors and 3 modes:
                //
                //       mask = [m_1 m_2] (These are either 0 or 1, so m_1 * m_1 == m_1)
                //
                //  templates = [[a_1 b_1 c_1],
                //               [a_2 b_2 c_2]]
                //
                //     signal = [s_1 s_2]
                //
                //        RHS =  (mask * templates)^T  X  (mask * signals^T)
                //            =  [[a_1 * s_1 * m_1 + a_2 * s_2 * m_2],
                //                [b_1 * s_1 * m_1 + b_2 * s_2 * m_2],
                //                [c_1 * s_1 * m_1 + c_2 * s_2 * m_2]]
                //
                //          A = (mask * templates)^T  X  (mask * templates)
                //            = [ [a_1 * a_1 * m_1 + a_2 * a_2 * m_2,
                //                 a_1 * b_1 * m_1 + a_2 * b_2 * m_2,
                //                 a_1 * c_1 * m_1 + a_2 * c_2 * m_2],
                //                [b_1 * a_1 * m_1 + b_2 * a_2 * m_2,
                //                 b_1 * b_1 * m_1 + b_2 * b_2 * m_2,
                //                 b_1 * c_1 * m_1 + b_2 * c_2 * m_2],
                //                [c_1 * a_1 * m_1 + c_2 * a_2 * m_2,
                //                 c_1 * b_1 * m_1 + c_2 * b_2 * m_2,
                //                 c_1 * c_1 * m_1 + c_2 * c_2 * m_2] ]
                //

                for (int32_t idet = 0; idet < ndet; ++idet) {
                    // For each detector...
                    if (det_group[idet] != igroup) {
                        // This detectors is not in this group
                        continue;
                    }

                    // Mask value for this detector
                    double det_mask = (masks[isamp * ndet + idet] == 0) ? 0.0 : 1.0;

                    // Signal value for this detector
                    double det_sig = signals[isamp * ndet + idet];

                    for (int32_t imode = 0; imode < nmode; ++imode) {
                        int32_t tmpl_off = idet * nmode;
                        rhs[imode] += templates[tmpl_off + imode] * det_sig * det_mask;

                        for (int32_t jmode = imode; jmode < nmode; ++jmode) {
                            double val = templates[tmpl_off + imode] *
                                         templates[tmpl_off + jmode] * det_mask;
                            A[imode * nmode + jmode] += val;
                            if (jmode > imode) {
                                A[jmode * nmode + imode] += val;
                            }
                        }
                    }
                }

                // DGELSS will overwrite RHS with the fitting
                // coefficients.  A is overwritten with
                // singular vectors.
                toast::lapack_dgelss(
                    &inmode, &inmode, &one, A.data(), &inmode,
                    rhs.data(), &inmode, singular_values.data(), &rcond_limit,
                    &rank, work.data(), &lwork, &info
                    );
                int64_t offset = isamp * (ngroup * nmode) + igroup * nmode;
                if (info == 0) {
                    // Solve was successful
                    for (int64_t m = 0; m < nmode; ++m) {
                        coeff[offset + m] = rhs[m];
                    }
                } else {
                    // Failed
                    for (int64_t m = 0; m < nmode; ++m) {
                        coeff[offset + m] = 0.0;
                    }
                }
            }
        }
    }

    return;
}
