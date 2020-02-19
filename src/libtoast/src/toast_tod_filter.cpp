
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <string.h>
#include <omp.h>

#include <toast/sys_utils.hpp>
#include <toast/math_lapack.hpp>
#include <toast/tod_filter.hpp>


void toast::filter_polynomial(int64_t order, size_t n, uint8_t * flags,
                              std::vector <double *> const & signals, size_t nscan,
                              int64_t const * starts, int64_t const * stops) {
    // Process the signals, one subscan at a time.  There is only one
    // flag vector, because the flags must be identical to apply the
    // same template matrix.

    size_t nsignal = signals.size();

    #pragma \
    omp parallel default(none) shared(order, signals, flags, n, nsignal, starts, stops, nscan)
    {
        char upper = 'U';
        char lower = 'L';
        char notrans = 'N';
        char trans = 'T';
        char diag = 'U';
        char nodiag = 'N';
        double fzero = 0.0;
        double fone = 1.0;
        int one = 1;
        int zero = 0;

        int64_t norder = order + 1;
        int fnorder = static_cast <int> (norder);

        #pragma omp for schedule(static)
        for (size_t iscan = 0; iscan < nscan; ++iscan) {
            int64_t start = starts[iscan];
            int64_t stop = stops[iscan];
            if (start < 0) start = 0;
            if (stop > n - 1) stop = n - 1;
            if (stop < start) continue;
            int64_t scanlen = stop - start + 1;
            int fscanlen = static_cast <int> (scanlen);

            // Build the full template matrix used to clean the signal.
            // We subtract the template value even from flagged samples to
            // support point source masking etc.

            toast::AlignedVector <double> full_templates(scanlen * norder);

            double dx = 2. / scanlen;
            double xstart = 0.5 * dx - 1;

            // FIXME: can we be more clever here and remove the conditionals
            // inside the for loop?
            #pragma omp simd
            for (int64_t i = 0; i < scanlen; ++i) {
                double x = xstart + i * dx;
                int64_t offset = i * norder;
                if (norder > 0) full_templates[offset++] = 1;
                if (norder > 1) full_templates[offset++] = x;
                for (int64_t iorder = 1; iorder < norder - 1; ++iorder) {
                    full_templates[offset] =
                        ((2 * iorder + 1) * x * full_templates[offset - 1]
                         - iorder * full_templates[offset - 2]) / (iorder + 1);
                    ++offset;
                }
            }

            // Assemble the flagged template matrix used in the linear
            // regression

            toast::AlignedVector <double> templates(scanlen * norder);

            for (int64_t i = 0; i < scanlen; ++i) {
                if (flags[start + i] != 0) continue;
                for (int64_t offset = i * norder; offset < (i + 1) * norder;
                     ++offset) {
                    templates[offset] = full_templates[offset];
                }
            }

            toast::AlignedVector <double> cov(norder * norder);

            // invcov = templates x templates.T

            toast::lapack_syrk(&upper, &notrans, &fnorder, &fscanlen, &fone,
                               templates.data(), &fnorder, &fzero, cov.data(),
                               &fnorder);

            // invert cov = invcov^-1 using Cholesky decomposition

            int info = 0;
            toast::lapack_potrf(&upper, &fnorder, cov.data(), &fnorder, &info);
            if (info == 0) {
                toast::lapack_potri(&upper, &fnorder, cov.data(), &fnorder,
                                    &info);
            }

            if (info) {
                // The matrix is singular.  Raise the appropriate quality
                // flags.
                for (int i = 0; i < scanlen; ++i) {
                    flags[start + i] = 255;
                }
                continue;
            }

            // Symmetrize for dgemv later on (workaround for dtrmv issue)

            for (int row = 0; row < norder; ++row) {
                for (int col = row + 1; col < norder; ++col) {
                    cov[row * norder + col] = cov[col * norder + row];
                }
            }

            // Filter every signal

            toast::AlignedVector <double> proj(norder);
            toast::AlignedVector <double> coeff(norder);
            toast::AlignedVector <double> noise(scanlen);

            for (size_t isignal = 0; isignal < nsignal; ++isignal) {
                double * signal = signals[isignal] + start;

                // proj = templates x signal

                toast::lapack_gemv(&notrans, &fnorder, &fscanlen, &fone,
                                   templates.data(), &fnorder, signal, &one,
                                   &fzero, proj.data(), &one);

                // coeff = cov x proj

                // For whatever reason, trmv refused to yield the right
                // answer...
                // Use general matrix vector multiply instead

                // toast::lapack::trmv( &upper, &trans, &nodiag, &norder, cov,
                //                     &norder, coeff, &one );

                toast::lapack_gemv(&trans, &fnorder, &fnorder, &fone,
                                   cov.data(), &fnorder, proj.data(), &one,
                                   &fzero, coeff.data(), &one);

                // noise = templates.T x coeff

                toast::lapack_gemv(&trans, &fnorder, &fscanlen, &fone,
                                   full_templates.data(), &fnorder,
                                   coeff.data(), &one, &fzero, noise.data(),
                                   &one);

                // Subtract noise

                for (int64_t i = 0; i < scanlen; ++i) signal[i] -= noise[i];
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
        int nthread = omp_get_num_threads();
        int id_thread = omp_get_thread_num();

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
    if (start_order == 0) {
        for (size_t i = 0; i < nsample; ++i) templates[i] = 1;
    }

    // order == 1
    if (start_order <= 1) {
        memcpy(templates + (1 - start_order) * nsample, x, nsample * sizeof(double));
    }

    const size_t buflen = 1000;
    size_t nbuf = nsample / buflen + 1;

#pragma omp parallel for \
    schedule(static) default(none) shared(x, templates, start_order, stop_order, nsample, nbuf)
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
            for (size_t i = 0; i < n;
                 ++i) next[i] = 2 * x[istart + i] * val[i] - prev[i];
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
