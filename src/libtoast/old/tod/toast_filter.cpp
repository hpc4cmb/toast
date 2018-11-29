/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_tod_internal.hpp>

#include <sstream>
#include <iostream>
#include <iomanip>


void toast::filter::polyfilter(
    const long order, double **signals, uint8_t *flags,
    const size_t n, const size_t nsignal,
    const long *starts, const long *stops, const size_t nscan ) {

    // Process the signals, one subscan at a time.  There is only one
    // flag vector, because the flags must be identical to apply the
    // same template matrix.

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

    int norder = order + 1;

#pragma omp parallel for schedule(static)
    for ( int iscan=0; iscan<nscan; ++iscan ) {
        int start = starts[iscan];
        int stop = stops[iscan];
        if ( start < 0 ) start = 0;
        if ( stop > n-1 ) stop = n-1;
        if ( stop < start ) continue;
        int scanlen = stop - start + 1;

        // Build the full template matrix used to clean the signal.
        // We subtract the template value even from flagged samples to
        // support point source masking etc.

        toast::mem::simd_array<double> full_templates(scanlen*norder);

        double dx = 2. / scanlen;
        double xstart = 0.5*dx - 1;
        for ( int i=0; i<scanlen; ++i ) {
            double x = xstart + i*dx;
            int offset = i*norder;
            if ( norder > 0 ) full_templates[offset++] = 1;
            if ( norder > 1 ) full_templates[offset++] = x;
            for ( int iorder=1; iorder<norder-1; ++iorder ) {
                full_templates[offset] =
                    ((2*iorder+1)*x*full_templates[offset-1]
                     - iorder*full_templates[offset-2]) / (iorder+1);
                ++offset;
            }
        }

        // Assemble the flagged template matrix used in the linear regression

        toast::mem::simd_array<double> templates(scanlen*norder);

        for ( int i=0; i<scanlen; ++i ) {
            if ( flags[start+i] ) continue;
            for ( int offset=i*norder; offset<(i+1)*norder; ++offset )
                templates[offset] = full_templates[offset];
        }

        toast::mem::simd_array<double> cov(norder*norder);

        // invcov = templates x templates.T

        toast::lapack::syrk( &upper, &notrans, &norder, &scanlen, &fone,
                             templates, &norder, &fzero, cov,
                             &norder );

        // invert cov = invcov^-1 using LU decomposition

        int info = 0;
        toast::lapack::potrf( &upper, &norder, cov, &norder, &info );
        if ( info == 0 ) {
            toast::lapack::potri( &upper, &norder, cov, &norder, &info );
        }

        if ( info ) {
            // The matrix is singular.  Raise the appropriate quality
            // flags.
            for ( int i=0; i<scanlen; ++i ) {
                flags[start+i] = 255;
            }
            continue;
        }

        // Symmetrize for dgemv later on (workaround for dtrmv issue)

        for ( int row=0; row<norder; ++row ) {
          for ( int col=row+1; col<norder; ++col ) {
            cov[row*norder + col] = cov[col*norder + row];
          }
        }

        // Filter every signal

        toast::mem::simd_array<double> proj(norder);
        toast::mem::simd_array<double> coeff(norder);
        toast::mem::simd_array<double> noise(scanlen);

        for ( int isignal=0; isignal<nsignal; ++isignal ) {
            double *signal = signals[isignal] + start;

            // proj = templates x signal

            toast::lapack::gemv( &notrans, &norder, &scanlen, &fone, templates,
                                 &norder, signal, &one, &fzero, proj, &one );

            // coeff = cov x proj

            // For whatever reason, trmv refused to yield the right answer...
            // Use general matrix vector multiply instead

            //toast::lapack::trmv( &upper, &trans, &nodiag, &norder, cov,
            //                     &norder, coeff, &one );

            toast::lapack::gemv( &trans, &norder, &norder, &fone,
                                 cov, &norder, proj, &one,
                                 &fzero, coeff, &one );

            // noise = templates.T x coeff

            toast::lapack::gemv( &trans, &norder, &scanlen, &fone,
                                 full_templates, &norder, coeff, &one, &fzero,
                                 noise, &one );

            // Subtract noise

            for ( int i=0; i<scanlen; ++i ) signal[i] -= noise[i];
        }

    }

    return;
}
