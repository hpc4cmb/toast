/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
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
        if ( stop > n ) stop = n;
        int scanlen = stop - start;
        if ( scanlen < 1 ) continue;

        // Build the template matrix

        double *templates = static_cast< double* >(
            toast::mem::aligned_alloc (
                scanlen*norder*sizeof(double), toast::mem::SIMD_ALIGN ) );

        double dx = 2. / scanlen;
        double xstart = 0.5*dx - 1;        
        for ( int i=0; i<scanlen; ++i ) {
            if ( flags[i] ) continue;
            double x = xstart + i*dx;            
            int offset = i*norder;
            if ( norder > 0 ) templates[offset++] = 1;
            if ( norder > 1 ) templates[offset++] = x;
            for ( int iorder=1; iorder<norder-1; ++iorder ) {
                templates[offset] =
                    ((2*iorder+1)*x*templates[offset-1]
                     - iorder*templates[offset-2]) / (iorder+1);
                ++offset;
            }
        }

        double *cov = static_cast< double* >(
            toast::mem::aligned_alloc (
                norder*norder*sizeof(double), toast::mem::SIMD_ALIGN ) );

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
            for ( int i=start; i<stop; ++i ) {
                flags[i] = 255;
            }
            continue;
        }

        // Filter every signal

        double *coeff = static_cast< double* >(
            toast::mem::aligned_alloc (
                norder*sizeof(double), toast::mem::SIMD_ALIGN ) );

        double *noise = static_cast< double* >(
            toast::mem::aligned_alloc (
                scanlen*sizeof(double), toast::mem::SIMD_ALIGN ) );

        for ( int isignal=0; isignal<nsignal; ++isignal ) {
            double *signal = signals[isignal] + start;

            // proj = templates x signal

            toast::lapack::gemv( &notrans, &norder, &scanlen, &fone, templates,
                                 &norder, signal, &one, &fzero, coeff, &one );

            // coeff = cov x proj

            toast::lapack::trmv( &upper, &trans, &nodiag, &norder, cov,
                                 &norder, coeff, &one );

            // noise = templates.T x coeff

            toast::lapack::gemv( &trans, &norder, &scanlen, &fone, templates,
                                 &norder, coeff, &one, &fzero, noise, &one );

            // Subtract noise

            for ( int i=0; i<scanlen; ++i ) signal[i] -= noise[i];
        }

        // Free workspace

        toast::mem::aligned_free( templates );
        toast::mem::aligned_free( cov );
        toast::mem::aligned_free( coeff );
        toast::mem::aligned_free( noise );

    }

    return;
}
