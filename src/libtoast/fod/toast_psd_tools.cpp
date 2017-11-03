/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_fod_internal.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <cmath>


void toast::fod::autosums ( int64_t n,
                            double const * x,
                            uint8_t const * good,
                            int64_t lagmax,
                            double * sums,
                            int64_t * hits )
{

    int64_t i, j;
    int64_t lag;

    double lagsum;
    int64_t hitsum;

    toast::mem::simd_array<double> xgood(n);
    toast::mem::simd_array<uint8_t> gd(n);


    for ( i = 0; i < n; ++i ) {
        if ( good[i] != 0 ) {
            xgood[i] = x[i];
            gd[i] = 1;
        } else {
            xgood[i] = 0.0;
            gd[i] = 0;
        }
    }

    #pragma omp parallel for default(none) private(i, j, lag, lagsum, hitsum) shared(n, gd, lagmax, xgood, sums, hits) schedule(dynamic)
    for ( lag = 0; lag < lagmax; ++lag )
    {
        j = lag;
        lagsum = 0.0;
        hitsum = 0.0;
        for ( i = 0; i < ( n - lag ); ++i ) {
            lagsum += xgood[i] * xgood[j];
            hitsum += gd[i] * gd[j];
            j++;
        }
        sums[lag] = lagsum;
        hits[lag] = hitsum;
    }

    return;
}


