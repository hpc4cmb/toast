
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/fod_psd.hpp>

#include <cmath>


void toast::fod_autosums(int64_t n, double const * x, uint8_t const * good,
                         int64_t lagmax, double * sums, int64_t * hits) {
    toast::AlignedVector <double> xgood(n);
    toast::AlignedVector <uint8_t> gd(n);

    for (int64_t i = 0; i < n; ++i) {
        if (good[i] != 0) {
            xgood[i] = x[i];
            gd[i] = 1;
        } else {
            xgood[i] = 0.0;
            gd[i] = 0;
        }
    }

    #pragma \
    omp parallel for default(none) shared(n, gd, lagmax, xgood, sums, hits) schedule(static, 100)
    for (int64_t lag = 0; lag < lagmax; ++lag) {
        int64_t j = lag;
        double lagsum = 0.0;
        int64_t hitsum = 0;
        for (int64_t i = 0; i < (n - lag); ++i) {
            lagsum += xgood[i] * xgood[j];
            hitsum += gd[i] * gd[j];
            j++;
        }
        sums[lag] = lagsum;
        hits[lag] = hitsum;
    }

    return;
}

void toast::fod_crosssums(int64_t n, double const * x, double const * y,
                          uint8_t const * good, int64_t lagmax, double * sums,
                          int64_t * hits) {
    toast::AlignedVector <double> xgood(n);
    toast::AlignedVector <double> ygood(n);
    toast::AlignedVector <uint8_t> gd(n);

    for (int64_t i = 0; i < n; ++i) {
        if (good[i] != 0) {
            xgood[i] = x[i];
            ygood[i] = y[i];
            gd[i] = 1;
        } else {
            xgood[i] = 0.0;
            ygood[i] = 0.0;
            gd[i] = 0;
        }
    }

    #pragma \
    omp parallel for default(none) shared(n, gd, lagmax, xgood, ygood, sums, hits) schedule(static, 100)
    for (int64_t lag = 0; lag < lagmax; ++lag) {
        int64_t i, j;
        double lagsum = 0.0;
        int64_t hitsum = 0;
        for (i = 0, j = lag; i < (n - lag); ++i, ++j) {
            lagsum += xgood[i] * ygood[j];
            hitsum += gd[i] * gd[j];
        }
        // Use symmetry to double the statistics
        for (i = 0, j = lag; i < (n - lag); ++i, ++j) {
            lagsum += xgood[j] * ygood[i];
        }
        sums[lag] = lagsum;
        hits[lag] = 2 * hitsum;
    }

    return;
}
