
// Copyright (c) 2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_fma.hpp>


// be careful of incorrect result if the pragma is put outside
void toast::inplace_weighted_sum(int const n_out, int const n_weights, double * const out, double const * const weights, double const * const * const arrays) {
    if (! toast::is_aligned(out)) {
        for (int i = 0; i < n_weights; ++i) {
            double const weight = weights[i];
            double const * const array = arrays[i];
            # pragma omp parallel for default(shared) schedule(static)
            for (int j = 0; j < n_out; ++j) {
                out[j] += weight * array[j];
            }
        }
    } else {
        for (int i = 0; i < n_weights; ++i) {
            double const weight = weights[i];
            double const * const array = arrays[i];
            if (toast::is_aligned(array)) {
                # pragma omp parallel for simd default(shared) schedule(static)
                for (int j = 0; j < n_out; ++j) {
                    out[j] += weight * array[j];
                }
            } else {
                # pragma omp parallel for default(shared) schedule(static)
                for (int j = 0; j < n_out; ++j) {
                    out[j] += weight * array[j];
                }
            }
        }
    }
}


// the following produces correct results, but slower.
// void toast::inplace_weighted_sum(int n_out, int n_weights, double * out, double const * weights, double ** arrays) {
//     # pragma omp parallel for simd
//     for (int j = 0; j < n_out; ++j) {
//         for (int i = 0; i < n_weights; ++i) {
//             out[j] += weights[i] * arrays[i][j];
//         }
//     }
// }
