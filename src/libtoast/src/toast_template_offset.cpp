
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/template_offset.hpp>

#include <cmath>

void toast::template_offset_add_to_signal(int64_t step_length, int64_t n_amp,
                                          double * amplitudes,
                                          int64_t n_data, double * data) {
    // All but the last amplitude have the same number of samples.
    if (toast::is_aligned(amplitudes) && toast::is_aligned(data)) {
        #pragma omp simd
        for (int64_t i = 0; i < n_amp - 1; ++i) {
            int64_t doff = i * step_length;
            for (int64_t j = 0; j < step_length; ++j) {
                data[doff + j] += amplitudes[i];
            }
        }
    } else {
        for (int64_t i = 0; i < n_amp - 1; ++i) {
            int64_t doff = i * step_length;
            for (int64_t j = 0; j < step_length; ++j) {
                data[doff + j] += amplitudes[i];
            }
        }
    }

    // Now handle the final amplitude.
    for (int64_t j = (n_amp - 1) * step_length; j < n_data; ++j) {
        data[j] += amplitudes[n_amp - 1];
    }
    return;
}

void toast::template_offset_project_signal(int64_t step_length, int64_t n_data,
                                           double * data, int64_t n_amp,
                                           double * amplitudes) {
    // All but the last amplitude have the same number of samples.
    if (toast::is_aligned(amplitudes) && toast::is_aligned(data)) {
        #pragma omp simd
        for (int64_t i = 0; i < n_amp - 1; ++i) {
            int64_t doff = i * step_length;
            for (int64_t j = 0; j < step_length; ++j) {
                amplitudes[i] += data[doff + j];
            }
        }
    } else {
        for (int64_t i = 0; i < n_amp - 1; ++i) {
            int64_t doff = i * step_length;
            for (int64_t j = 0; j < step_length; ++j) {
                amplitudes[i] += data[doff + j];
            }
        }
    }

    // Now handle the final amplitude.
    for (int64_t j = (n_amp - 1) * step_length; j < n_data; ++j) {
        amplitudes[n_amp - 1] += data[j];
    }
    return;
}
