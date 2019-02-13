/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_map_internal.hpp>
#include <toast_util_internal.hpp>

#include <cstring>
#include <iostream>

#ifdef _OPENMP
#  include <omp.h>
#endif

void toast::map_tools::fast_scanning32(double * toi, int64_t const nsamp,
                                       int64_t const * pixels,
                                       double const * weights,
                                       int64_t const nweight,
                                       float const * bmap) {
    memset(toi, 0, nsamp * sizeof(double));
    #pragma omp parallel for
    for (int64_t row = 0; row < nsamp; ++row) {
        int64_t offset = row * nweight;
        for (int64_t col = 0; col < nweight; ++col) {
            int64_t pix = pixels[offset];
            if (pix < 0) continue;
            double weight = weights[offset];
            toi[row] += bmap[pix] * weight;
            ++offset;
        }
    }
}
