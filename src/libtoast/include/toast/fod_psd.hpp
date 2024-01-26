
// Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_FOD_PSD_HPP
#define TOAST_FOD_PSD_HPP

#include <cstddef>
#include <cstdint>

namespace toast {
void fod_autosums(int64_t n, const double * x, const uint8_t * good,
                  int64_t lagmax, double * sums, int64_t * hits,
		  int64_t all_sums);

void fod_crosssums(int64_t n, const double * x, const double * y,
                   const uint8_t * good, int64_t lagmax, double * sums,
                   int64_t * hits, int64_t all_sums, int64_t symmetric);
}

#endif // ifndef TOAST_FOD_PSD_HPP
