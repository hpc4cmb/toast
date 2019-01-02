
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TOD_SIMNOISE_HPP
#define TOAST_TOD_SIMNOISE_HPP

#include <cstddef>
#include <cstdint>


namespace toast {

void tod_sim_noise_timestream(
    uint64_t realization, uint64_t telescope, uint64_t component,
    uint64_t obsindx, uint64_t detindx, double rate, int64_t firstsamp,
    int64_t samples, int64_t oversample, const double * freq,
    const double * psd, int64_t psdlen, double * noise);

}

#endif // ifndef TOAST_TOD_SIMNOISE_HPP
