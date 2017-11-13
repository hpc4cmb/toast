/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_SIM_NOISE_HPP
#define TOAST_SIM_NOISE_HPP

#include <math.h>
#include <sstream>


namespace toast { namespace sim_noise {

void sim_noise_timestream(
    const uint64_t realization, const uint64_t telescope,
    const uint64_t component, const uint64_t obsindx, const uint64_t detindx,
    const double rate, const uint64_t firstsamp, const uint64_t samples,
    const uint64_t oversample, const double *freq, const double *psd,
    const uint64_t psdlen, double *noise);

} }

#endif
