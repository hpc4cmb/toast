
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TOD_POINTING_HPP
#define TOAST_TOD_POINTING_HPP

#include <toast/math_healpix.hpp>


namespace toast {
void pointing_matrix_healpix(toast::HealpixPixels const & hpix,
                             bool nest, double eps, double cal,
                             std::string const & mode, size_t n,
                             double const * pdata, double const * hwpang,
                             uint8_t const * flags,
                             int64_t * pixels, double * weights);
}

#endif // ifndef TOAST_TOD_POINTING_HPP
