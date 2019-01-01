
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TOD_FILTER_HPP
#define TOAST_TOD_FILTER_HPP

namespace toast {

void filter_polynomial(int64_t order, double ** signals,
                       uint8_t * flags, size_t n, size_t nsignal,
                       int64_t const * starts, int64_t const * stops,
                       size_t nscan);

}

#endif // ifndef TOAST_TOD_FILTER_HPP
