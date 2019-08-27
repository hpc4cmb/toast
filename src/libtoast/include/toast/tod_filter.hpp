
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TOD_FILTER_HPP
#define TOAST_TOD_FILTER_HPP

namespace toast {
void filter_polynomial(int64_t order, size_t nsignal, uint8_t * flags,
                       std::vector <double *> const & signals, size_t nscan,
                       int64_t const * starts, int64_t const * stops);
}

#endif // ifndef TOAST_TOD_FILTER_HPP
