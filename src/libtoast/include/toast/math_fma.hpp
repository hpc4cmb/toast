/*
   Copyright (c) 2021 by the parties listed in the AUTHORS file.
   All rights reserved.  Use of this source code is governed by
   a BSD-style license that can be found in the LICENSE file.
 */

#ifndef TOAST_MATH_FMA_HPP
#define TOAST_MATH_FMA_HPP


namespace toast {
void inplace_weighted_sum(
    int const n_out,
    int const n_weights,
    double * const out,
    double const * const weights,
    double const * const * const arrays
    );
}

#endif // ifndef TOAST_MATH_FMA_HPP
