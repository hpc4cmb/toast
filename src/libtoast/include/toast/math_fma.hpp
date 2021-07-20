/*
   Copyright (c) 2021 by the parties listed in the AUTHORS file.
   All rights reserved.  Use of this source code is governed by
   a BSD-style license that can be found in the LICENSE file.
 */

#ifndef TOAST_MATH_FMA_HPP
#define TOAST_MATH_FMA_HPP


namespace toast {
void inplace_weighted_sum(int n_out, int n_weights, double * out, double const * weights, double ** arrays);
}

#endif
