
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TOD_FILTER_HPP
#define TOAST_TOD_FILTER_HPP

namespace toast {
void filter_polynomial(int64_t order, size_t nsignal, uint8_t * flags,
                       std::vector <double *> const & signals, size_t nscan,
                       int64_t const * starts, int64_t const * stops);
void bin_templates(double * signal, double * templates, uint8_t * good,
                   double * invcov, double * proj, size_t nsample, size_t ntemplate);
void legendre(double * x, double * templates, size_t start_order, size_t stop_order,
               size_t nsample);
void chebyshev(double * x, double * templates, size_t start_order, size_t stop_order,
               size_t nsample);
void add_templates(double * signal, double * templates, double * coeff, size_t nsample,
                   size_t ntemplate);
}

#endif // ifndef TOAST_TOD_FILTER_HPP
