/*
 Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
 All rights reserved.  Use of this source code is governed by
 a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_FOD_HPP
#define TOAST_FOD_HPP

#include <toast/math.hpp>
#include <toast/atm.hpp>
#include <toast/tod.hpp>


namespace toast { namespace fod {

void autosums(int64_t n, const double * x, const uint8_t * good, int64_t lagmax,
		double * sums, int64_t * hits);

void crosssums(int64_t n, const double * x, const double * y,
		const uint8_t * good, int64_t lagmax, double * sums, int64_t * hits);

} }


#endif
