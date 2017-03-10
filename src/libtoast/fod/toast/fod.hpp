/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_FOD_HPP
#define TOAST_FOD_HPP

#include <toast/math.hpp>
//#include <toast/atm.hpp>
#include <toast/tod.hpp>


namespace toast { namespace fod {

    void autosums ( int64_t n, double const * x, uint8_t const * good, int64_t lagmax, double * sums, int64_t * hits );

} }


#endif

