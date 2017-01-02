/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#ifndef CTOAST_HPP
#define CTOAST_HPP

#include <ctoast.h>

#include <toast_internal.hpp>


namespace ctoast {  

    // enum conversions

    toast::fft::plan_type convert_from_c ( ctoast_fft_plan_type in );

    ctoast_fft_plan_type convert_to_c ( toast::fft::plan_type in );

    toast::fft::direction convert_from_c ( ctoast_fft_direction in );

    ctoast_fft_direction convert_to_c ( toast::fft::direction in );

}

#endif


