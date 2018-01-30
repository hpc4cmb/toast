/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_INTERNAL_HPP
#define TOAST_INTERNAL_HPP

#include <config.h>
#include <toast.hpp>

#include <sstream>
#include <string>
#include <cstdlib>

namespace toast {

template <typename _Tp>
_Tp get_env(const std::string& env_id, _Tp _default = _Tp())
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp var = _Tp();
        iss >> var;
        return var;
    }
    // return default if not specified in environment
    return _default;
}

}

#include <toast_math_internal.hpp>
#include <toast_atm_internal.hpp>
#include <toast_tod_internal.hpp>
#include <toast_fod_internal.hpp>
#include <toast_map_internal.hpp>

#include <toast_test.hpp>

#endif
