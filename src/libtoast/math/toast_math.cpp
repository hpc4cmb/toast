/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#include <cstring>

#include <sstream>


toast::exception::exception ( const char * msg, const char * file, int line ) : std::exception () {
    snprintf ( msg_, msg_len_, "Exeption at line %d of file %s:  %s", line, file, msg );
    return;
}


toast::exception::~exception ( ) throw () {
    return;
}


const char * toast::exception::what() const throw() { 
    return msg_;
}


