/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#include <cstring>

#include <sstream>


void * toast::mem::aligned_alloc ( size_t size, size_t align ) {
    void * mem = NULL;
    int ret = posix_memalign ( &mem, align, size );
    if ( ret != 0 ) {
        std::ostringstream o;
        o << "cannot allocate " << size << " bytes of memory with alignment " << align;
        TOAST_THROW( o.str().c_str() );
    }
    memset ( mem, 0, size );
    return mem;
}


void toast::mem::aligned_free ( void * ptr ) {
    free ( ptr );
    return;
}


