/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_internal.hpp>


// Initialize MPI in a consistent way

void toast::init ( int argc, char *argv[] ) {

    int ret;
    int initialized;
    int threadprovided;

    ret = MPI_Initialized( &initialized );

    if ( ! initialized ) {
        
        #ifdef HAVE_ELEMENTAL

        // If we are using Elemental, let it initialize MPI
        El::Initialize ( argc, argv );
        
        #else
        
        ret = MPI_Init_thread ( &argc, &argv, MPI_THREAD_FUNNELED, &threadprovided );
        
        #endif

    }

    return;
}



