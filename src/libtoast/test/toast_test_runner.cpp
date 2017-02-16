/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>


int toast::test::runner ( int argc, char *argv[] ) {

    ::testing::InitGoogleTest ( &argc, argv );

    toast::init ( argc, argv );

    // FIXME:  rank 0 of MPI_COMM_WORLD should create the test
    // output directory here if it does not exist.

    int result = RUN_ALL_TESTS();

    return result;
}

