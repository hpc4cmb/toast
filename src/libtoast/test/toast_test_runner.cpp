/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>


int toast::test::runner ( int argc, char *argv[] ) {

    toast::init ( argc, argv );

    ::testing::InitGoogleTest ( &argc, argv );

    return RUN_ALL_TESTS();
}

