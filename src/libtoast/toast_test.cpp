/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>


using namespace std;
using namespace toast;


int main ( int argc, char *argv[] ) {
    int ret = toast::test::runner ( argc, argv );
    toast::finalize();
    return ret;
}


