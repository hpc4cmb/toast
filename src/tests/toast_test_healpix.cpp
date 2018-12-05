/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


TEST_F( TOASThealpixTest, pixelops ) {

    // These numbers were generated with the included script.

    #include "data_healpix.cpp"

    healpix::pixels hpx ( nside );

    int64_t comp_pixring[ntest];
    int64_t comp_pixnest[ntest];
    double comp_theta[ntest];
    double comp_phi[ntest];

    hpx.ang2ring ( ntest, theta, phi, comp_pixring );
    for ( int64_t i = 0; i < ntest; ++i ) {
        //std::cerr << i << ": (" << theta[i] << "," << phi[i] << ") = " << pixring[i] << " =? " << comp_pixring[i] << std::endl;
        EXPECT_EQ( pixring[i], comp_pixring[i] );
    }

    hpx.ang2nest ( ntest, theta, phi, comp_pixnest );
    for ( int64_t i = 0; i < ntest; ++i ) {
        EXPECT_EQ( pixnest[i], comp_pixnest[i] );
    }

    hpx.ring2nest ( ntest, comp_pixring, comp_pixnest );
    for ( int64_t i = 0; i < ntest; ++i ) {
        EXPECT_EQ( pixnest[i], comp_pixnest[i] );
    }

}

