/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const size_t polyfilterTest::order = 3;
const size_t polyfilterTest::n = 1000;


TEST_F( polyfilterTest, filter ) {

    vector<double> signal1(n);
    vector<double> signal2(n);
    vector<double> signal3(n);
    vector<unsigned char> flags(n, 0);

    double *signals[3];

    signals[0] = signal1.data();
    signals[1] = signal2.data();
    signals[2] = signal3.data();

    for ( int i=0; i<n; ++i ) {
        signal1[i] = 1;
        signal2[i] = i;
        signal3[i] = i*i;
    }
    size_t nsignal = 3;

    long starts[] = { 0, n/2 };
    long stops[] = { n/2, n };
    size_t nscan = 2;

    double rms1start = 0, rms2start = 0, rms3start = 0;

    for ( int i=0; i<n; ++i ) {
        rms1start += signal1[i]*signal1[i];
        rms2start += signal2[i]*signal2[i];
        rms3start += signal3[i]*signal3[i];
    }

    rms1start = sqrt(rms1start / n);
    rms2start = sqrt(rms2start / n);
    rms3start = sqrt(rms3start / n);

    toast::filter::polyfilter( order, signals, flags.data(), n, nsignal,
                               starts, stops, nscan );

    double rms1 = 0, rms2 = 0, rms3 = 0;

    for ( int i=0; i<n; ++i ) {
        rms1 += signal1[i]*signal1[i];
        rms2 += signal2[i]*signal2[i];
        rms3 += signal3[i]*signal3[i];
    }

    rms1 = sqrt(rms1 / n);
    rms2 = sqrt(rms2 / n);
    rms3 = sqrt(rms3 / n);

    EXPECT_LT( std::abs(rms1/rms1start), 1e-6 );
    EXPECT_LT( std::abs(rms2/rms2start), 1e-6 );
    EXPECT_LT( std::abs(rms3/rms3start), 1e-6 );

}
