/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const int TOASTsfTest::size = 1000;


void TOASTsfTest::SetUp () {

    angin = toast::mem::simd_array<double>(size);
    sinout = toast::mem::simd_array<double>(size);
    cosout = toast::mem::simd_array<double>(size);
    xin = toast::mem::simd_array<double>(size);
    yin = toast::mem::simd_array<double>(size);
    atanout = toast::mem::simd_array<double>(size);
    sqin = toast::mem::simd_array<double>(size);
    sqout = toast::mem::simd_array<double>(size);
    rsqin = toast::mem::simd_array<double>(size);
    rsqout = toast::mem::simd_array<double>(size);
    expin = toast::mem::simd_array<double>(size);
    expout = toast::mem::simd_array<double>(size);
    login = toast::mem::simd_array<double>(size);
    logout = toast::mem::simd_array<double>(size);
        
    for ( int i = 0; i < size; ++i ) {
        angin[i] = (double)(i + 1) * ( 2.0 * PI / (double)(size + 1) );
        sinout[i] = ::sin(angin[i]);
        cosout[i] = ::cos(angin[i]);
        xin[i] = cosout[i];
        yin[i] = sinout[i];
        atanout[i] = ::atan2(yin[i], xin[i]);

        sqin[i] = (double)i / (double)size;
        rsqin[i] = sqin[i];
        sqout[i] = ::sqrt(sqin[i]);
        rsqout[i] = 1.0 / ::sqrt(rsqin[i]);

        expin[i] = -10.0 + (double)i * 20.0 / (double)size;
        expout[i] = ::exp(expin[i]);
        login[i] = expout[i];
        logout[i] = ::log(login[i]);
    }

    return;
}


void TOASTsfTest::TearDown () {

    return;
}


TEST_F( TOASTsfTest, trig ) {
    double comp1[size];
    double comp2[size];

    sf::sin ( size, angin, comp1 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( sinout[i], comp1[i] );
    }

    sf::cos ( size, angin, comp2 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( cosout[i], comp2[i] );
    }

    sf::sincos ( size, angin, comp1, comp2 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( sinout[i], comp1[i] );
        EXPECT_DOUBLE_EQ( cosout[i], comp2[i] );
    }

    sf::atan2 ( size, yin, xin, comp1 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( atanout[i], comp1[i] );
    }
}


TEST_F( TOASTsfTest, fasttrig ) {
    double comp1[size];
    double comp2[size];

    sf::fast_sin ( size, angin, comp1 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_FLOAT_EQ( sinout[i], comp1[i] );
    }

    sf::fast_cos ( size, angin, comp2 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_FLOAT_EQ( cosout[i], comp2[i] );
    }

    sf::fast_sincos ( size, angin, comp1, comp2 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_FLOAT_EQ( sinout[i], comp1[i] );
        EXPECT_FLOAT_EQ( cosout[i], comp2[i] );
    }

    sf::fast_atan2 ( size, yin, xin, comp1 );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_FLOAT_EQ( atanout[i], comp1[i] );
    }
}


TEST_F( TOASTsfTest, sqrtlog ) {
    double comp[size];

    sf::sqrt ( size, sqin, comp );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( sqout[i], comp[i] );
    }

    sf::rsqrt ( size, rsqin, comp );
    for ( int i = 0; i < size; ++i ) {
        if(std::isfinite(comp[i]) || std::isfinite(rsqout[i])) {
            EXPECT_DOUBLE_EQ( rsqout[i], comp[i] );
        }
    }

    sf::exp ( size, expin, comp );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( expout[i], comp[i] );
    }

    sf::log ( size, login, comp );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( logout[i], comp[i] );
    }
}


TEST_F( TOASTsfTest, fast_sqrtlog ) {
    double comp[size];

    sf::fast_sqrt ( size, sqin, comp );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_FLOAT_EQ( sqout[i], comp[i] );
    }

    sf::fast_rsqrt ( size, rsqin, comp );
    for ( int i = 0; i < size; ++i ) {
        if(std::isfinite(comp[i]) || std::isfinite(rsqout[i])) {
            EXPECT_FLOAT_EQ( rsqout[i], comp[i] );
        }
    }

    sf::fast_exp ( size, expin, comp );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_FLOAT_EQ( expout[i], comp[i] );
    }

    sf::fast_log ( size, login, comp );
    for ( int i = 0; i < size; ++i ) {
        EXPECT_FLOAT_EQ( logout[i], comp[i] );
    }
}


TEST_F( TOASTsfTest, fast_erfinv ) {
    double in[10] = {
        -9.990000e-01,
        -7.770000e-01,
        -5.550000e-01,
        -3.330000e-01,
        -1.110000e-01,
        1.110000e-01,
        3.330000e-01,
        5.550000e-01,
        7.770000e-01,
        9.990000e-01
    };

    double check[10] = {
        -2.326753765513524e+00,
        -8.616729665092674e-01,
        -5.400720684419686e-01,
        -3.042461029341061e-01,
        -9.869066534119145e-02,
        9.869066534119164e-02,
        3.042461029341063e-01,
        5.400720684419690e-01,
        8.616729665092677e-01,
        2.326753765513546e+00
    };

    double out[10];

    sf::fast_erfinv ( 10, in, out );

    for ( int i = 0; i < 10; ++i ) {
        EXPECT_FLOAT_EQ( out[i], check[i] );
    }
}


