/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const int sfTest::size = 1000;


void sfTest::SetUp () {

    angin = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    sinout = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    cosout = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    xin = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    yin = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    atanout = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    sqin = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    sqout = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    rsqin = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    rsqout = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    expin = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    expout = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    login = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    logout = (double*) mem::aligned_alloc ( size * sizeof(double), mem::SIMD_ALIGN );
    
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


void sfTest::TearDown () {

    mem::aligned_free ( (void*) angin );
    mem::aligned_free ( (void*) sinout );
    mem::aligned_free ( (void*) cosout );
    mem::aligned_free ( (void*) xin );
    mem::aligned_free ( (void*) yin );
    mem::aligned_free ( (void*) atanout );
    mem::aligned_free ( (void*) sqin );
    mem::aligned_free ( (void*) sqout );
    mem::aligned_free ( (void*) rsqin );
    mem::aligned_free ( (void*) rsqout );
    mem::aligned_free ( (void*) expin );
    mem::aligned_free ( (void*) expout );
    mem::aligned_free ( (void*) login );
    mem::aligned_free ( (void*) logout );

    return;
}


TEST_F( sfTest, trig ) {
    double comp1[size];
    double comp2[size];

    sf::sin ( size, angin, comp1 );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( sinout[i], comp1[i] );
    }

    sf::cos ( size, angin, comp2 );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( cosout[i], comp2[i] );
    }

    sf::sincos ( size, angin, comp1, comp2 );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( sinout[i], comp1[i] );
        EXPECT_DOUBLE_EQ( cosout[i], comp2[i] );
    }

    sf::atan2 ( size, yin, xin, comp1 );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( atanout[i], comp1[i] );
    }
}


TEST_F( sfTest, fasttrig ) {
    double comp1[size];
    double comp2[size];

    sf::fast_sin ( size, angin, comp1 );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( sinout[i], comp1[i] );
    }

    sf::fast_cos ( size, angin, comp2 );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( cosout[i], comp2[i] );
    }

    sf::fast_sincos ( size, angin, comp1, comp2 );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( sinout[i], comp1[i] );
        EXPECT_FLOAT_EQ( cosout[i], comp2[i] );
    }

    sf::fast_atan2 ( size, yin, xin, comp1 );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( atanout[i], comp1[i] );
    }
}


TEST_F( sfTest, sqrtlog ) {
    double comp[size];

    sf::sqrt ( size, sqin, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( sqout[i], comp[i] );
    }

    sf::rsqrt ( size, rsqin, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( rsqout[i], comp[i] );
    }

    sf::exp ( size, expin, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( expout[i], comp[i] );
    }

    sf::log ( size, login, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_DOUBLE_EQ( logout[i], comp[i] );
    }
}


TEST_F( sfTest, fast_sqrtlog ) {
    double comp[size];

    sf::fast_sqrt ( size, sqin, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( sqout[i], comp[i] );
    }

    sf::fast_rsqrt ( size, rsqin, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( rsqout[i], comp[i] );
    }

    sf::fast_exp ( size, expin, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( expout[i], comp[i] );
    }

    sf::fast_log ( size, login, comp );
    for ( int i; i < size; ++i ) {
        EXPECT_FLOAT_EQ( logout[i], comp[i] );
    }
}

