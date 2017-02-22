/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#ifndef TOAST_TEST_HPP
#define TOAST_TEST_HPP

#include <toast_internal.hpp>

#include <gtest/gtest.h>

// #include <ctoast.h>


namespace toast { namespace test {
    int runner ( int argc, char *argv[] );
}}


class qarrayTest : public ::testing::Test {

    public :

        qarrayTest () { }
        ~qarrayTest () { }
        virtual void SetUp();
        virtual void TearDown() { }

        static const double q1[];
        static const double q1inv[];
        static const double q2[];
        static const double qtonormalize[];
        static const double qnormalized[];
        static const double vec[];
        static const double vec2[];
        static const double qeasy[];
        static const double mult_result[];
        static const double rot_by_q1[];
        static const double rot_by_q2[];

};


class rngTest : public ::testing::Test {

    public :

        rngTest () { }
        ~rngTest () { }
        virtual void SetUp();
        virtual void TearDown() { }

        static const int64_t size;
        static const uint64_t counter[];
        static const uint64_t key[];
        static const uint64_t counter00[];
        static const uint64_t key00[];

        static const double array_gaussian[];
        static const double array_m11[];
        static const double array_01[];
        static const uint64_t array_uint64[];

        static const double array00_gaussian[];
        static const double array00_m11[];
        static const double array00_01[];
        static const uint64_t array00_uint64[];

};


class sfTest : public ::testing::Test {

    public :

        sfTest () { }
        ~sfTest () { }
        virtual void SetUp();
        virtual void TearDown();

        static const int size;
        double * angin;
        double * sinout;
        double * cosout;
        double * xin;
        double * yin;
        double * atanout;
        double * sqin;
        double * sqout;
        double * rsqin;
        double * rsqout;
        double * expin;
        double * expout;
        double * login;
        double * logout;

};


class healpixTest : public ::testing::Test {

    public :

        healpixTest () { }
        ~healpixTest () { }
        virtual void SetUp() { }
        virtual void TearDown() { }

};


class fftTest : public ::testing::Test {

    public :

        fftTest () { }
        ~fftTest () { }
        virtual void SetUp() { }
        virtual void TearDown() { }

        static const int64_t length;
        static const int64_t n;

};


#endif
