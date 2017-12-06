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

class TOASTtimingTest : public ::testing::Test
{
public:
    TOASTtimingTest () { }
    ~TOASTtimingTest () { }
    virtual void SetUp() { }
    virtual void TearDown() { }
};


class TOASTqarrayTest : public ::testing::Test {

    public :

        TOASTqarrayTest () { }
        ~TOASTqarrayTest () { }
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


class TOASTrngTest : public ::testing::Test {

    public :

        TOASTrngTest () { }
        ~TOASTrngTest () { }
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


class TOASTsfTest : public ::testing::Test {

    public :

        TOASTsfTest () { }
        ~TOASTsfTest () { }
        virtual void SetUp();
        virtual void TearDown();

        static const int size;
        toast::mem::simd_array<double> angin;
        toast::mem::simd_array<double> sinout;
        toast::mem::simd_array<double> cosout;
        toast::mem::simd_array<double> xin;
        toast::mem::simd_array<double> yin;
        toast::mem::simd_array<double> atanout;
        toast::mem::simd_array<double> sqin;
        toast::mem::simd_array<double> sqout;
        toast::mem::simd_array<double> rsqin;
        toast::mem::simd_array<double> rsqout;
        toast::mem::simd_array<double> expin;
        toast::mem::simd_array<double> expout;
        toast::mem::simd_array<double> login;
        toast::mem::simd_array<double> logout;

};


class TOASThealpixTest : public ::testing::Test {

    public :

        TOASThealpixTest () { }
        ~TOASThealpixTest () { }
        virtual void SetUp() { }
        virtual void TearDown() { }

};


class TOASTfftTest : public ::testing::Test {

    public :

        TOASTfftTest () { }
        ~TOASTfftTest () { }
        virtual void SetUp() { }
        virtual void TearDown() { }

        void runbatch(int64_t nbatch, toast::fft::r1d_p forward, 
            toast::fft::r1d_p reverse);

        static const int64_t length;
        static const int64_t n;

};


class TOASTcovTest : public ::testing::Test {

    public :

        TOASTcovTest () { }
        ~TOASTcovTest () { }
        virtual void SetUp() { }
        virtual void TearDown() { }

        static const int64_t nsm;
        static const int64_t npix;
        static const int64_t nnz;
        static const int64_t nsamp;
        static const int64_t scale;

};


class TOASTmpiShmemTest : public ::testing::Test {

    public :

        TOASTmpiShmemTest () { }
        ~TOASTmpiShmemTest () { }
        virtual void SetUp() { }
        virtual void TearDown() { }

        static const size_t n;

};


class TOASTpolyfilterTest : public ::testing::Test {

public :

    TOASTpolyfilterTest () { }
    ~TOASTpolyfilterTest () { }
    virtual void SetUp() { }
    virtual void TearDown() { }

    static const size_t order, n;

};



#endif
