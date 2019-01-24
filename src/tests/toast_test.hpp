
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TEST_HPP
#define TOAST_TEST_HPP

#include <toast.hpp>

#include <gtest/gtest.h>


namespace toast { namespace test {

int runner(int argc, char * argv[]);

} }

class TOASTenvTest : public ::testing::Test {
    public:

        TOASTenvTest() {}

        ~TOASTenvTest() {}

        virtual void SetUp() {}

        virtual void TearDown() {}
};

class TOASTutilsTest : public ::testing::Test {
    public:

        TOASTutilsTest() {}

        ~TOASTutilsTest() {}

        virtual void SetUp() {}

        virtual void TearDown() {}
};


class TOASTqarrayTest : public ::testing::Test {
    public:

        TOASTqarrayTest() {}

        ~TOASTqarrayTest() {}

        virtual void SetUp();
        virtual void TearDown() {}

        toast::AlignedVector <double> q1;
        toast::AlignedVector <double> q1inv;
        toast::AlignedVector <double> q2;
        toast::AlignedVector <double> qtonormalize;
        toast::AlignedVector <double> qnormalized;
        toast::AlignedVector <double> vec;
        toast::AlignedVector <double> vec2;
        toast::AlignedVector <double> qeasy;
        toast::AlignedVector <double> mult_result;
        toast::AlignedVector <double> rot_by_q1;
        toast::AlignedVector <double> rot_by_q2;
};


class TOASTrngTest : public ::testing::Test {
    public:

        TOASTrngTest() {}

        ~TOASTrngTest() {}

        virtual void SetUp();
        virtual void TearDown() {}

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
    public:

        TOASTsfTest() {}

        ~TOASTsfTest() {}

        virtual void SetUp();
        virtual void TearDown();

        static const int size;
        toast::AlignedVector <double> angin;
        toast::AlignedVector <double> sinout;
        toast::AlignedVector <double> cosout;
        toast::AlignedVector <double> xin;
        toast::AlignedVector <double> yin;
        toast::AlignedVector <double> atanout;
        toast::AlignedVector <double> sqin;
        toast::AlignedVector <double> sqout;
        toast::AlignedVector <double> rsqin;
        toast::AlignedVector <double> rsqout;
        toast::AlignedVector <double> expin;
        toast::AlignedVector <double> expout;
        toast::AlignedVector <double> login;
        toast::AlignedVector <double> logout;
};


class TOASThealpixTest : public ::testing::Test {
    public:

        TOASThealpixTest() {}

        ~TOASThealpixTest() {}

        virtual void SetUp() {}

        virtual void TearDown() {}
};


class TOASTfftTest : public ::testing::Test {
    public:

        TOASTfftTest() {}

        ~TOASTfftTest() {}

        virtual void SetUp() {}

        virtual void TearDown() {}

        void runbatch(int64_t nbatch,
                      toast::FFTPlanReal1D::pshr forward,
                      toast::FFTPlanReal1D::pshr reverse);

        static const int64_t length;
        static const int64_t n;
};


class TOASTcovTest : public ::testing::Test {
    public:

        TOASTcovTest() {}

        ~TOASTcovTest() {}

        virtual void SetUp() {}

        virtual void TearDown() {}

        static const int64_t nsm;
        static const int64_t npix;
        static const int64_t nnz;
        static const int64_t nsamp;
        static const int64_t scale;
};

//
// class TOASTmpiShmemTest : public ::testing::Test {
//     public:
//
//         TOASTmpiShmemTest() {}
//
//         ~TOASTmpiShmemTest() {}
//
//         virtual void SetUp() {}
//
//         virtual void TearDown() {}
//
//         static const size_t n;
// };
//

class TOASTpolyfilterTest : public ::testing::Test {
    public:

        TOASTpolyfilterTest() {}

        ~TOASTpolyfilterTest() {}

        virtual void SetUp() {}

        virtual void TearDown() {}

        static const int64_t order, n;
};


#endif // ifndef TOAST_TEST_HPP
