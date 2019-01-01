
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

        std::vector <double, toast::simd_allocator <double> > q1;
        std::vector <double, toast::simd_allocator <double> > q1inv;
        std::vector <double, toast::simd_allocator <double> > q2;
        std::vector <double, toast::simd_allocator <double> > qtonormalize;
        std::vector <double, toast::simd_allocator <double> > qnormalized;
        std::vector <double, toast::simd_allocator <double> > vec;
        std::vector <double, toast::simd_allocator <double> > vec2;
        std::vector <double, toast::simd_allocator <double> > qeasy;
        std::vector <double, toast::simd_allocator <double> > mult_result;
        std::vector <double, toast::simd_allocator <double> > rot_by_q1;
        std::vector <double, toast::simd_allocator <double> > rot_by_q2;
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
        std::vector <double, toast::simd_allocator <double> > angin;
        std::vector <double, toast::simd_allocator <double> > sinout;
        std::vector <double, toast::simd_allocator <double> > cosout;
        std::vector <double, toast::simd_allocator <double> > xin;
        std::vector <double, toast::simd_allocator <double> > yin;
        std::vector <double, toast::simd_allocator <double> > atanout;
        std::vector <double, toast::simd_allocator <double> > sqin;
        std::vector <double, toast::simd_allocator <double> > sqout;
        std::vector <double, toast::simd_allocator <double> > rsqin;
        std::vector <double, toast::simd_allocator <double> > rsqout;
        std::vector <double, toast::simd_allocator <double> > expin;
        std::vector <double, toast::simd_allocator <double> > expout;
        std::vector <double, toast::simd_allocator <double> > login;
        std::vector <double, toast::simd_allocator <double> > logout;
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
