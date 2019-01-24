
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>

#include <cmath>


const int64_t TOASTrngTest::size = 11;
const uint64_t TOASTrngTest::counter[] = {1357111317, 888118218888};
const uint64_t TOASTrngTest::key[] = {3405692589, 3131965165};
const uint64_t TOASTrngTest::counter00[] = {0, 0};
const uint64_t TOASTrngTest::key00[] = {0, 0};

// const double rngTest::array_gaussian[] = { -0.602799, 2.141513, -0.433604,
// 0.493275, -0.037459, -0.926340, -0.536562, -0.064849, -0.662582, -1.024292,
// -0.170119 };

const double TOASTrngTest::array_m11[] =
{-0.951008, 0.112014, -0.391117, 0.858437, -0.232332, -0.929797, 0.513278,
 -0.722889, -0.439833, 0.814677, 0.466897};

const double TOASTrngTest::array_01[] =
{0.524496, 0.056007, 0.804442, 0.429218, 0.883834, 0.535102, 0.256639,
 0.638556, 0.780084, 0.407338, 0.233448};

const uint64_t TOASTrngTest::array_uint64[] =
{9675248043493244317ul, 1033143684219887964ul, 14839328367301273822ul,
 7917682351778602270ul, 16303863741333868668ul, 9870884412429777903ul,
 4734154306332135586ul, 11779270208507399991ul, 14390002533568630569ul,
 7514066637753215609ul, 4306362335420736255ul};

// const double rngTest::array00_gaussian[] = { -0.680004, -0.633214,
// -1.523790, -1.847484, -0.427139, 0.991348, 0.601200, 0.481707, -0.085967,
// 0.110980, -1.220734 };

const double TOASTrngTest::array00_m11[] =
{-0.478794, -0.704256, 0.533997, 0.004571, 0.392376, -0.785938, -0.373569,
 0.866371, 0.325575, -0.266422, 0.937621};

const double TOASTrngTest::array00_01[] =
{0.760603, 0.647872, 0.266998, 0.002285, 0.196188, 0.607031, 0.813215,
 0.433185, 0.162788, 0.866789, 0.468810};

const uint64_t TOASTrngTest::array00_uint64[] =
{14030652003081164901ul, 11951131804325250240ul, 4925249918008276254ul,
 42156276261651215ul, 3619028682724454876ul, 11197741606642300638ul,
 15001177968947004470ul, 7990859118804543502ul, 3002902877118036975ul,
 15989435820833075781ul, 8648023362736035120ul};


void TOASTrngTest::SetUp() {
    return;
}

TEST_F(TOASTrngTest, reprod) {
    double result1[size];
    double result2[size];

    toast::rng_dist_normal(size, key[0], key[1], counter[0], counter[1],
                           result1);
    toast::rng_dist_normal(size, key[0], key[1], counter[0], counter[1] + 5,
                           result2);

    for (size_t i = 0; i < size - 5; ++i) {
        ASSERT_NEAR(result1[i + 5], result2[i], 1.0e-4);
    }
}


TEST_F(TOASTrngTest, reprod_multi) {
    // Use one stream per thread
    auto & env = toast::Environment::get();
    int nstream = env.max_threads();

    // The data buffer we will process in segments
    size_t nfull = nstream * size;
    toast::AlignedVector <double> data(nfull);

    // Parameters of all streams.  We are actually processing different
    // chunks of the same stream in this case, so the keys / counters are
    // identical, but the offsets and data pointers are different.
    size_t ndata[nstream];
    uint64_t mkey1[nstream];
    uint64_t mkey2[nstream];
    uint64_t mcounter1[nstream];
    uint64_t mcounter2[nstream];
    double * pdata[nstream];
    for (size_t i = 0; i < nstream; ++i) {
        ndata[i] = size;
        mkey1[i] = key[0];
        mkey2[i] = key[1];
        mcounter1[i] = counter[0];
        mcounter2[i] = counter[1] + i * size;
        pdata[i] = &(data[i * size]);
    }

    toast::rng_multi_dist_normal(nstream, ndata, mkey1, mkey2, mcounter1,
                                 mcounter2, pdata);

    // Compare this to the result of generating the whole stream serially.

    double compare[nfull];
    toast::rng_dist_normal(nfull, key[0], key[1], counter[0], counter[1],
                           compare);

    for (size_t i = 0; i < nfull; ++i) {
        ASSERT_NEAR(compare[i], data[i], 1.0e-4);
    }
}


TEST_F(TOASTrngTest, uniform11) {
    double result[size];

    toast::rng_dist_uniform_11(size, key[0], key[1], counter[0], counter[1],
                               result);

    for (size_t i = 0; i < size; ++i) {
        ASSERT_NEAR(array_m11[i], result[i], 1.0e-4);
    }

    toast::rng_dist_uniform_11(size, key00[0], key00[1], counter00[0],
                               counter00[1], result);

    for (size_t i = 0; i < size; ++i) {
        ASSERT_NEAR(array00_m11[i], result[i], 1.0e-4);
    }
}


TEST_F(TOASTrngTest, uniform11_multi) {
    // Use one stream per thread
    auto & env = toast::Environment::get();
    int nstream = env.max_threads();

    // The data buffer we will process in segments
    size_t nfull = nstream * size;
    toast::AlignedVector <double> data(nfull);

    // Parameters of all streams.  We are actually processing different
    // chunks of the same stream in this case, so the keys / counters are
    // identical, but the offsets and data pointers are different.
    size_t ndata[nstream];
    uint64_t mkey1[nstream];
    uint64_t mkey2[nstream];
    uint64_t mcounter1[nstream];
    uint64_t mcounter2[nstream];
    double * pdata[nstream];
    for (size_t i = 0; i < nstream; ++i) {
        ndata[i] = size;
        mkey1[i] = key[0];
        mkey2[i] = key[1];
        mcounter1[i] = counter[0];
        mcounter2[i] = counter[1] + i * size;
        pdata[i] = &(data[i * size]);
    }

    toast::rng_multi_dist_uniform_11(nstream, ndata, mkey1, mkey2, mcounter1,
                                     mcounter2, pdata);

    // Compare this to the result of generating the whole stream serially.

    double compare[nfull];
    toast::rng_dist_uniform_11(nfull, key[0], key[1], counter[0], counter[1],
                               compare);

    for (size_t i = 0; i < nfull; ++i) {
        ASSERT_NEAR(compare[i], data[i], 1.0e-4);
    }
}


TEST_F(TOASTrngTest, uniform01) {
    double result[size];

    toast::rng_dist_uniform_01(size, key[0], key[1], counter[0], counter[1],
                               result);

    for (size_t i = 0; i < size; ++i) {
        ASSERT_NEAR(array_01[i], result[i], 1.0e-4);
    }

    toast::rng_dist_uniform_01(size, key00[0], key00[1], counter00[0],
                               counter00[1], result);

    for (size_t i = 0; i < size; ++i) {
        ASSERT_NEAR(array00_01[i], result[i], 1.0e-4);
    }
}


TEST_F(TOASTrngTest, uniform01_multi) {
    // Use one stream per thread
    auto & env = toast::Environment::get();
    int nstream = env.max_threads();

    // The data buffer we will process in segments
    size_t nfull = nstream * size;
    toast::AlignedVector <double> data(nfull);

    // Parameters of all streams.  We are actually processing different
    // chunks of the same stream in this case, so the keys / counters are
    // identical, but the offsets and data pointers are different.
    size_t ndata[nstream];
    uint64_t mkey1[nstream];
    uint64_t mkey2[nstream];
    uint64_t mcounter1[nstream];
    uint64_t mcounter2[nstream];
    double * pdata[nstream];
    for (size_t i = 0; i < nstream; ++i) {
        ndata[i] = size;
        mkey1[i] = key[0];
        mkey2[i] = key[1];
        mcounter1[i] = counter[0];
        mcounter2[i] = counter[1] + i * size;
        pdata[i] = &(data[i * size]);
    }

    toast::rng_multi_dist_uniform_01(nstream, ndata, mkey1, mkey2, mcounter1,
                                     mcounter2, pdata);

    // Compare this to the result of generating the whole stream serially.

    double compare[nfull];
    toast::rng_dist_uniform_01(nfull, key[0], key[1], counter[0], counter[1],
                               compare);

    for (size_t i = 0; i < nfull; ++i) {
        ASSERT_NEAR(compare[i], data[i], 1.0e-4);
    }
}


TEST_F(TOASTrngTest, uint64) {
    uint64_t result[size];

    toast::rng_dist_uint64(size, key[0], key[1], counter[0], counter[1],
                           result);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(array_uint64[i], result[i]);
    }

    toast::rng_dist_uint64(size, key00[0], key00[1], counter00[0],
                           counter00[1], result);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(array00_uint64[i], result[i]);
    }
}


TEST_F(TOASTrngTest, uint64_multi) {
    // Use one stream per thread
    auto & env = toast::Environment::get();
    int nstream = env.max_threads();

    // The data buffer we will process in segments
    size_t nfull = nstream * size;
    toast::AlignedVector <uint64_t> data(nfull);

    // Parameters of all streams.  We are actually processing different
    // chunks of the same stream in this case, so the keys / counters are
    // identical, but the offsets and data pointers are different.
    size_t ndata[nstream];
    uint64_t mkey1[nstream];
    uint64_t mkey2[nstream];
    uint64_t mcounter1[nstream];
    uint64_t mcounter2[nstream];
    uint64_t * pdata[nstream];
    for (size_t i = 0; i < nstream; ++i) {
        ndata[i] = size;
        mkey1[i] = key[0];
        mkey2[i] = key[1];
        mcounter1[i] = counter[0];
        mcounter2[i] = counter[1] + i * size;
        pdata[i] = &(data[i * size]);
    }

    toast::rng_multi_dist_uint64(nstream, ndata, mkey1, mkey2, mcounter1,
                                 mcounter2, pdata);

    // Compare this to the result of generating the whole stream serially.

    uint64_t compare[nfull];
    toast::rng_dist_uint64(nfull, key[0], key[1], counter[0], counter[1],
                           compare);

    for (size_t i = 0; i < nfull; ++i) {
        EXPECT_EQ(compare[i], data[i]);
    }
}
