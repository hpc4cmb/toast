/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const int64_t TOASTrngTest::size = 11;
const uint64_t TOASTrngTest::counter[] = { 1357111317, 888118218888 };
const uint64_t TOASTrngTest::key[] = { 3405692589, 3131965165 };
const uint64_t TOASTrngTest::counter00[] = { 0, 0 };
const uint64_t TOASTrngTest::key00[] = { 0, 0 };

//const double rngTest::array_gaussian[] = { -0.602799, 2.141513, -0.433604, 0.493275, -0.037459, -0.926340, -0.536562, -0.064849, -0.662582, -1.024292, -0.170119 };

const double TOASTrngTest::array_m11[] = { -0.951008, 0.112014, -0.391117, 0.858437, -0.232332, -0.929797, 0.513278, -0.722889, -0.439833, 0.814677, 0.466897 };
const double TOASTrngTest::array_01[] = { 0.524496, 0.056007, 0.804442, 0.429218, 0.883834, 0.535102, 0.256639, 0.638556, 0.780084, 0.407338, 0.233448 };
const uint64_t TOASTrngTest::array_uint64[] = { 9675248043493244317ul, 1033143684219887964ul, 14839328367301273822ul, 7917682351778602270ul, 16303863741333868668ul, 9870884412429777903ul, 4734154306332135586ul, 11779270208507399991ul, 14390002533568630569ul, 7514066637753215609ul, 4306362335420736255ul };

//const double rngTest::array00_gaussian[] = { -0.680004, -0.633214, -1.523790, -1.847484, -0.427139, 0.991348, 0.601200, 0.481707, -0.085967, 0.110980, -1.220734 };

const double TOASTrngTest::array00_m11[] = { -0.478794, -0.704256, 0.533997, 0.004571, 0.392376, -0.785938, -0.373569, 0.866371, 0.325575, -0.266422, 0.937621 };
const double TOASTrngTest::array00_01[] = { 0.760603, 0.647872, 0.266998, 0.002285, 0.196188, 0.607031, 0.813215, 0.433185, 0.162788, 0.866789, 0.468810 };
const uint64_t TOASTrngTest::array00_uint64[] = { 14030652003081164901ul, 11951131804325250240ul, 4925249918008276254ul, 42156276261651215ul, 3619028682724454876ul, 11197741606642300638ul, 15001177968947004470ul, 7990859118804543502ul, 3002902877118036975ul, 15989435820833075781ul, 8648023362736035120ul };

const size_t num_threads = 4;

void TOASTrngTest::SetUp () {
    return;
}

//============================================================================//
// helper function to generate MT array of keys and counters
uint64_t** construct(const size_t nthread,
                     const uint64_t keys[], const uint64_t counters[])
{
    uint64_t** data = new uint64_t*[4];
    for(size_t i = 0; i < 4; ++i)
        data[i] = new uint64_t[nthread];

    for(size_t j = 0; j < nthread; ++j)
    {
        data[0][j] = keys[0];
        data[1][j] = keys[1];
        data[2][j] = counters[0];
        data[3][j] = counters[1];
    }
    return data;
}

//============================================================================//
// helper function to free memory from MT array of keys and counters
void deconstruct(uint64_t** data)
{
    for(size_t i = 0; i < 4; ++i)
        delete [] data[i];
}

//============================================================================//

TEST_F( TOASTrngTest, reprod ) {
    double result1[size];
    double result2[size];

    rng::dist_normal ( size, key[0], key[1], counter[0], counter[1], result1 );
    rng::dist_normal ( size, key[0], key[1], counter[0], counter[1]+5, result2 );

    for ( size_t i = 0; i < size-5; ++i ) {
        ASSERT_NEAR( result1[i+5], result2[i], 1.0e-4 );
    }
}

//============================================================================//

TEST_F( TOASTrngTest, reprod_mt ) {
    const size_t nthread = num_threads;

    toast::mem::simd_array<double> result1(nthread*size);
    toast::mem::simd_array<double> result2(nthread*size);

    uint64_t counter5[] = { counter[0], (uint64_t) counter[1]+5 };
    uint64_t** d1 = construct(nthread, key, counter);
    uint64_t** d2 = construct(nthread, key, counter5);

    rng::mt::dist_normal (nthread, size, d1[0], d1[1], d1[2], d1[3], result1 );
    rng::mt::dist_normal (nthread, size, d2[0], d2[1], d2[2], d2[3], result2 );

    deconstruct(d1);
    deconstruct(d2);

    for ( size_t j = 0; j < nthread; ++j)
    {
        size_t offset = j*size;
        for ( size_t i = 0; i < size-5; ++i )
            ASSERT_NEAR( result1[offset + (i+5)], result2[offset + i], 1.0e-4 );
    }
}

//============================================================================//

TEST_F( TOASTrngTest, uniform11 ) {
    double result[size];

    rng::dist_uniform_11 ( size, key[0], key[1], counter[0], counter[1], result );

    for ( size_t i = 0; i < size; ++i ) {
        ASSERT_NEAR( array_m11[i], result[i], 1.0e-4 );
    }

    rng::dist_uniform_11 ( size, key00[0], key00[1], counter00[0], counter00[1], result );

    for ( size_t i = 0; i < size; ++i ) {
        ASSERT_NEAR( array00_m11[i], result[i], 1.0e-4 );
    }
}

//============================================================================//

TEST_F( TOASTrngTest, uniform11_mt ) {
    const size_t nthread = num_threads;
    toast::mem::simd_array<double> result(nthread*size);

    {
        uint64_t** d = construct(nthread, key, counter);
        rng::mt::dist_uniform_11 (nthread, size, d[0], d[1], d[2], d[3], result);
        deconstruct(d);
    }

    for ( size_t j = 0; j < nthread; ++j)
    {
        size_t offset = j*size;
        for ( size_t i = 0; i < size; ++i )
            ASSERT_NEAR( array_m11[i], result[i + offset], 1.0e-4 );
    }

    {
        uint64_t** d = construct(nthread, key00, counter00);
        rng::mt::dist_uniform_11 (nthread, size, d[0], d[1], d[2], d[3], result);
        deconstruct(d);
    }

    for ( size_t j = 0; j < nthread; ++j)
    {
        size_t offset = j*size;
        for ( size_t i = 0; i < size; ++i )
            ASSERT_NEAR( array00_m11[i], result[i + offset], 1.0e-4 );
    }
}

//============================================================================//

TEST_F( TOASTrngTest, uniform01 ) {
    double result[size];

    rng::dist_uniform_01 ( size, key[0], key[1], counter[0], counter[1], result );

    for ( size_t i = 0; i < size; ++i ) {
        ASSERT_NEAR( array_01[i], result[i], 1.0e-4 );
    }

    rng::dist_uniform_01 ( size, key00[0], key00[1], counter00[0], counter00[1], result );

    for ( size_t i = 0; i < size; ++i ) {
        ASSERT_NEAR( array00_01[i], result[i], 1.0e-4 );
    }
}

//============================================================================//

TEST_F( TOASTrngTest, uniform10_mt ) {
    const size_t nthread = num_threads;
    toast::mem::simd_array<double> result(nthread*size);

    {
        uint64_t** d = construct(nthread, key, counter);
        rng::mt::dist_uniform_01(nthread, size, d[0], d[1], d[2], d[3], result);
        deconstruct(d);
    }

    for ( size_t j = 0; j < nthread; ++j)
    {
        size_t offset = j*size;
        for ( size_t i = 0; i < size; ++i )
            ASSERT_NEAR(array_01[i], result[i + offset], 1.0e-4 );
    }

    {
        uint64_t** d = construct(nthread, key00, counter00);
        rng::mt::dist_uniform_01(nthread, size, d[0], d[1], d[2], d[3], result);
        deconstruct(d);
    }

    for ( size_t j = 0; j < nthread; ++j)
    {
        size_t offset = j*size;
        for ( size_t i = 0; i < size; ++i )
            ASSERT_NEAR(array00_01[i], result[i + offset], 1.0e-4 );
    }
}

//============================================================================//

TEST_F( TOASTrngTest, uint64 ) {
    uint64_t result[size];

    rng::dist_uint64 ( size, key[0], key[1], counter[0], counter[1], result );

    for ( size_t i = 0; i < size; ++i ) {
        EXPECT_EQ( array_uint64[i], result[i] );
    }

    rng::dist_uint64 ( size, key00[0], key00[1], counter00[0], counter00[1], result );

    for ( size_t i = 0; i < size; ++i ) {
        EXPECT_EQ( array00_uint64[i], result[i] );
    }
}

//============================================================================//

TEST_F( TOASTrngTest, uint64_mt ) {
    const size_t nthread = num_threads;
    toast::mem::simd_array<uint64_t> result(nthread*size);

    {
        uint64_t** d = construct(nthread, key, counter);
        rng::mt::dist_uint64(nthread, size, d[0], d[1], d[2], d[3], result);
        deconstruct(d);
    }

    for ( size_t j = 0; j < nthread; ++j)
    {
        size_t offset = j*size;
        for ( size_t i = 0; i < size; ++i )
            EXPECT_EQ( array_uint64[i], result[i + offset] );
    }

    {
        uint64_t** d = construct(nthread, key00, counter00);
        rng::mt::dist_uint64(nthread, size, d[0], d[1], d[2], d[3], result);
        deconstruct(d);
    }

    for ( size_t j = 0; j < nthread; ++j)
    {
        size_t offset = j*size;
        for ( size_t i = 0; i < size; ++i )
            EXPECT_EQ( array00_uint64[i], result[i + offset] );
    }
}

//============================================================================//
