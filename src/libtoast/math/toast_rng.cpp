/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#include <cstring>
#include <cmath>
#include <sstream>
#include <functional>
#include <algorithm>

#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

#ifdef HAVE_MKL
#  include <mkl.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

//============================================================================//
int get_num_threads()
{
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}
//============================================================================//


typedef r123::Threefry2x64 RNG;


//============================================================================//
// Native unsigned 64bit integer randoms.
void toast::rng::dist_uint64 ( size_t n,
                               uint64_t key1, uint64_t key2,
                               uint64_t counter1, uint64_t counter2,
                               uint64_t* data, size_t beg) {    

    RNG rng;
    RNG::ukey_type uk = {{ key1, key2 }};

    const size_t end = beg + n;
#   pragma omp simd
    for ( size_t i = beg; i < end; ++i ) {
        data[i] = rng(RNG::ctr_type({{ counter1, counter2 + (i-beg) }}),
                      RNG::key_type(uk))[0];
    }

    return;
}


//============================================================================//
// Uniform double precision values on [0.0, 1.0]
void toast::rng::dist_uniform_01 ( size_t n,
                                   uint64_t key1, uint64_t key2,
                                   uint64_t counter1, uint64_t counter2,
                                   double* data, size_t beg) {
  /*
#ifdef HAVE_MKL
    // want little endian (e.g. least bits first)
#   if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    uint64_t _key1 = key1;
    uint64_t _key2 = key2;
    uint64_t _cnt1 = counter1;
    uint64_t _cnt2 = counter2;
#   elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint64_t _key1 = swap_endian<uint64_t>(key1);
    uint64_t _key2 = swap_endian<uint64_t>(key2);
    uint64_t _cnt1 = swap_endian<uint64_t>(counter1);
    uint64_t _cnt2 = swap_endian<uint64_t>(counter2);
#   else
#       error Unknown byte order __FILE__ @ __LINE__
#   endif

    VSLStreamStatePtr stream;

    union {
        uint64_t param4[4];
        uint32_t param8[8];
    } params;

    //params.param4[0] = _key1;
    //params.param4[1] = _key2;
    //params.param4[2] = _cnt1;
    //params.param4[3] = _cnt2;

    params.param8[0] = (uint32_t) _key1;
    params.param8[2] = (uint32_t) _key2;
    params.param8[4] = (uint32_t) _cnt1;
    params.param8[6] = (uint32_t) _cnt2;
    params.param8[1] = params.param8[3] = params.param8[5] = params.param8[7] = 0;

    int ret = vslNewStreamEx(&stream, VSL_BRNG_ARS5, 8, params.param8);
    if (ret == 0)
    {
        ret = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n,
                           data, 0.0, 1.0);
        if (ret == 0)
            ret = vslDeleteStream(&stream);
    }

    if(ret == 0)
        return;
#endif
  */

    RNG rng;
    RNG::ukey_type uk = {{ key1, key2 }};

    const size_t end = beg + n;
#   pragma omp simd
    for ( size_t i = beg; i < end; ++i ) {
        data[i] = r123::u01<double, uint64_t>(
                    rng(RNG::ctr_type({{ counter1, counter2 + (i-beg) }}),
                        RNG::key_type(uk))[0]);
    }

    return;
}


//============================================================================//
// Uniform double precision values on [-1.0, 1.0]
void toast::rng::dist_uniform_11 ( size_t n,
                                   uint64_t key1, uint64_t key2,
                                   uint64_t counter1, uint64_t counter2,
                                   double * data, size_t beg ) {

    RNG rng;
    RNG::ukey_type uk = {{ key1, key2 }};

    const size_t end = beg + n;
#   pragma omp simd
    for ( size_t i = beg; i < end; ++i ) {
        data[i] = r123::uneg11<double, uint64_t>(
                    rng(RNG::ctr_type({{ counter1, counter2 + (i-beg) }}),
                        RNG::key_type(uk))[0]);
    }

    return;
}


//============================================================================//
// Normal distribution.
void toast::rng::dist_normal ( size_t n,
                               uint64_t key1, uint64_t key2,
                               uint64_t counter1, uint64_t counter2,
                               double * data, size_t beg ) {

    // first compute uniform randoms on [0.0, 1.0)

    toast::mem::simd_array<double> uni(n);

    toast::rng::dist_uniform_01 ( n, key1, key2, counter1, counter2, uni );
    for ( size_t i = 0; i < n; ++i ) {
        uni[i] = 2.0 * uni[i] - 1.0;
    }

    // now use the inverse error function

    double* ldata = data+beg;
    toast::sf::fast_erfinv ( n, uni, ldata );

    size_t end = beg + n;
    double rttwo = ::sqrt(2.0);
    for ( size_t i = beg; i < end; ++i ) {
        data[i] *= rttwo;
    }

    return;
}


//============================================================================//
//
//                      MT versions of the above functions
//
//============================================================================//

//============================================================================//
// generic wrapper for calling MT versions of random distribution wrappers
template <typename _Tp, typename _Func>
void dist_mt ( _Func func,
               size_t blocks,      size_t n,
               uint64_t* key1,     uint64_t* key2,
               uint64_t* counter1, uint64_t* counter2,
               _Tp* data )
{
    // the lambda function type
    typedef std::function<void(const size_t&, const size_t&)> lambda_t;

    // lambda function to run in parallel
    auto _run = [=] (const size_t& _beg, const size_t& _end)
    {
        constexpr size_t offset = 0;
        for(size_t i = _beg; i < _end; ++i)
            func(n, key1[i], key2[i], counter1[i], counter2[i],
                 data, offset + i*n);
    };

    // use preprocessor defined usage of MT model
    toast::math::execute_mt<size_t, lambda_t>(_run, 0, blocks, 1);
}


//============================================================================//
// Native unsigned 64bit integer randoms (MT-version)
void toast::rng::mt::dist_uint64 ( size_t blocks,      size_t n,
                                   uint64_t* key1,     uint64_t* key2,
                                   uint64_t* counter1, uint64_t* counter2,
                                   uint64_t* data )
{
    // call dist_uint64 in parallel
    dist_mt(toast::rng::dist_uint64,
            blocks, n, key1, key2, counter1, counter2, data);
}


//============================================================================//
// Uniform double precision values on [0.0, 1.0] (MT-version)
void toast::rng::mt::dist_uniform_01 ( size_t blocks,      size_t n,
                                       uint64_t* key1,     uint64_t* key2,
                                       uint64_t* counter1, uint64_t* counter2,
                                       double* data )
{
    // call dist_uniform_01 in parallel
    dist_mt(toast::rng::dist_uniform_01,
            blocks, n, key1, key2, counter1, counter2, data);
}


//============================================================================//
// Uniform double precision values on [-1.0, 1.0] (MT-version)
void toast::rng::mt::dist_uniform_11 ( size_t blocks,      size_t n,
                                       uint64_t* key1,     uint64_t* key2,
                                       uint64_t* counter1, uint64_t* counter2,
                                       double* data )
{
    // call dist_uniform_11 in parallel
    dist_mt(toast::rng::dist_uniform_11,
            blocks, n, key1, key2, counter1, counter2, data);
}

//============================================================================//
// Normal distribution (MT-version)
void toast::rng::mt::dist_normal ( size_t blocks,      size_t n,
                                   uint64_t* key1,     uint64_t* key2,
                                   uint64_t* counter1, uint64_t* counter2,
                                   double* data )
{
    // call dist_normal in parallel
    dist_mt(toast::rng::dist_normal,
            blocks, n, key1, key2, counter1, counter2, data);
}


//============================================================================//

