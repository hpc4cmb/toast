/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#include <cstring>
#include <cmath>

#include <sstream>


#include <Random123/threefry.h>
#include <Random123/uniform.hpp>


typedef r123::Threefry2x64 RNG;


// Native unsigned 64bit integer randoms.

void toast::rng::dist_uint64 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data ) {
    RNG rng;
    RNG::ctr_type c = {{ counter1, counter2 }};
    RNG::ukey_type uk = {{ key1, key2 }};
    RNG::key_type k = uk;

    RNG::ctr_type r;
    for ( size_t i = 0; i < n; ++i ) {
        r = rng ( c, k );
        data[i] = r[0];
        c[1] += 1;
    }
    return;
}


// Uniform double precision values on [0.0, 1.0]

void toast::rng::dist_uniform_01 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data ) {
    RNG rng;
    RNG::ctr_type c = {{ counter1, counter2 }};
    RNG::ukey_type uk = {{ key1, key2 }};
    RNG::key_type k = uk;

    RNG::ctr_type r;
    for ( size_t i = 0; i < n; ++i ) {
        r = rng ( c, k );
        data[i] = r123::u01 < double, uint64_t > ( r[0] );
        c[1] += 1;
    }
    return;
}


// Uniform double precision values on [-1.0, 1.0]

void toast::rng::dist_uniform_11 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data ) {
    RNG rng;
    RNG::ctr_type c = {{ counter1, counter2 }};
    RNG::ukey_type uk = {{ key1, key2 }};
    RNG::key_type k = uk;

    RNG::ctr_type r;
    for ( size_t i = 0; i < n; ++i ) {
        r = rng ( c, k );
        data[i] = r123::uneg11 < double, uint64_t > ( r[0] );
        c[1] += 1;
    }
    return;
}


// Normal distribution.

void toast::rng::dist_normal ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data ) {
    RNG rng;
    RNG::ctr_type c = {{ counter1, counter2 }};
    RNG::ukey_type uk = {{ key1, key2 }};
    RNG::key_type k = uk;

    RNG::ctr_type r;
    for ( size_t i = 0; i < n; ++i ) {
        r = rng ( c, k );
        data[i] = ::sqrt ( -2.0 * ::log ( r123::u01 < double, uint64_t > ( r[0] ) ) ) * ::cos ( TWOPI * r123::u01 < double, uint64_t > ( r[1] ) );
        c[1] += 1;
    }
    return;
}


