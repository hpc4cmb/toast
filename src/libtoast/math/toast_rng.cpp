/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#include <cstring>
#include <cmath>

#include <sstream>

#ifdef HAVE_MKL
#  include <mkl.h>
#endif


#include <Random123/threefry.h>
#include <Random123/uniform.hpp>


typedef r123::Threefry2x64 RNG;


// Native unsigned 64bit integer randoms.

void toast::rng::dist_uint64 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data ) {
    
    #pragma omp parallel default(shared)
    {
        RNG rng;
        RNG::ctr_type c;
        RNG::ukey_type uk = {{ key1, key2 }};
        RNG::key_type k = uk;
        RNG::ctr_type r;

        c[0] = counter1;

        #pragma omp for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            c[1] = counter2 + i;
            r = rng ( c, k );
            data[i] = r[0];
        }
    }

    return;
}


// Uniform double precision values on [0.0, 1.0]

void toast::rng::dist_uniform_01 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data ) {

    #pragma omp parallel default(shared)
    {
        RNG rng;
        RNG::ctr_type c;
        RNG::ukey_type uk = {{ key1, key2 }};
        RNG::key_type k = uk;
        RNG::ctr_type r;

        c[0] = counter1;

        #pragma omp for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            c[1] = counter2 + i;
            r = rng ( c, k );
            data[i] = r123::u01 < double, uint64_t > ( r[0] );
        }
    }

    return;
}


// Uniform double precision values on [-1.0, 1.0]

void toast::rng::dist_uniform_11 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data ) {

    #pragma omp parallel default(shared)
    {
        RNG rng;
        RNG::ctr_type c;
        RNG::ukey_type uk = {{ key1, key2 }};
        RNG::key_type k = uk;
        RNG::ctr_type r;

        c[0] = counter1;

        #pragma omp for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            c[1] = counter2 + i;
            r = rng ( c, k );
            data[i] = r123::uneg11 < double, uint64_t > ( r[0] );
        }
    }

    return;
}


// Normal distribution.

void toast::rng::dist_normal ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data ) {

    // first compute uniform randoms on [0.0, 1.0)

    toast::mem::simd_array<double> uni(n);

    toast::rng::dist_uniform_01 ( n, key1, key2, counter1, counter2, uni );
    for ( size_t i = 0; i < n; ++i ) {
        uni[i] = 2.0 * uni[i] - 1.0;
    }

    // now use the inverse error function

    toast::sf::fast_erfinv ( n, uni, data );

    double rttwo = ::sqrt(2.0);
    for ( size_t i = 0; i < n; ++i ) {
        data[i] *= rttwo;
    }

    return;
}


