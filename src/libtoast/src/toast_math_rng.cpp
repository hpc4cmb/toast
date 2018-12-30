
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_sf.hpp>
#include <toast/math_rng.hpp>

#include <cmath>
#include <vector>
#include <algorithm>

#include <Random123/threefry.h>
#include <Random123/uniform.hpp>


typedef r123::Threefry2x64 RNG;


// Unsigned 64bit random integers
void toast::rng_dist_uint64(size_t n, uint64_t key1, uint64_t key2,
                            uint64_t counter1, uint64_t counter2,
                            uint64_t * data) {
    RNG rng;
    RNG::ukey_type uk = {{key1, key2}};

    if (toast::is_aligned(data)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            data[i] = rng(
                RNG::ctr_type({{counter1, counter2 + i}}),
                RNG::key_type(uk))[0];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            data[i] = rng(
                RNG::ctr_type({{counter1, counter2 + i}}),
                RNG::key_type(uk))[0];
        }
    }
    return;
}

// Uniform double precision values on [0.0, 1.0]
void toast::rng_dist_uniform_01(size_t n,
                                uint64_t key1, uint64_t key2,
                                uint64_t counter1, uint64_t counter2,
                                double * data) {
    RNG rng;
    RNG::ukey_type uk = {{key1, key2}};

    if (toast::is_aligned(data)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            data[i] = r123::u01 <double, uint64_t> (
                rng(RNG::ctr_type({{counter1, counter2 + i}}),
                    RNG::key_type(uk))[0]);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            data[i] = r123::u01 <double, uint64_t> (
                rng(RNG::ctr_type({{counter1, counter2 + i}}),
                    RNG::key_type(uk))[0]);
        }
    }

    return;
}

// Uniform double precision values on [-1.0, 1.0]
void toast::rng_dist_uniform_11(size_t n,
                                uint64_t key1, uint64_t key2,
                                uint64_t counter1, uint64_t counter2,
                                double * data) {
    RNG rng;
    RNG::ukey_type uk = {{key1, key2}};

    if (toast::is_aligned(data)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            data[i] = r123::uneg11 <double, uint64_t> (
                rng(RNG::ctr_type({{counter1, counter2 + i}}),
                    RNG::key_type(uk))[0]);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            data[i] = r123::uneg11 <double, uint64_t> (
                rng(RNG::ctr_type({{counter1, counter2 + i}}),
                    RNG::key_type(uk))[0]);
        }
    }
    return;
}

// Normal distribution.
void toast::rng_dist_normal(size_t n,
                            uint64_t key1, uint64_t key2,
                            uint64_t counter1, uint64_t counter2,
                            double * data) {
    // First compute uniform randoms on [0.0, 1.0)
    toast::simd_array <double> uni(n);

    toast::rng_dist_uniform_01(n, key1, key2, counter1, counter2,
                               uni.data());

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        uni[i] = 2.0 * uni[i] - 1.0;
    }

    // now use the inverse error function

    double * ldata = &(data[0]);

    toast::vfast_erfinv(n, uni.data(), ldata);

    double rttwo = ::sqrt(2.0);

    if (toast::is_aligned(data)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            data[i] *= rttwo;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            data[i] *= rttwo;
        }
    }
    return;
}

// Unsigned 64bit random integers, multiple streams.  The streams may be
// arbitrary lengths with arbitrary starting counters.  Note that with suitable
// input data pointers and counter2 values, this function can
// provide threaded generation of a single stream, or threaded generation of
// data from different streams.
void toast::rng_multi_dist_uint64(size_t nstream,
                                  size_t const * ndata,
                                  uint64_t const * key1,
                                  uint64_t const * key2,
                                  uint64_t const * counter1,
                                  uint64_t const * counter2,
                                  uint64_t ** data) {
    #pragma omp parallel for schedule(dynamic) default(none) shared(nstream, \
    ndata, key1, key2, counter1, counter2, data)
    for (size_t s = 0; s < nstream; ++s) {
        toast::rng_dist_uint64(ndata[s], key1[s], key2[s],
                               counter1[s], counter2[s],
                               data[s]);
    }
    return;
}

// Uniform double precision values on [0.0, 1.0], multiple streams.  The
// streams may be arbitrary lengths with arbitrary starting counters.  Note
// that with suitable input data pointers and counter2 values, this function
// can provide threaded generation of a single stream, or threaded generation
// of data from different streams.
void toast::rng_multi_dist_uniform_01(size_t nstream,
                                      size_t const * ndata,
                                      uint64_t const * key1,
                                      uint64_t const * key2,
                                      uint64_t const * counter1,
                                      uint64_t const * counter2,
                                      double ** data) {
    #pragma omp parallel for schedule(dynamic) default(none) shared(nstream, \
    ndata, key1, key2, counter1, counter2, data)
    for (size_t s = 0; s < nstream; ++s) {
        toast::rng_dist_uniform_01(ndata[s], key1[s], key2[s],
                                   counter1[s], counter2[s],
                                   data[s]);
    }
    return;
}

// Uniform double precision values on [-1.0, 1.0], multiple streams.  The
// streams may be arbitrary lengths with arbitrary starting counters.  Note
// that with suitable input data pointers and counter2 values, this function
// can provide threaded generation of a single stream, or threaded generation
// of data from different streams.
void toast::rng_multi_dist_uniform_11(size_t nstream,
                                      size_t const * ndata,
                                      uint64_t const * key1,
                                      uint64_t const * key2,
                                      uint64_t const * counter1,
                                      uint64_t const * counter2,
                                      double ** data) {
    #pragma omp parallel for schedule(dynamic) default(none) shared(nstream, \
    ndata, key1, key2, counter1, counter2, data)
    for (size_t s = 0; s < nstream; ++s) {
        toast::rng_dist_uniform_11(ndata[s], key1[s], key2[s],
                                   counter1[s], counter2[s],
                                   data[s]);
    }
    return;
}

// Unit variance normal distribution, with multiple streams.  The
// streams may be arbitrary lengths with arbitrary starting counters.  Note
// that with suitable input data pointers and counter2 values, this function
// can provide threaded generation of a single stream, or threaded generation
// of data from different streams.
void toast::rng_multi_dist_normal(size_t nstream,
                                  size_t const * ndata,
                                  uint64_t const * key1,
                                  uint64_t const * key2,
                                  uint64_t const * counter1,
                                  uint64_t const * counter2,
                                  double ** data) {
    #pragma omp parallel for schedule(dynamic) default(none) shared(nstream, \
    ndata, key1, key2, counter1, counter2, data)
    for (size_t s = 0; s < nstream; ++s) {
        toast::rng_dist_normal(ndata[s], key1[s], key2[s],
                               counter1[s], counter2[s],
                               data[s]);
    }
    return;
}
