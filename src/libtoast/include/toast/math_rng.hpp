
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_RNG_HPP
#define TOAST_MATH_RNG_HPP

#include <cstddef>
#include <cstdint>


namespace toast {
void rng_dist_uint64(size_t n, uint64_t key1, uint64_t key2, uint64_t counter1,
                     uint64_t counter2, uint64_t * data);

void rng_dist_uniform_01(size_t n, uint64_t key1, uint64_t key2,
                         uint64_t counter1, uint64_t counter2, double * data);

void rng_dist_uniform_11(size_t n, uint64_t key1, uint64_t key2,
                         uint64_t counter1, uint64_t counter2, double * data);

void rng_dist_normal(size_t n, uint64_t key1, uint64_t key2, uint64_t counter1,
                     uint64_t counter2, double * data);

void rng_multi_dist_uint64(size_t nstream, size_t const * ndata,
                           uint64_t const * key1, uint64_t const * key2,
                           uint64_t const * counter1,
                           uint64_t const * counter2, uint64_t ** data);

void rng_multi_dist_uniform_01(size_t nstream, size_t const * ndata,
                               uint64_t const * key1, uint64_t const * key2,
                               uint64_t const * counter1,
                               uint64_t const * counter2, double ** data);

void rng_multi_dist_uniform_11(size_t nstream, size_t const * ndata,
                               uint64_t const * key1, uint64_t const * key2,
                               uint64_t const * counter1,
                               uint64_t const * counter2, double ** data);

void rng_multi_dist_normal(size_t nstream, size_t const * ndata,
                           uint64_t const * key1, uint64_t const * key2,
                           uint64_t const * counter1,
                           uint64_t const * counter2, double ** data);
}

#endif // ifndef TOAST_RNG_HPP
