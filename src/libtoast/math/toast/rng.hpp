/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_RNG_HPP
#define TOAST_RNG_HPP


namespace toast { namespace rng {

    void dist_uint64 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data );

    void dist_uniform_01 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

    void dist_uniform_11 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

    void dist_normal ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

} }

#endif

