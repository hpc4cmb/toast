
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_math_rng(py::module & m) {
    // RNG math functions

    m.def(
        "rng_dist_uint64", [](uint64_t key1, uint64_t key2,
                              uint64_t counter1, uint64_t counter2,
                              py::ssize_t n) {
            std::vector <py::ssize_t> shp = {n};
            py::dtype dt("u8");
            auto out = AlignedArray::create(shp, dt);
            uint64_t * outraw = reinterpret_cast <uint64_t *> (
                out->data.data());
            toast::rng_dist_uint64(n, key1, key2, counter1, counter2,
                                   outraw);
            return out;
        }, R"(
        Generate random unsigned 64bit integers.

        A new aligned-memory array is created for the result and returned.
        For each sample, the key1, key2, and counter1 values remain fixed
        and the counter2 value is incremented.

        Args:
            key1 (uint64):  The first element of the key.
            key2 (uint64):  The second element of the key.
            counter1 (uint64):  The first element of the counter.
            counter2 (uint64):  The second element of the counter.  This is
                effectively the sample index in the stream defined by the
                other 3 values.
            n (int):  The number of samples to generate.

        Returns:
            (AlignedArray):  a new array with the result.

    )");

    m.def(
        "rng_dist_uint64", [](uint64_t key1, uint64_t key2,
                              uint64_t counter1, uint64_t counter2,
                              py::buffer data) {
            pybuffer_check_uint64_1D(data);
            py::buffer_info info = data.request();
            uint64_t * raw = reinterpret_cast <uint64_t *> (info.ptr);
            toast::rng_dist_uint64(info.size, key1, key2, counter1, counter2,
                                   raw);
            return;
        }, R"(
        Generate random unsigned 64bit integers.

        The provided input array is populated with values.  The dtype of the
        input array should be compatible with unsigned 64bit integers.  To
        guarantee SIMD vectorization, the input array should be aligned
        (i.e. use an AlignedArray).

        Args:
            key1 (uint64):  The first element of the key.
            key2 (uint64):  The second element of the key.
            counter1 (uint64):  The first element of the counter.
            counter2 (uint64):  The second element of the counter.  This is
                effectively the sample index in the stream defined by the
                other 3 values.
            data (array):  The array to populate.

        Returns:
            None.

    )");

    //
    // void rng_dist_uniform_01(size_t n, uint64_t key1, uint64_t key2,
    //                          uint64_t counter1, uint64_t counter2,
    //                          double * data);
    //
    // void rng_dist_uniform_11(size_t n, uint64_t key1, uint64_t key2,
    //                          uint64_t counter1, uint64_t counter2,
    //                          double * data);
    //
    // void rng_dist_normal(size_t n, uint64_t key1, uint64_t key2,
    //                      uint64_t counter1,
    //                      uint64_t counter2, double * data);
    //
    // void rng_multi_dist_uint64(size_t nstream, size_t const * ndata,
    //                            uint64_t const * key1, uint64_t const * key2,
    //                            uint64_t const * counter1,
    //                            uint64_t const * counter2, uint64_t ** data);
    //
    // void rng_multi_dist_uniform_01(size_t nstream, size_t const * ndata,
    //                                uint64_t const * key1,
    //                                uint64_t const * key2,
    //                                uint64_t const * counter1,
    //                                uint64_t const * counter2, double **
    // data);
    //
    // void rng_multi_dist_uniform_11(size_t nstream, size_t const * ndata,
    //                                uint64_t const * key1,
    //                                uint64_t const * key2,
    //                                uint64_t const * counter1,
    //                                uint64_t const * counter2, double **
    // data);
    //
    // void rng_multi_dist_normal(size_t nstream, size_t const * ndata,
    //                            uint64_t const * key1, uint64_t const * key2,
    //                            uint64_t const * counter1,
    //                            uint64_t const * counter2, double ** data);

    return;
}
