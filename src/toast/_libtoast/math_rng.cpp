
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_math_rng(py::module & m) {
    // RNG math functions

    m.def(
        "rng_dist_uint64", [](uint64_t key1, uint64_t key2,
                              uint64_t counter1, uint64_t counter2,
                              py::buffer data) {
            pybuffer_check_1D <uint64_t> (data);
            py::buffer_info info = data.request();
            uint64_t * raw = reinterpret_cast <uint64_t *> (info.ptr);
            toast::rng_dist_uint64(info.size, key1, key2, counter1, counter2,
                                   raw);
            return;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "data"), R"(
        Generate random unsigned 64bit integers.

        The provided input array is populated with values.  The dtype of the
        input array should be compatible with unsigned 64bit integers.  To
        guarantee SIMD vectorization, the input array should be aligned
        (i.e. use an AlignedU64).

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

    m.def(
        "rng_dist_uniform_01", [](uint64_t key1, uint64_t key2,
                                  uint64_t counter1, uint64_t counter2,
                                  py::buffer data) {
            pybuffer_check_1D <double> (data);
            py::buffer_info info = data.request();
            double * raw = reinterpret_cast <double *> (info.ptr);
            toast::rng_dist_uniform_01(info.size, key1, key2, counter1,
                                       counter2, raw);
            return;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "data"), R"(
        Generate uniform randoms on the interval [0.0, 1.0].

        The provided input array is populated with values.  The dtype of the
        input array should be compatible with 64bit floating point values.  To
        guarantee SIMD vectorization, the input array should be aligned
        (i.e. use an AlignedF64).

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

    m.def(
        "rng_dist_uniform_11", [](uint64_t key1, uint64_t key2,
                                  uint64_t counter1, uint64_t counter2,
                                  py::buffer data) {
            pybuffer_check_1D <double> (data);
            py::buffer_info info = data.request();
            double * raw = reinterpret_cast <double *> (info.ptr);
            toast::rng_dist_uniform_11(info.size, key1, key2, counter1,
                                       counter2, raw);
            return;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "data"), R"(
        Generate uniform randoms on the interval [-1.0, 1.0].

        The provided input array is populated with values.  The dtype of the
        input array should be compatible with 64bit floating point values.  To
        guarantee SIMD vectorization, the input array should be aligned
        (i.e. use an AlignedF64).

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

    m.def(
        "rng_dist_normal", [](uint64_t key1, uint64_t key2,
                              uint64_t counter1, uint64_t counter2,
                              py::buffer data) {
            pybuffer_check_1D <double> (data);
            py::buffer_info info = data.request();
            double * raw = reinterpret_cast <double *> (info.ptr);
            toast::rng_dist_normal(info.size, key1, key2, counter1,
                                   counter2, raw);
            return;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "data"), R"(
        Generate samples from a unit-variance gaussian distribution.

        The provided input array is populated with values.  The dtype of the
        input array should be compatible with 64bit floating point values.  To
        guarantee SIMD vectorization, the input array should be aligned
        (i.e. use an AlignedF64).

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

    // Generate multiple streams in parallel.

    m.def(
        "rng_multi_dist_uint64", [](py::buffer key1, py::buffer key2,
                                    py::buffer counter1, py::buffer counter2,
                                    std::vector <size_t> lengths) {
            pybuffer_check_1D <uint64_t> (key1);
            pybuffer_check_1D <uint64_t> (key2);
            pybuffer_check_1D <uint64_t> (counter1);
            pybuffer_check_1D <uint64_t> (counter2);
            py::buffer_info info_k1 = key1.request();
            py::buffer_info info_k2 = key2.request();
            py::buffer_info info_c1 = counter1.request();
            py::buffer_info info_c2 = counter2.request();
            uint64_t * raw_k1 = reinterpret_cast <uint64_t *> (info_k1.ptr);
            uint64_t * raw_k2 = reinterpret_cast <uint64_t *> (info_k2.ptr);
            uint64_t * raw_c1 = reinterpret_cast <uint64_t *> (info_c1.ptr);
            uint64_t * raw_c2 = reinterpret_cast <uint64_t *> (info_c2.ptr);
            size_t nstream = lengths.size();
            if ((info_k1.size != nstream) || (info_k2.size != nstream) ||
                (info_c1.size != nstream) || (info_c2.size != nstream)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "buffers have different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            py::list out;
            for (size_t i = 0; i < nstream; ++i) {
                out.append(py::cast(toast::AlignedU64(lengths[i])));
            }
            std::vector <uint64_t *> bufs(nstream);
            for (size_t i = 0; i < nstream; ++i) {
                auto ap = out[i].cast <toast::AlignedU64 *> ();
                bufs[i] = ap->data();
            }
            toast::rng_multi_dist_uint64(nstream, lengths.data(), raw_k1,
                                         raw_k2, raw_c1, raw_c2, bufs.data());
            return out;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "lengths"), R"(
        Generate multiple streams of random unsigned 64bit integers.

        The streams may be arbitrary lengths with arbitrary starting
        counters.  Note that with suitable key and counter values, this
        function can provide threaded generation of a single stream, or
        threaded generation of data from different streams.

        A list of aligned memory buffers is returned.

        Args:
            key1 (array):  The key1 values for all streams.
            key2 (array):  The key2 values for all streams.
            counter1 (array):  The counter1 values for all streams.
            counter2 (array):  The counter2 values for all streams.  This is
                effectively the sample index in the stream defined by the
                other 3 values.
            lengths (array):  The number of samples to generate for each
                stream.

        Returns:
            (list):  list of AlignedU64 buffers, one per stream.

    )");

    m.def(
        "rng_multi_dist_uniform_01", [](py::buffer key1, py::buffer key2,
                                        py::buffer counter1,
                                        py::buffer counter2,
                                        std::vector <size_t> lengths) {
            pybuffer_check_1D <uint64_t> (key1);
            pybuffer_check_1D <uint64_t> (key2);
            pybuffer_check_1D <uint64_t> (counter1);
            pybuffer_check_1D <uint64_t> (counter2);
            py::buffer_info info_k1 = key1.request();
            py::buffer_info info_k2 = key2.request();
            py::buffer_info info_c1 = counter1.request();
            py::buffer_info info_c2 = counter2.request();
            uint64_t * raw_k1 = reinterpret_cast <uint64_t *> (info_k1.ptr);
            uint64_t * raw_k2 = reinterpret_cast <uint64_t *> (info_k2.ptr);
            uint64_t * raw_c1 = reinterpret_cast <uint64_t *> (info_c1.ptr);
            uint64_t * raw_c2 = reinterpret_cast <uint64_t *> (info_c2.ptr);
            size_t nstream = lengths.size();
            if ((info_k1.size != nstream) || (info_k2.size != nstream) ||
                (info_c1.size != nstream) || (info_c2.size != nstream)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "buffers have different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            py::list out;
            for (size_t i = 0; i < nstream; ++i) {
                out.append(py::cast(toast::AlignedF64(lengths[i])));
            }
            std::vector <double *> bufs(nstream);
            for (size_t i = 0; i < nstream; ++i) {
                auto ap = out[i].cast <toast::AlignedF64 *> ();
                bufs[i] = ap->data();
            }
            toast::rng_multi_dist_uniform_01(nstream, lengths.data(), raw_k1,
                                             raw_k2, raw_c1, raw_c2,
                                             bufs.data());
            return out;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "lengths"), R"(
        Generate multiple streams of uniform randoms on interval [0.0, 1.0].

        The streams may be arbitrary lengths with arbitrary starting
        counters.  Note that with suitable key and counter values, this
        function can provide threaded generation of a single stream, or
        threaded generation of data from different streams.

        A list of aligned memory buffers is returned.

        Args:
            key1 (array):  The key1 values for all streams.
            key2 (array):  The key2 values for all streams.
            counter1 (array):  The counter1 values for all streams.
            counter2 (array):  The counter2 values for all streams.  This is
                effectively the sample index in the stream defined by the
                other 3 values.
            lengths (array):  The number of samples to generate for each
                stream.

        Returns:
            (list):  list of AlignedF64 buffers, one per stream.

    )");

    m.def(
        "rng_multi_dist_uniform_11", [](py::buffer key1, py::buffer key2,
                                        py::buffer counter1,
                                        py::buffer counter2,
                                        std::vector <size_t> lengths) {
            pybuffer_check_1D <uint64_t> (key1);
            pybuffer_check_1D <uint64_t> (key2);
            pybuffer_check_1D <uint64_t> (counter1);
            pybuffer_check_1D <uint64_t> (counter2);
            py::buffer_info info_k1 = key1.request();
            py::buffer_info info_k2 = key2.request();
            py::buffer_info info_c1 = counter1.request();
            py::buffer_info info_c2 = counter2.request();
            uint64_t * raw_k1 = reinterpret_cast <uint64_t *> (info_k1.ptr);
            uint64_t * raw_k2 = reinterpret_cast <uint64_t *> (info_k2.ptr);
            uint64_t * raw_c1 = reinterpret_cast <uint64_t *> (info_c1.ptr);
            uint64_t * raw_c2 = reinterpret_cast <uint64_t *> (info_c2.ptr);
            size_t nstream = lengths.size();
            if ((info_k1.size != nstream) || (info_k2.size != nstream) ||
                (info_c1.size != nstream) || (info_c2.size != nstream)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "buffers have different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            py::list out;
            for (size_t i = 0; i < nstream; ++i) {
                out.append(py::cast(toast::AlignedF64(lengths[i])));
            }
            std::vector <double *> bufs(nstream);
            for (size_t i = 0; i < nstream; ++i) {
                auto ap = out[i].cast <toast::AlignedF64 *> ();
                bufs[i] = ap->data();
            }
            toast::rng_multi_dist_uniform_11(nstream, lengths.data(), raw_k1,
                                             raw_k2, raw_c1, raw_c2,
                                             bufs.data());
            return out;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "lengths"), R"(
        Generate multiple streams of uniform randoms on interval [-1.0, 1.0].

        The streams may be arbitrary lengths with arbitrary starting
        counters.  Note that with suitable key and counter values, this
        function can provide threaded generation of a single stream, or
        threaded generation of data from different streams.

        A list of aligned memory buffers is returned.

        Args:
            key1 (array):  The key1 values for all streams.
            key2 (array):  The key2 values for all streams.
            counter1 (array):  The counter1 values for all streams.
            counter2 (array):  The counter2 values for all streams.  This is
                effectively the sample index in the stream defined by the
                other 3 values.
            lengths (array):  The number of samples to generate for each
                stream.

        Returns:
            (list):  list of AlignedF64 buffers, one per stream.

    )");

    m.def(
        "rng_multi_dist_normal", [](py::buffer key1, py::buffer key2,
                                    py::buffer counter1,
                                    py::buffer counter2,
                                    std::vector <size_t> lengths) {
            pybuffer_check_1D <uint64_t> (key1);
            pybuffer_check_1D <uint64_t> (key2);
            pybuffer_check_1D <uint64_t> (counter1);
            pybuffer_check_1D <uint64_t> (counter2);
            py::buffer_info info_k1 = key1.request();
            py::buffer_info info_k2 = key2.request();
            py::buffer_info info_c1 = counter1.request();
            py::buffer_info info_c2 = counter2.request();
            uint64_t * raw_k1 = reinterpret_cast <uint64_t *> (info_k1.ptr);
            uint64_t * raw_k2 = reinterpret_cast <uint64_t *> (info_k2.ptr);
            uint64_t * raw_c1 = reinterpret_cast <uint64_t *> (info_c1.ptr);
            uint64_t * raw_c2 = reinterpret_cast <uint64_t *> (info_c2.ptr);
            size_t nstream = lengths.size();
            if ((info_k1.size != nstream) || (info_k2.size != nstream) ||
                (info_c1.size != nstream) || (info_c2.size != nstream)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "buffers have different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            py::list out;
            for (size_t i = 0; i < nstream; ++i) {
                out.append(py::cast(toast::AlignedF64(lengths[i])));
            }
            std::vector <double *> bufs(nstream);
            for (size_t i = 0; i < nstream; ++i) {
                auto ap = out[i].cast <toast::AlignedF64 *> ();
                bufs[i] = ap->data();
            }
            toast::rng_multi_dist_normal(nstream, lengths.data(), raw_k1,
                                         raw_k2, raw_c1, raw_c2, bufs.data());
            return out;
        }, py::arg("key1"), py::arg("key2"), py::arg("counter1"),
        py::arg("counter2"), py::arg(
            "lengths"), R"(
        Generate multiple streams from unit-variance gaussian distributions.

        The streams may be arbitrary lengths with arbitrary starting
        counters.  Note that with suitable key and counter values, this
        function can provide threaded generation of a single stream, or
        threaded generation of data from different streams.

        A list of aligned memory buffers is returned.

        Args:
            key1 (array):  The key1 values for all streams.
            key2 (array):  The key2 values for all streams.
            counter1 (array):  The counter1 values for all streams.
            counter2 (array):  The counter2 values for all streams.  This is
                effectively the sample index in the stream defined by the
                other 3 values.
            lengths (array):  The number of samples to generate for each
                stream.

        Returns:
            (list):  list of AlignedF64 buffers, one per stream.

    )");

    return;
}
