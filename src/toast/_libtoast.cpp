
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>

using size_container = py::detail::any_container <ssize_t>;

template <>
std::string align_format <int8_t> () {
    return std::string("b");
}

template <>
std::string align_format <int16_t> () {
    return std::string("h");
}

template <>
std::string align_format <int32_t> () {
    return std::string("i");
}

template <>
std::string align_format <int64_t> () {
    return std::string("l");
}

template <>
std::string align_format <uint8_t> () {
    return std::string("B");
}

template <>
std::string align_format <uint16_t> () {
    return std::string("H");
}

template <>
std::string align_format <uint32_t> () {
    return std::string("I");
}

template <>
std::string align_format <uint64_t> () {
    return std::string("L");
}

template <>
std::string align_format <float> () {
    return std::string("f");
}

template <>
std::string align_format <double> () {
    return std::string("d");
}

// Helper functions to check numpy array data types and dimensions.


PYBIND11_MODULE(_libtoast, m) {
    m.doc() = R"(
    Interface to C++ TOAST library.

    )";

    // Register aligned array types
    register_aligned <AlignedI8> (m, "AlignedI8");
    register_aligned <AlignedU8> (m, "AlignedU8");
    register_aligned <AlignedI16> (m, "AlignedI16");
    register_aligned <AlignedU16> (m, "AlignedU16");
    register_aligned <AlignedI32> (m, "AlignedI32");
    register_aligned <AlignedU32> (m, "AlignedU32");
    register_aligned <AlignedI64> (m, "AlignedI64");
    register_aligned <AlignedU64> (m, "AlignedU64");
    register_aligned <AlignedF32> (m, "AlignedF32");
    register_aligned <AlignedF64> (m, "AlignedF64");

    init_sys(m);
    init_math_sf(m);
    init_math_rng(m);
    init_math_qarray(m);
    init_math_healpix(m);
    init_math_fft(m);
    init_fod_psd(m);
    init_tod_filter(m);
    init_tod_pointing(m);
    init_tod_simnoise(m);
    init_todmap_scanning(m);
    init_map_cov(m);

    // Internal unit test runner
    m.def(
        "libtoast_tests", [](py::list argv) {
            size_t narg = argv.size();
            std::vector <std::string> argbuffer;
            for (auto const & a : argv) {
                argbuffer.push_back(py::cast <std::string> (a));
            }
            char ** carg = (char **)std::malloc(narg * sizeof(char *));
            for (size_t i = 0; i < narg; ++i) {
                carg[i] = &(argbuffer[i][0]);
            }
            toast::test_runner(narg, carg);
            free(carg);
            return;
        }, py::arg(
            "argv"), R"(
        Run serial compiled tests from the internal libtoast.

        Args:
            argv (list):  The sys.argv or compatible list.

        Returns:
            None

    )");
}
