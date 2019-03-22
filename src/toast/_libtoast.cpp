
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
}
