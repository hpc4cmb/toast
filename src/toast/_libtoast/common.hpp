
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef LIBTOAST_COMMON_HPP
#define LIBTOAST_COMMON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#include <pybind11/stl_bind.h>

#include <toast_internal.hpp>

#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>

namespace py = pybind11;


template <typename T>
std::vector <char> align_format() {
    return std::vector <char> ({'V'});
}

template <>
std::vector <char> align_format <int8_t> ();

template <>
std::vector <char> align_format <int16_t> ();

template <>
std::vector <char> align_format <int32_t> ();

template <>
std::vector <char> align_format <int64_t> ();

template <>
std::vector <char> align_format <uint8_t> ();

template <>
std::vector <char> align_format <uint16_t> ();

template <>
std::vector <char> align_format <uint32_t> ();

template <>
std::vector <char> align_format <uint64_t> ();

template <>
std::vector <char> align_format <float> ();

template <>
std::vector <char> align_format <double> ();


template <typename T>
void pybuffer_check_1D(py::buffer data) {
    auto log = toast::Logger::get();
    py::buffer_info info = data.request();
    std::vector <char> tp = align_format <T> ();
    bool valid = false;
    for (auto const & atp : tp) {
        if (info.format[0] == atp) {
            valid = true;
        }
    }
    if (!valid) {
        std::ostringstream o;
        o << "Python buffer is type '" << info.format
          << "', which is not in compatible list {";
        for (auto const & atp : tp) {
            o << "'" << atp << "',";
        }
        o << "}";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    if (info.ndim != 1) {
        std::ostringstream o;
        o << "Python buffer has " << info.ndim
          << " dimensions instead of one, shape = ";
        for (auto const & d : info.shape) {
            o << d << ", ";
        }
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return;
}

template <typename T>
void pybuffer_check(py::buffer data) {
    auto log = toast::Logger::get();
    py::buffer_info info = data.request();
    std::vector <char> tp = align_format <T> ();
    bool valid = false;
    for (auto const & atp : tp) {
        if (info.format[0] == atp) {
            valid = true;
        }
    }
    if (!valid) {
        std::ostringstream o;
        o << "Python buffer is type '" << info.format
          << "', which is not in compatible list {";
        for (auto const & atp : tp) {
            o << "'" << atp << "',";
        }
        o << "}";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return;
}

template <typename C>
std::unique_ptr <C> aligned_uptr(size_t n) {
    return std::unique_ptr <C> (new C(n));
}

#endif // ifndef LIBTOAST_COMMON_HPP
