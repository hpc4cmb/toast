
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef _LIBTOAST_COMMON_HPP
#define _LIBTOAST_COMMON_HPP

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
#include <unordered_map>
#include <map>

namespace py = pybind11;


std::string get_format(std::string const & input);


template <typename T>
T * extract_buffer(
    py::buffer data,
    char const * name,
    size_t assert_dims,
    std::vector <int64_t> & shape,
    std::vector <int64_t> assert_shape
) {
    // Get buffer info structure
    auto info = data.request();

    // Extract the format character for the target type
    std::string target_format = get_format(py::format_descriptor <T>::format());

    // Extract the format for the input buffer
    std::string buffer_format = get_format(info.format);

    // Verify format string
    if (buffer_format != target_format) {
        // On 64bit linux, numpy is internally inconsistent with the
        // format codes for int64_t and long long:
        //   https://github.com/numpy/numpy/issues/12264
        // Here we treat them as equivalent.
        if (((buffer_format == "q") || (buffer_format == "l"))
            && ((target_format == "q") || (target_format == "l"))) {
            // What could possibly go wrong...
        } else {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Object " << name << " has format \"" << buffer_format
              << "\" instead of \"" << target_format << "\"";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }

    // Verify itemsize
    if (info.itemsize != sizeof(T)) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "Object " << name << " has item size of "
          << info.itemsize << " instead of " << sizeof(T);
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    // Verify number of dimensions
    if (info.ndim != assert_dims) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "Object " << name << " has " << info.ndim
          << " dimensions instead of " << assert_dims;
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    // Get array dimensions
    for (py::ssize_t d = 0; d < info.ndim; d++) {
        shape[d] = info.shape[d];
    }

    // Check strides and verify that memory is contiguous
    size_t stride = info.itemsize;
    for (int d = info.ndim - 1; d >= 0; d--) {
        if (info.strides[d] != stride) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Object " << name
              << ": python buffers must be contiguous in memory.";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
        stride *= info.shape[d];
    }

    // If the user wants to verify any of the dimensions, do that
    for (py::ssize_t d = 0; d < info.ndim; d++) {
        if (assert_shape[d] >= 0) {
            // We are checking this dimension
            if (assert_shape[d] != shape[d]) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Object " << name << " dimension " << d
                  << " has length " << shape[d]
                  << " instead of " << assert_shape[d];
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
        }
    }

    return static_cast <T *> (info.ptr);
}

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
        } else if ((info.format[0] == '<') && (info.format[1] == atp)) {
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
        } else if ((info.format[0] == '<') && (info.format[1] == atp)) {
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
