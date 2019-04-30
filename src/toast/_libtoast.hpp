
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

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

// Aligned memory vector types

using AlignedI8 = toast::AlignedVector <int8_t>;
using AlignedU8 = toast::AlignedVector <uint8_t>;
using AlignedI16 = toast::AlignedVector <int16_t>;
using AlignedU16 = toast::AlignedVector <uint16_t>;
using AlignedI32 = toast::AlignedVector <int32_t>;
using AlignedU32 = toast::AlignedVector <uint32_t>;
using AlignedI64 = toast::AlignedVector <int64_t>;
using AlignedU64 = toast::AlignedVector <uint64_t>;
using AlignedF32 = toast::AlignedVector <float>;
using AlignedF64 = toast::AlignedVector <double>;

PYBIND11_MAKE_OPAQUE(AlignedI8);
PYBIND11_MAKE_OPAQUE(AlignedU8);
PYBIND11_MAKE_OPAQUE(AlignedI16);
PYBIND11_MAKE_OPAQUE(AlignedU16);
PYBIND11_MAKE_OPAQUE(AlignedI32);
PYBIND11_MAKE_OPAQUE(AlignedU32);
PYBIND11_MAKE_OPAQUE(AlignedI64);
PYBIND11_MAKE_OPAQUE(AlignedU64);
PYBIND11_MAKE_OPAQUE(AlignedF32);
PYBIND11_MAKE_OPAQUE(AlignedF64);

template <typename T>
std::string align_format() {
    return std::string("void");
}

template <>
std::string align_format <int8_t> ();

template <>
std::string align_format <int16_t> ();

template <>
std::string align_format <int32_t> ();

template <>
std::string align_format <int64_t> ();

template <>
std::string align_format <uint8_t> ();

template <>
std::string align_format <uint16_t> ();

template <>
std::string align_format <uint32_t> ();

template <>
std::string align_format <uint64_t> ();

template <>
std::string align_format <float> ();

template <>
std::string align_format <double> ();


template <typename T>
void pybuffer_check_1D(py::buffer data) {
    auto log = toast::Logger::get();
    py::buffer_info info = data.request();
    std::string tp = align_format <T> ();
    if (info.format != tp) {
        std::ostringstream o;
        o << "Python buffer is type '" << info.format
          << "', not type '" << tp << "'";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    if (info.ndim != 1) {
        std::ostringstream o;
        o << "Python buffer is not one-dimensional";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return;
}

template <typename C>
std::unique_ptr <C> aligned_uptr(size_t n) {
    return std::unique_ptr <C> (new C(n));
}

template <typename C>
void register_aligned(py::module & m, char const * name) {
    py::class_ <C> (m, name, py::buffer_protocol())
    .def(py::init <>())
    .def(py::init <typename C::size_type>())
    .def_static("zeros", [](typename C::size_type nelem) {
                    std::unique_ptr <C> ret(new C(nelem));
                    std::fill(ret->begin(), ret->end(), 0);
                    return ret;
                })
    .def_static("ones", [](typename C::size_type nelem) {
                    std::unique_ptr <C> ret(new C(nelem));
                    std::fill(ret->begin(), ret->end(), 1);
                    return ret;
                })
    .def("pop_back", &C::pop_back)
    .def("push_back", (void (C::*)(
                           const typename C::value_type &)) & C::push_back)
    .def("resize", (void (C::*)(typename C::size_type count)) & C::resize)
    .def("size", &C::size)
    .def("clear", &C::clear)
    .def_buffer(
        [](C & self) -> py::buffer_info {
            std::string format = align_format <typename C::value_type> ();
            return py::buffer_info(
                static_cast <void *> (self.data()),
                sizeof(typename C::value_type),
                format,
                1,
                {self.size()},
                {sizeof(typename C::value_type)}
                );
        })
    .def("__len__", [](const C & self) {
             return self.size();
         })
    .def("__iter__", [](C & self) {
             return py::make_iterator(self.begin(), self.end());
         }, py::keep_alive <0, 1>())
    .def("__setitem__",
         [](C & self, typename C::size_type i,
            const typename C::value_type & t) {
             if (i >= self.size()) {
                 throw py::index_error();
             }
             self[i] = t;
         })
    .def("__getitem__",
         [](C & self, typename C::size_type i) -> typename C::value_type & {
             if (i >= self.size()) {
                 throw py::index_error();
             }
             return self[i];
         })
    .def("__setitem__",
         [](C & self, py::slice slice, py::buffer other) {
             size_t start, stop, step, slicelength;
             if (!slice.compute(self.size(), &start, &stop, &step,
                                &slicelength)) {
                 throw py::error_already_set();
             }
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);

             if (slicelength != info.size) {
                 throw std::runtime_error(
                     "Left and right hand size of slice assignment have different sizes!");
             }

             for (size_t i = 0; i < slicelength; ++i) {
                 self[start] = raw[i];
                 start += step;
             }
         })
    .def("__getitem__",
         [](C & self, py::slice slice) {
             size_t start, stop, step, slicelength;
             if (!slice.compute(self.size(), &start, &stop, &step,
                                &slicelength)) {
                 throw py::error_already_set();
             }
             std::unique_ptr <C> ret(new C(slicelength));
             for (size_t i = 0; i < slicelength; ++i) {
                 (*ret)[i] = self[start];
                 start += step;
             }
             return ret;
         })
    .def("__lt__", [](const C & self, typename C::value_type val) {
             py::array_t <bool> ret(self.size());
             auto result = ret.mutable_data();
             for (size_t i = 0; i < self.size(); ++i) {
                 result[i] = (self[i] < val);
             }
             return ret;
         })
    .def("__le__", [](const C & self, typename C::value_type val) {
             py::array_t <bool> ret(self.size());
             auto result = ret.mutable_data();
             for (size_t i = 0; i < self.size(); ++i) {
                 result[i] = (self[i] <= val);
             }
             return ret;
         })
    .def("__gt__", [](const C & self, typename C::value_type val) {
             py::array_t <bool> ret(self.size());
             auto result = ret.mutable_data();
             for (size_t i = 0; i < self.size(); ++i) {
                 result[i] = (self[i] > val);
             }
             return ret;
         })
    .def("__ge__", [](const C & self, typename C::value_type val) {
             py::array_t <bool> ret(self.size());
             auto result = ret.mutable_data();
             for (size_t i = 0; i < self.size(); ++i) {
                 result[i] = (self[i] >= val);
             }
             return ret;
         })
    .def("__eq__", [](const C & self, typename C::value_type val) {
             py::array_t <bool> ret(self.size());
             auto result = ret.mutable_data();
             for (size_t i = 0; i < self.size(); ++i) {
                 result[i] = (self[i] == val);
             }
             return ret;
         })
    .def("__ne__", [](const C & self, typename C::value_type val) {
             py::array_t <bool> ret(self.size());
             auto result = ret.mutable_data();
             for (size_t i = 0; i < self.size(); ++i) {
                 result[i] = (self[i] != val);
             }
             return ret;
         })
    .def("__repr__",
         [name](C const & self) {
             size_t npre = 1;
             if (self.size() > 2) {
                 npre = 2;
             }
             size_t npost = 0;
             if (self.size() > 1) {
                 npost = 1;
             }
             if (self.size() > 3) {
                 npost = 2;
             }
             std::string dots = "";
             if (self.size() > 4) {
                 dots = " ...";
             }
             std::ostringstream o;
             o << "<" << name << " " << self.size() << " elements:";
             for (size_t i = 0; i < npre; ++i) {
                 o << " " << self[i];
             }
             o << dots;
             for (size_t i = 0; i < npost; ++i) {
                 o << " " << self[self.size() - npost + i];
             }
             o << ">";
             return o.str();
         });

    return;
}

// Helper functions to check numpy array data types and dimensions.
void pybuffer_check_double_1D(py::buffer data);
void pybuffer_check_uint64_1D(py::buffer data);

// Initialize all the pieces of the bindings.
void init_sys(py::module & m);
void init_math_sf(py::module & m);
void init_math_rng(py::module & m);
void init_math_qarray(py::module & m);
void init_math_healpix(py::module & m);
void init_math_fft(py::module & m);
