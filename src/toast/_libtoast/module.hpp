
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef LIBTOAST_HPP
#define LIBTOAST_HPP

#include <common.hpp>

PYBIND11_MAKE_OPAQUE(toast::AlignedI8);
PYBIND11_MAKE_OPAQUE(toast::AlignedU8);
PYBIND11_MAKE_OPAQUE(toast::AlignedI16);
PYBIND11_MAKE_OPAQUE(toast::AlignedU16);
PYBIND11_MAKE_OPAQUE(toast::AlignedI32);
PYBIND11_MAKE_OPAQUE(toast::AlignedU32);
PYBIND11_MAKE_OPAQUE(toast::AlignedI64);
PYBIND11_MAKE_OPAQUE(toast::AlignedU64);
PYBIND11_MAKE_OPAQUE(toast::AlignedF32);
PYBIND11_MAKE_OPAQUE(toast::AlignedF64);


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
    .def("clear", [](C & self) {
             C().swap(self);
             return;
         })
    .def("address", [](C & self) {
             return (int64_t)((void *)self.data());
         })
    .def("array", [](C & self) -> py::array_t <typename C::value_type> {
             py::array_t <typename C::value_type> ret({self.size()},
                                                      {sizeof(typename C::value_type)},
                                                      self.data(), py::cast(self));
             return ret;
         })
    .def_buffer(
        [](C & self) -> py::buffer_info {
            std::string format =
                py::format_descriptor <typename C::value_type>::format();
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

    // Set and get individual elements
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

    // Set and get a slice
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
    .def("__setitem__",
         [](C & self, py::slice slice, const typename C::value_type & t) {
             size_t start, stop, step, slicelength;
             if (!slice.compute(self.size(), &start, &stop, &step,
                                &slicelength)) {
                 throw py::error_already_set();
             }
             for (size_t i = 0; i < slicelength; ++i) {
                 self[start] = t;
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

    // Set and get explicit indices
    .def("__setitem__",
         [](C & self, py::array_t <int64_t> indices, py::buffer other) {
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);

             if (indices.size() != info.size) {
                 throw std::runtime_error(
                           "Left and right hand indexed assignment have different sizes!");
             }

             auto * dat = indices.data();

             for (size_t i = 0; i < info.size; ++i) {
                 self[dat[i]] = raw[i];
             }
         })
    .def("__setitem__",
         [](C & self, py::array_t <int64_t> indices, const typename C::value_type & t) {
             auto * dat = indices.data();
             for (size_t i = 0; i < indices.size(); ++i) {
                 self[dat[i]] = t;
             }
         })
    .def("__getitem__",
         [](C & self, py::array_t <int64_t> indices) {
             auto * dat = indices.data();
             std::unique_ptr <C> ret(new C(indices.size()));
             for (size_t i = 0; i < indices.size(); ++i) {
                 (*ret)[i] = self[dat[i]];
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

    // Arithmetic
    .def("__iadd__",
         [](C & self, py::buffer other) {
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);

             if (self.size() != info.size) {
                 throw std::runtime_error(
                           "Object and operand have different sizes!");
             }

             for (size_t i = 0; i < info.size; ++i) {
                 self[i] += raw[i];
             }
         })
    .def("__iadd__",
         [](C & self, typename C::value_type val) {
             for (size_t i = 0; i < self.size(); ++i) {
                 self[i] += val;
             }
         })
    .def("__isub__",
         [](C & self, py::buffer other) {
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);

             if (self.size() != info.size) {
                 throw std::runtime_error(
                           "Object and operand have different sizes!");
             }

             for (size_t i = 0; i < info.size; ++i) {
                 self[i] -= raw[i];
             }
         })
    .def("__isub__",
         [](C & self, typename C::value_type val) {
             for (size_t i = 0; i < self.size(); ++i) {
                 self[i] -= val;
             }
         })
    .def("__imul__",
         [](C & self, py::buffer other) {
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);

             if (self.size() != info.size) {
                 throw std::runtime_error(
                           "Object and operand have different sizes!");
             }

             for (size_t i = 0; i < info.size; ++i) {
                 self[i] *= raw[i];
             }
         })
    .def("__imul__",
         [](C & self, typename C::value_type val) {
             for (size_t i = 0; i < self.size(); ++i) {
                 self[i] *= val;
             }
         })
    .def("__add__",
         [](C & self, py::buffer other) {
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);
             if (self.size() != info.size) {
                 throw std::runtime_error(
                           "Object and operand have different sizes!");
             }
             std::unique_ptr <C> ret(new C(self));
             for (size_t i = 0; i < info.size; ++i) {
                 (*ret)[i] += raw[i];
             }
             return ret;
         })
    .def("__add__",
         [](C & self, typename C::value_type val) {
             std::unique_ptr <C> ret(new C(self));
             for (size_t i = 0; i < self.size(); ++i) {
                 (*ret)[i] += val;
             }
             return ret;
         })
    .def("__sub__",
         [](C & self, py::buffer other) {
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);
             if (self.size() != info.size) {
                 throw std::runtime_error(
                           "Object and operand have different sizes!");
             }
             std::unique_ptr <C> ret(new C(self));
             for (size_t i = 0; i < info.size; ++i) {
                 (*ret)[i] -= raw[i];
             }
             return ret;
         })
    .def("__sub__",
         [](C & self, typename C::value_type val) {
             std::unique_ptr <C> ret(new C(self));
             for (size_t i = 0; i < self.size(); ++i) {
                 (*ret)[i] -= val;
             }
             return ret;
         })
    .def("__mul__",
         [](C & self, py::buffer other) {
             pybuffer_check_1D <typename C::value_type> (other);
             py::buffer_info info = other.request();
             typename C::value_type * raw =
                 reinterpret_cast <typename C::value_type *> (info.ptr);
             if (self.size() != info.size) {
                 throw std::runtime_error(
                           "Object and operand have different sizes!");
             }
             std::unique_ptr <C> ret(new C(self));
             for (size_t i = 0; i < info.size; ++i) {
                 (*ret)[i] *= raw[i];
             }
             return ret;
         })
    .def("__mul__",
         [](C & self, typename C::value_type val) {
             std::unique_ptr <C> ret(new C(self));
             for (size_t i = 0; i < self.size(); ++i) {
                 (*ret)[i] *= val;
             }
             return ret;
         })

    // string representation
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

// Initialize all the pieces of the bindings.
void init_sys(py::module & m);
void init_intervals(py::module & m);
void init_math_misc(py::module & m);
void init_math_sf(py::module & m);
void init_math_rng(py::module & m);
void init_math_qarray(py::module & m);
void init_math_healpix(py::module & m);
void init_math_fft(py::module & m);
void init_fod_psd(py::module & m);
void init_tod_filter(py::module & m);
void init_tod_pointing(py::module & m);
void init_tod_simnoise(py::module & m);
void init_todmap_scanning(py::module & m);
void init_map_cov(py::module & m);
void init_pixels(py::module & m);
void init_todmap_mapmaker(py::module & m);
void init_atm(py::module & m);
void init_template_offset(py::module & m);
void init_accelerator(py::module & m);
void init_ops_pointing_detector(py::module & m);
void init_ops_stokes_weights(py::module & m);
void init_ops_pixels_healpix(py::module & m);
void init_ops_mapmaker_utils(py::module & m);

#endif // ifndef LIBTOAST_HPP
