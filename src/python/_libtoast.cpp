
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>

using size_container = py::detail::any_container <ssize_t>;

// This helper class wraps an aligned memory buffer of bytes which can
// represent any dtype.

AlignedArray::AlignedArray(std::vector <py::ssize_t> const & shp,
                           py::dtype dt) {
    init(shp, dt);
}

AlignedArray::AlignedArray(py::buffer input) {
    py::buffer_info info = input.request();
    py::dtype dt(info.format);
    init(info.shape, dt);
    std::memcpy(data.data(), info.ptr,
                flatsize * itemsize);
}

AlignedArray::~AlignedArray() {}

std::unique_ptr <AlignedArray>
AlignedArray::create(std::vector <py::ssize_t> const & shp, py::dtype dt) {
    std::unique_ptr <AlignedArray> ret(new AlignedArray(shp, dt));
    std::fill(ret->data.begin(), ret->data.end(), 0);
    return ret;
}

std::unique_ptr <AlignedArray>
AlignedArray::zeros_like(py::buffer other) {
    std::unique_ptr <AlignedArray> ret(new AlignedArray(other));
    std::fill(ret->data.begin(), ret->data.end(), 0);
    return ret;
}

std::unique_ptr <AlignedArray>
AlignedArray::empty_like(py::buffer other) {
    std::unique_ptr <AlignedArray> ret(new AlignedArray(other));
    return ret;
}

void AlignedArray::init(std::vector <py::ssize_t> const & shp, py::dtype dt) {
    shape.clear();
    shape.resize(shp.size());
    std::copy(shp.begin(), shp.end(), shape.begin());
    dtype = dt;
    itemsize = dtype.itemsize();
    flatsize = 1;
    for (auto const & s : shape) {
        flatsize *= s;
    }
    data.resize(itemsize * flatsize);
    return;
}

// Helper functions to check numpy array data types and dimensions.

void pybuffer_check_double_1D(py::buffer data) {
    auto log = toast::Logger::get();
    py::buffer_info info = data.request();
    if (info.format != "d") {
        std::ostringstream o;
        o << "Python buffer is not float64 type";
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

void pybuffer_check_uint64_1D(py::buffer data) {
    auto log = toast::Logger::get();
    py::buffer_info info = data.request();
    if (info.format != "L") {
        std::ostringstream o;
        o << "Python buffer is not uint64 type";
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

PYBIND11_MODULE(_libtoast, m) {
    m.doc() = R"(
    Interface to C++ TOAST library.

    )";

    // Define a wrapper around our internal aligned memory vector class.
    // Expose the memory with the python / numpy buffer protocol.

    py::class_ <AlignedArray, std::unique_ptr <AlignedArray> > (
        m, "AlignedArray", py::buffer_protocol(), py::dynamic_attr())
    .def(py::init <std::vector <py::ssize_t> const &, py::dtype> ())
    .def(py::init <py::buffer> ())
    .def_readonly("dtype", &AlignedArray::dtype)
    .def_readonly("shape", &AlignedArray::shape)
    .def("empty_like", &AlignedArray::empty_like)
    .def("zeros_like", &AlignedArray::zeros_like)
    .def_buffer(
        [](AlignedArray & self) -> py::buffer_info {
            std::vector <py::ssize_t> bstrides;
            py::ssize_t strd = 1;
            for (auto const & s : self.shape) {
                bstrides.push_back(strd);
                strd *= s;
            }
            std::reverse(bstrides.begin(), bstrides.end());
            py::buffer_info binfo(
                static_cast <void *> (self.data.data()),
                self.itemsize,
                std::string(1, self.dtype.kind()),
                py::ssize_t(self.shape.size()),
                size_container(self.shape),
                size_container(bstrides)
                );
            return binfo;
        })
    .def("__repr__",
         [](AlignedArray const & self) {
             std::ostringstream sh;
             sh << "(";
             for (auto const & s : self.shape) {
                 sh << s << ",";
             }
             sh << ")";
             std::ostringstream o;
             o << "<AlignedArray type=" << self.dtype.kind()
               << " itemsize=" << self.dtype.itemsize()
               << " shape=" << sh.str()
               << " " << self.flatsize << " total elements>";
             return o.str();
         }
         );

    init_sys(m);
    init_math_sf(m);
    init_math_rng(m);
}
