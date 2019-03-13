
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
    dtype = dt;
    itemsize = dtype.itemsize();

    // Now compute the buffer format string from the dtype kind and itemsize.
    char ckind = dtype.kind();
    std::string kind(1, ckind);
    if (kind.compare("b") == 0) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "dtype kind \"" << kind << "\" (bool) is not supported";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else if (kind.compare("i") == 0) {
        switch (itemsize) {
            case 1:
                format = "b";
                break;

            case 2:
                format = "h";
                break;

            case 4:
                format = "i";
                break;

            case 8:
                format = "q";
                break;

            default:
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "dtype kind \"" << kind
                  << "\" does not support an item size of " << itemsize;
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
                break;
        }
    } else if (kind.compare("u") == 0) {
        switch (itemsize) {
            case 1:
                format = "B";
                break;

            case 2:
                format = "H";
                break;

            case 4:
                format = "I";
                break;

            case 8:
                format = "Q";
                break;

            default:
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "dtype kind \"" << kind
                  << "\" does not support an item size of " << itemsize;
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
                break;
        }
    } else if (kind.compare("f") == 0) {
        switch (itemsize) {
            case 2:
                format = "e";
                break;

            case 4:
                format = "f";
                break;

            case 8:
                format = "d";
                break;

            default:
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "dtype kind \"" << kind
                  << "\" does not support an item size of " << itemsize;
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
                break;
        }
    } else if (kind.compare("c") == 0) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "dtype kind \"" << kind << "\" (complex float) is not supported";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else if (kind.compare("m") == 0) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "dtype kind \"" << kind << "\" (time delta) is not supported";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else if (kind.compare("M") == 0) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "dtype kind \"" << kind << "\" (datetime) is not supported";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else if (kind.compare("O") == 0) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "dtype kind \"" << kind << "\" (Object) is not supported";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else if (kind.compare("S") == 0) {
        format = "s";
    } else if (kind.compare("U") == 0) {
        switch (itemsize) {
            case 1:
                format = "s";
                break;

            case 2:
                format = "H";
                break;

            case 4:
                format = "I";
                break;

            default:
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "dtype kind \"" << kind
                  << "\" (Unicode) does not support an item size of "
                  << itemsize;
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
                break;
        }
    } else if (kind.compare("V") == 0) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "dtype kind \"" << kind << "\" (Void) is not supported";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "Unknown dtype kind \"" << kind << "\"";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    // The shape
    shape.clear();
    shape.resize(shp.size());
    std::copy(shp.begin(), shp.end(), shape.begin());

    // Compute the flatpacked size and strides
    flatsize = 1;
    strides.clear();
    py::ssize_t strd = 1;
    for (auto const & s : shape) {
        flatsize *= s;
        strides.push_back(strd);
        strd *= s;
    }
    std::reverse(strides.begin(), strides.end());
    for (auto & b : strides) {
        b *= itemsize;
    }

    // Allocate the data.
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

    // py::options options;
    // options.disable_function_signatures();

    // Define a wrapper around our internal aligned memory vector class.
    // Expose the memory with the python / numpy buffer protocol.

    py::class_ <AlignedArray, std::unique_ptr <AlignedArray> > (
        m, "AlignedArray", py::buffer_protocol(), py::dynamic_attr())
    .def(py::init <std::vector <py::ssize_t> const &, py::dtype> ())
    .def(py::init <py::buffer> ())
    .def_readonly("dtype", &AlignedArray::dtype)
    .def_readonly("format", &AlignedArray::format)
    .def_readonly("shape", &AlignedArray::shape)
    .def_readonly("strides", &AlignedArray::strides)
    .def_readonly("flatsize", &AlignedArray::flatsize)
    .def_readonly("itemsize", &AlignedArray::itemsize)
    .def("empty_like", &AlignedArray::empty_like)
    .def("zeros_like", &AlignedArray::zeros_like)
    .def_buffer(
        [](AlignedArray & self) -> py::buffer_info {
            py::buffer_info binfo(
                static_cast <void *> (self.data.data()),
                self.itemsize,
                self.format,
                py::ssize_t(self.shape.size()),
                size_container(self.shape),
                size_container(self.strides)
                );
            return binfo;
        })
    .def("__getitem__",
         [](AlignedArray & self, py::slice slice) ->
         std::unique_ptr <AlignedArray> {
             size_t datasize = (size_t)(self.data.size() / self.itemsize);
             size_t start, stop, step, slicelength;
             if (!slice.compute(datasize, &start, &stop, &step,
                                &slicelength)) {
                 throw error_already_set();
             }

             Vector * seq = new Vector();
             seq->reserve((size_t)slicelength);

             for (size_t i = 0; i < slicelength; ++i) {
                 seq->push_back(v[start]);
                 start += step;
             }
             return seq;
         },
         arg("s"),
         "Retrieve list elements using a slice object"
         );
    .def("__getitem__",
         [](AlignedArray & self, py::slice slice) ->
         std::unique_ptr <AlignedArray> {
             size_t datasize = (size_t)(self.data.size() / self.itemsize);
             size_t start, stop, step, slicelength;
             if (!slice.compute(datasize, &start, &stop, &step,
                                &slicelength)) {
                 throw error_already_set();
             }

             Vector * seq = new Vector();
             seq->reserve((size_t)slicelength);

             for (size_t i = 0; i < slicelength; ++i) {
                 seq->push_back(v[start]);
                 start += step;
             }
             return seq;
         },
         arg("s"),
         "Retrieve list elements using a slice object"
         );
    cl.def("__setitem__",
           [](Vector & v, SizeType i, const T & t) {
               if (i >= v.size()) throw index_error();
               v[i] = t;
           }
           );
    cl.def("__setitem__",
           [](Vector & v, slice slice,  const Vector & value) {
               size_t start, stop, step, slicelength;
               if (!slice.compute(
                       v.size(), &start, &stop, &step,
                       &slicelength)) throw error_already_set();

               if (slicelength !=
                   value.size()) throw std::runtime_error(
                       "Left and right hand size of slice assignment have
    different sizes!");

               for (size_t i = 0; i < slicelength; ++i) {
                   v[start] = value[i];
                   start += step;
               }
           },
           "Assign list elements using a slice object"
           )

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
         });


    init_sys(m);
    init_math_sf(m);
    init_math_rng(m);
}
