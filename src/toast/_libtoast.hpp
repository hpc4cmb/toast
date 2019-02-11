
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <pybind11/stl_bind.h>

#include <toast.hpp>

#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace py = pybind11;


// This helper class wraps an aligned memory buffer of bytes which can
// represent any dtype.

class AlignedArray {
    public:

        AlignedArray(std::vector <py::ssize_t> const & shp, py::dtype dt);

        AlignedArray(py::buffer input);

        ~AlignedArray();

        static std::unique_ptr <AlignedArray> create(
            std::vector <py::ssize_t> const & shp, py::dtype dt);

        static std::unique_ptr <AlignedArray> zeros_like(py::buffer other);

        static std::unique_ptr <AlignedArray> empty_like(py::buffer other);

        py::dtype dtype;
        toast::AlignedVector <uint8_t> data;
        std::vector <py::ssize_t> shape;
        py::ssize_t flatsize;
        py::ssize_t itemsize;

    private:

        void init(std::vector <py::ssize_t> const & shp, py::dtype dt);
};


// Helper functions to check numpy array data types and dimensions.
void pybuffer_check_double_1D(py::buffer data);
void pybuffer_check_uint64_1D(py::buffer data);

// Initialize all the pieces of the bindings.
void init_sys(py::module & m);
void init_math_sf(py::module & m);
void init_math_rng(py::module & m);
