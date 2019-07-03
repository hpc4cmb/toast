
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>

#include <toast_mpi_internal.hpp>


using size_container = py::detail::any_container <ssize_t>;


// Currently the only compiled code that uses MPI and needs to be bound to python is
// the atmosphere simulation code.  If the number of things increases, we should split
// this file into multiple files.

void init_mpi_atm(py::module & m) {
    return;
}

PYBIND11_MODULE(_libtoast_mpi, m) {
    m.doc() = R"(
    Interface to C++ TOAST MPI library.

    )";

    init_mpi_atm(m);
}
