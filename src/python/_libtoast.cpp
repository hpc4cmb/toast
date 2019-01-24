
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


PYBIND11_MODULE(_libtoast, m) {
    m.doc() = R"(
    Interface to C++ TOAST library.

    )";

    init_sys(m);
    init_math_sf(m);
}
