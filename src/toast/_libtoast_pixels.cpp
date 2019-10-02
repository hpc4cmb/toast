
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>

void global_to_local(py::array_t <long> global_pixels,
                     long npix_submap,
                     py::array_t <long> global2local,
                     py::array_t <long> local_submaps,
                     py::array_t <long> local_pixels) {
    auto fast_global_pixels = global_pixels.unchecked <1>();
    auto fast_global2local = global2local.unchecked <1>();
    auto fast_submap = local_submaps.mutable_unchecked <1>();
    auto fast_local_pixels = local_pixels.mutable_unchecked <1>();

    double npix_submap_inv = 1. / npix_submap;
    for (size_t i = 0; i < global_pixels.size(); ++i) {
        long pixel = fast_global_pixels(i);
        long submap = 0;
        if (pixel < 0) {
            pixel = -1;
        } else {
            submap = pixel * npix_submap_inv;
            pixel -= submap * npix_submap;
        }
        fast_local_pixels(i) = pixel;
        submap = fast_global2local(submap);
        fast_submap(i) = submap;
    }
}

void init_pixels(py::module & m) {
    m.doc() = "Compiled kernels to support TOAST pixels classes";

    m.def("global_to_local", &global_to_local);
}
