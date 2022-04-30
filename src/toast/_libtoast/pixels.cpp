
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


template <typename T>
py::tuple global_to_local(
    py::array_t <T> global_pixels,
    size_t npix_submap,
    py::array_t <int64_t> global2local
) {
    // Get raw pointers to input.
    py::buffer_info gpinfo = global_pixels.request();
    py::buffer_info glinfo = global2local.request();
    T * global_pixels_raw = reinterpret_cast <T *> (gpinfo.ptr);
    int64_t * global_to_local_raw = reinterpret_cast <int64_t *> (glinfo.ptr);

    size_t nsamp = gpinfo.size;

    // Allocate output arrays.
    auto local_submaps = py::array_t <T> ();
    auto local_pixels = py::array_t <T> ();
    local_submaps.resize({nsamp});
    local_pixels.resize({nsamp});

    // Get raw pointers to outputs
    py::buffer_info lsinfo = local_submaps.request();
    py::buffer_info lpinfo = local_pixels.request();
    T * local_submaps_raw = reinterpret_cast <T *> (lsinfo.ptr);
    T * local_pixels_raw = reinterpret_cast <T *> (lpinfo.ptr);

    // Call internal function
    toast::global_to_local <T> (
        nsamp, global_pixels_raw, npix_submap,
        global_to_local_raw, local_submaps_raw, local_pixels_raw
    );
    return py::make_tuple(local_submaps, local_pixels);
}

void init_pixels(py::module & m) {
    m.doc() = "Compiled kernels to support TOAST pixels classes";

    // Register versions of the function for likely integer types we want to
    // support.  We do not support unsigned types since we use the "-1" value
    // to mean invalid pixels.
    m.def("global_to_local", &global_to_local <int64_t>, py::arg("global_pixels"),
          py::arg("npix_submap"), py::arg(
              "global2local"), R"(
        Convert global pixel indices to local submaps and pixels within the submap.

        Args:
            global_pixels (array):  The global pixel indices.
            npix_submap (int):  The number of pixels in each submap.
            global2local (array, int64):  The local submap for each global submap.

        Returns:
            (tuple):  The (local submap, pixel within submap) for each global pixel.

    )");
    m.def("global_to_local", &global_to_local <int32_t>);
    m.def("global_to_local", &global_to_local <int16_t>);
}
