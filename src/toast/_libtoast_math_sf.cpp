
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_math_sf(py::module & m) {
    // vector math functions

    m.def(
        "vsin", [](py::buffer in) {
            pybuffer_check_double_1D(in);
            auto out = AlignedArray::zeros_like(in);
            py::buffer_info info = in.request();
            double * inraw = reinterpret_cast <double *> (info.ptr);
            double * outraw = reinterpret_cast <double *> (out->data.data());
            toast::vsin(info.size, inraw,
                        outraw);
            return out;
        }, py::arg(
            "in"), R"(
        Compute the Sine for an array of float64 values.

        A new aligned-memory array is created for the result and returned.
        To guarantee SIMD vectorization, the input array should be aligned
        (i.e. use an AlignedArray).

        Args:
            in (array-like):  1D array of float64 values.

        Returns:
            (AlignedArray):  a new array with the result.

    )");

    m.def(
        "vsin", [](py::buffer in, py::buffer out) {
            pybuffer_check_double_1D(in);
            pybuffer_check_double_1D(out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vsin(info_in.size, inraw,
                        outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the Sine for an array of float64 values.

        The results are stored in the provided output buffer.  To
        guarantee SIMD vectorization, the input and output arrays should be
        aligned (i.e. use an AlignedArray).

        Args:
            in (array-like):  1D array of float64 values.

            out (array-like):  1D array of float64 values.

        Returns:
            None
    )");

    //
    // void vsin(int n, double const * ang, double * sinout);
    // void vcos(int n, double const * ang, double * cosout);
    // void vsincos(int n, double const * ang, double * sinout, double *
    // cosout);
    // void vatan2(int n, double const * y, double const * x, double * ang);
    // void vsqrt(int n, double const * in, double * out);
    // void vrsqrt(int n, double const * in, double * out);
    // void vexp(int n, double const * in, double * out);
    // void vlog(int n, double const * in, double * out);
    //
    // void vfast_sin(int n, double const * ang, double * sinout);
    // void vfast_cos(int n, double const * ang, double * cosout);
    // void vfast_sincos(int n, double const * ang, double * sinout,
    //                   double * cosout);
    // void vfast_atan2(int n, double const * y, double const * x, double *
    // ang);
    // void vfast_sqrt(int n, double const * in, double * out);
    // void vfast_rsqrt(int n, double const * in, double * out);
    // void vfast_exp(int n, double const * in, double * out);
    // void vfast_log(int n, double const * in, double * out);
    // void vfast_erfinv(int n, double const * in, double * out);
}
