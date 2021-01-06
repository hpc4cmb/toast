
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_math_sf(py::module & m) {
    // vector math functions

    m.def(
        "vsin", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vsin(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the Sine for an array of float64 values.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vcos", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vcos(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the Cosine for an array of float64 values.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vsincos", [](py::buffer in, py::buffer sinout, py::buffer cosout) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (sinout);
            pybuffer_check_1D <double> (cosout);
            py::buffer_info info_in = in.request();
            py::buffer_info info_sinout = sinout.request();
            py::buffer_info info_cosout = cosout.request();
            if ((info_in.size != info_sinout.size) ||
                (info_in.size != info_cosout.size)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * sinoutraw = reinterpret_cast <double *> (info_sinout.ptr);
            double * cosoutraw = reinterpret_cast <double *> (info_cosout.ptr);
            toast::vsincos(info_in.size, inraw, sinoutraw, cosoutraw);
            return;
        }, py::arg("in"), py::arg("sinout"), py::arg(
            "cosout"), R"(
        Compute the sine and cosine for an array of float64 values.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            sinout (array_like):  1D array of float64 values.
            cosout (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vatan2", [](py::buffer y, py::buffer x, py::buffer ang) {
            pybuffer_check_1D <double> (y);
            pybuffer_check_1D <double> (x);
            pybuffer_check_1D <double> (ang);
            py::buffer_info info_x = x.request();
            py::buffer_info info_y = y.request();
            py::buffer_info info_ang = ang.request();
            if ((info_x.size != info_y.size) ||
                (info_x.size != info_ang.size)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * xraw = reinterpret_cast <double *> (info_x.ptr);
            double * yraw = reinterpret_cast <double *> (info_y.ptr);
            double * angraw = reinterpret_cast <double *> (info_ang.ptr);
            toast::vatan2(info_x.size, yraw, xraw, angraw);
            return;
        }, py::arg("y"), py::arg("x"), py::arg(
            "ang"), R"(
        Compute the arctangent of the y and x values.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            y (array_like):  1D array of float64 values.
            x (array_like):  1D array of float64 values.
            ang (array_like):  output angles as 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vsqrt", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vsqrt(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the sqrt an array of float64 values.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vrsqrt", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vrsqrt(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the inverse sqrt an array of float64 values.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vexp", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vexp(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute e^x for an array of float64 values.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vlog", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vlog(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the natural log of an array of float64 values.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    // Now the "fast" / less accurate versions.

    m.def(
        "vfast_sin", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vfast_sin(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the Sine for an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_cos", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vfast_cos(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the Cosine for an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_sincos", [](py::buffer in, py::buffer sinout,
                           py::buffer cosout) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (sinout);
            pybuffer_check_1D <double> (cosout);
            py::buffer_info info_in = in.request();
            py::buffer_info info_sinout = sinout.request();
            py::buffer_info info_cosout = cosout.request();
            if ((info_in.size != info_sinout.size) ||
                (info_in.size != info_cosout.size)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * sinoutraw = reinterpret_cast <double *> (info_sinout.ptr);
            double * cosoutraw = reinterpret_cast <double *> (info_cosout.ptr);
            toast::vfast_sincos(info_in.size, inraw, sinoutraw, cosoutraw);
            return;
        }, py::arg("in"), py::arg("sinout"), py::arg(
            "cosout"), R"(
        Compute the sine and cosine for an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            sinout (array_like):  1D array of float64 values.
            cosout (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_atan2", [](py::buffer y, py::buffer x, py::buffer ang) {
            pybuffer_check_1D <double> (y);
            pybuffer_check_1D <double> (x);
            pybuffer_check_1D <double> (ang);
            py::buffer_info info_x = x.request();
            py::buffer_info info_y = y.request();
            py::buffer_info info_ang = ang.request();
            if ((info_x.size != info_y.size) ||
                (info_x.size != info_ang.size)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * xraw = reinterpret_cast <double *> (info_x.ptr);
            double * yraw = reinterpret_cast <double *> (info_y.ptr);
            double * angraw = reinterpret_cast <double *> (info_ang.ptr);
            toast::vfast_atan2(info_x.size, yraw, xraw, angraw);
            return;
        }, py::arg("y"), py::arg("x"), py::arg(
            "ang"), R"(
        Compute the arctangent of the y and x values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            y (array_like):  1D array of float64 values.
            x (array_like):  1D array of float64 values.
            ang (array_like):  output angles as 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_sqrt", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vfast_sqrt(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the sqrt an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_rsqrt", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vfast_rsqrt(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the inverse sqrt an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_exp", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vfast_exp(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute e^x for an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_log", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vfast_log(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the natural log of an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "vfast_erfinv", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            if (info_in.size != info_out.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Input and output buffers are different sizes";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::vfast_erfinv(info_in.size, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the inverse error function for an array of float64 values.

        "Fast" version:  this function may run much faster than the
        standard version at the expense of errors in the least significant
        bits.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  1D array of float64 values.
            out (array_like):  1D array of float64 values.

        Returns:
            None

    )");

    return;
}
