
// Copyright (c) 2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_math_fma(py::module & m) {

    m.def(
        "inplace_weighted_sum",
        [](py::buffer out, py::buffer weights, py::args a) {
            py::buffer_info info_out = out.request();
            py::buffer_info info_weights = weights.request();

            if ((unsigned) info_weights.size != a.size()) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "weights and args are of different sizes.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            if (info_out.format != py::format_descriptor<double>::format()) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Incompatible format: expecting out to be a double array!";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * out_raw = reinterpret_cast <double *> (info_out.ptr);

            if (info_weights.format != py::format_descriptor<double>::format()) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Incompatible format: expecting weights to be a double array!";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * weights_raw = reinterpret_cast <double *> (info_weights.ptr);

            double** arrays = new double*[a.size()];
            for (size_t i = 0; i < a.size(); ++i) {
                // Use raw Python API here to avoid an extra, intermediate incref on the tuple item:
                py::handle array = PyTuple_GET_ITEM(a.ptr(), static_cast<py::ssize_t>(i));
                py::buffer array_buffer = array.cast<py::buffer>();
                py::buffer_info info_array = array_buffer.request();

                if (info_array.size != info_out.size) {
                    auto log = toast::Logger::get();
                    std::ostringstream o;
                    o << "An array from args does not have the same size with out.";
                    log.error(o.str().c_str());
                    throw std::runtime_error(o.str().c_str());
                }

                if (info_array.format != py::format_descriptor<double>::format()) {
                    auto log = toast::Logger::get();
                    std::ostringstream o;
                    o << "Incompatible format: expecting all array from args to be a double array!";
                    log.error(o.str().c_str());
                    throw std::runtime_error(o.str().c_str());
                }
                arrays[i] = reinterpret_cast <double *> (info_array.ptr);
            }

            toast::inplace_weighted_sum(info_out.size, info_weights.size, out_raw, weights_raw, arrays);
            delete [] arrays;
        },
        py::arg("out"),
        py::arg("weights"),
        R"(
            Compute a weighted sum of float64 values.

            The results are stored in the output buffer. To guarantee SIMD
            vectorization, the input (args) and output arrays (out) should be aligned
            (i.e. use an AlignedF64).

            Args:
                out (array_like):  1D array of float64 values.
                weights (array_like):  1D array of float64 values.
                *args (array_like):  1D array of float64 values.

            Returns:
                None

            Effectively equivalent to:

            >>> for weight, array in zip(weights, args):
            >>>     out += weight * array
        )");

    return;
}
