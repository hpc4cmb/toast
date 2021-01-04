
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_template_offset(py::module & m) {
    m.def(
        "template_offset_add_to_signal", [](int64_t step_length, py::buffer amplitudes,
                                            py::buffer data) {
            pybuffer_check_1D <double> (amplitudes);
            pybuffer_check_1D <double> (data);
            py::buffer_info info_amplitudes = amplitudes.request();
            py::buffer_info info_data = data.request();
            int64_t n_amp = info_amplitudes.size;
            int64_t n_data = info_data.size;
            double * raw_amplitudes = reinterpret_cast <double *> (info_amplitudes.ptr);
            double * raw_data = reinterpret_cast <double *> (info_data.ptr);
            toast::template_offset_add_to_signal(step_length, n_amp, raw_amplitudes,
                                                 n_data, raw_data);
            return;
        }, py::arg("step_length"), py::arg("amplitudes"), py::arg(
            "data"), R"(
        Accumulate offset amplitudes to timestream data.

        Each amplitude value is accumulated to `step_length` number of samples.  The
        final offset will be at least this many samples, but may be more if the step
        size does not evenly divide into the number of samples.

        Args:
            step_length (int64):  The minimum number of samples for each offset.
            amplitudes (array):  The float64 amplitude values.
            data (array):  The float64 timestream values to accumulate.

        Returns:
            None.

    )");

    m.def(
        "template_offset_project_signal", [](int64_t step_length, py::buffer data,
                                             py::buffer amplitudes) {
            pybuffer_check_1D <double> (amplitudes);
            pybuffer_check_1D <double> (data);
            py::buffer_info info_amplitudes = amplitudes.request();
            py::buffer_info info_data = data.request();
            int64_t n_amp = info_amplitudes.size;
            int64_t n_data = info_data.size;
            double * raw_amplitudes = reinterpret_cast <double *> (info_amplitudes.ptr);
            double * raw_data = reinterpret_cast <double *> (info_data.ptr);
            toast::template_offset_add_to_signal(step_length, n_data, raw_data,
                                                 n_amp, raw_amplitudes);
            return;
        }, py::arg("step_length"), py::arg("data"), py::arg(
            "amplitudes"), R"(
        Accumulate timestream data into offset amplitudes.

        Chunks of `step_length` number of samples are accumulated into the offset
        amplitudes.  If step_length does not evenly divide into the total number of
        samples, the final amplitude will be extended to include the remainder.

        Args:
            step_length (int64):  The minimum number of samples for each offset.
            data (array):  The float64 timestream values.
            amplitudes (array):  The float64 amplitude values.

        Returns:
            None.

    )");

    return;
}
