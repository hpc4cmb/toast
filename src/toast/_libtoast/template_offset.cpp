
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <accelerator.hpp>

#include <intervals.hpp>


// FIXME:  docstrings need to be updated if we keep these versions of the code.

void init_template_offset(py::module & m) {
    m.def(
        "template_offset_add_to_signal", [](
            int64_t step_length,
            int64_t amp_offset,
            py::buffer n_amp_views,
            py::buffer amplitudes,
            int32_t data_index,
            py::buffer det_data,
            py::buffer intervals,
            bool use_accel
        ) {
            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            double * raw_amplitudes = extract_buffer <double> (
                amplitudes, "amplitudes", 1, temp_shape, {-1}
            );
            int64_t n_amp = temp_shape[0];

            double * raw_det_data = extract_buffer <double> (
                det_data, "det_data", 2, temp_shape, {-1, -1}
            );
            int64_t n_all_det = temp_shape[0];
            int64_t n_samp = temp_shape[1];

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            int64_t * raw_n_amp_views = extract_buffer <int64_t> (
                n_amp_views, "n_amp_views", 1, temp_shape, {n_view}
            );

            if (offload) {
                #ifdef HAVE_OPENMP_TARGET

                double * dev_det_data = omgr.device_ptr(raw_det_data);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                double * dev_amplitudes = omgr.device_ptr(raw_amplitudes);

                # pragma omp target data \
                device(dev)              \
                map(to:                  \
                n_view,                  \
                n_samp,                  \
                data_index,              \
                step_length,             \
                amp_offset,              \
                raw_n_amp_views          \
                )
                {
                    int64_t offset = amp_offset;
                    # pragma omp target teams distribute firstprivate(offset) \
                        is_device_ptr( \
                            dev_amplitudes, \
                            dev_det_data, \
                            dev_intervals \
                        )
                    for (int64_t iview = 0; iview < n_view; iview++) {
                        # pragma omp parallel for default(shared)
                        for (
                            int64_t isamp = dev_intervals[iview].first;
                            isamp <= dev_intervals[iview].last;
                            isamp++
                        ) {
                            int64_t d = data_index * n_samp + isamp;
                            int64_t amp = offset + (int64_t)(isamp / step_length);
                            dev_det_data[d] += dev_amplitudes[amp];
                        }
                        offset += raw_n_amp_views[iview];
                    }
                }

                #endif // ifdef HAVE_OPENMP_TARGET
            } else {
                int64_t offset = amp_offset;
                for (int64_t iview = 0; iview < n_view; iview++) {
                    #pragma omp parallel for default(shared)
                    for (
                        int64_t isamp = raw_intervals[iview].first;
                        isamp <= raw_intervals[iview].last;
                        isamp++
                    ) {
                        int64_t d = data_index * n_samp + isamp;
                        int64_t amp = offset + (int64_t)(isamp / step_length);
                        raw_det_data[d] += raw_amplitudes[amp];
                    }
                    offset += raw_n_amp_views[iview];
                }
            }
            return;
        });

    m.def(
        "template_offset_project_signal", [](
            int32_t data_index,
            py::buffer det_data,
            int32_t flag_index,
            py::buffer flag_data,
            uint8_t flag_mask,
            int64_t step_length,
            int64_t amp_offset,
            py::buffer n_amp_views,
            py::buffer amplitudes,
            py::buffer intervals,
            bool use_accel
        ) {
            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            double * raw_amplitudes = extract_buffer <double> (
                amplitudes, "amplitudes", 1, temp_shape, {-1}
            );
            int64_t n_amp = temp_shape[0];

            double * raw_det_data = extract_buffer <double> (
                det_data, "det_data", 2, temp_shape, {-1, -1}
            );
            int64_t n_all_det = temp_shape[0];
            int64_t n_samp = temp_shape[1];

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            int64_t * raw_n_amp_views = extract_buffer <int64_t> (
                n_amp_views, "n_amp_views", 1, temp_shape, {n_view}
            );

            // Optionally use flags
            bool use_flags = false;
            uint8_t * raw_det_flags = (uint8_t *)omgr.null;
            if (flag_index >= 0) {
                raw_det_flags = extract_buffer <uint8_t> (
                    flag_data, "flag_data", 2, temp_shape, {-1, n_samp}
                );
                use_flags = true;
            }

            if (offload) {
                #ifdef HAVE_OPENMP_TARGET

                double * dev_det_data = omgr.device_ptr(raw_det_data);
                uint8_t * dev_det_flags = omgr.device_ptr(raw_det_flags);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                double * dev_amplitudes = omgr.device_ptr(raw_amplitudes);

                # pragma omp target data \
                device(dev)              \
                map(to:                  \
                n_view,                  \
                n_samp,                  \
                data_index,              \
                flag_index,              \
                step_length,             \
                amp_offset,              \
                raw_n_amp_views,         \
                use_flags                \
                )
                {
                    int64_t offset = amp_offset;
                    # pragma omp target teams distribute firstprivate(offset) \
                        is_device_ptr( \
                            dev_amplitudes, \
                            dev_det_data, \
                            dev_det_flags, \
                            dev_intervals \
                        )
                    for (int64_t iview = 0; iview < n_view; iview++) {
                        # pragma omp parallel for default(shared)
                        for (
                            int64_t isamp = dev_intervals[iview].first;
                            isamp <= dev_intervals[iview].last;
                            isamp++
                        ) {
                            int64_t d = data_index * n_samp + isamp;
                            int64_t amp = offset + (int64_t)(isamp / step_length);
                            double contrib = 0.0;
                            if (use_flags) {
                                int64_t f = flag_index * n_samp + isamp;
                                uint8_t check = dev_det_flags[f] & flag_mask;
                                if (check == 0) {
                                    contrib = dev_det_data[d];
                                }
                            } else {
                                contrib = raw_det_data[d];
                            }
                            # pragma omp atomic
                            dev_amplitudes[amp] += contrib;
                        }
                        offset += raw_n_amp_views[iview];
                    }
                }

                #endif // ifdef HAVE_OPENMP_TARGET
            } else {
                int64_t offset = amp_offset;
                for (int64_t iview = 0; iview < n_view; iview++) {
                    #pragma omp parallel for default(shared)
                    for (
                        int64_t isamp = raw_intervals[iview].first;
                        isamp <= raw_intervals[iview].last;
                        isamp++
                    ) {
                        int64_t d = data_index * n_samp + isamp;
                        int64_t amp = offset + (int64_t)(isamp / step_length);
                        double contrib = 0.0;
                        if (use_flags) {
                            int64_t f = flag_index * n_samp + isamp;
                            uint8_t check = raw_det_flags[f] & flag_mask;
                            if (check == 0) {
                                contrib = raw_det_data[d];
                            }
                        } else {
                            contrib = raw_det_data[d];
                        }
                        #pragma omp atomic
                        raw_amplitudes[amp] += contrib;
                    }
                    offset += raw_n_amp_views[iview];
                }
            }
            return;
        });

    m.def(
        "template_offset_apply_diag_precond", [](
            py::buffer offset_var,
            py::buffer amplitudes_in,
            py::buffer amplitudes_out,
            bool use_accel
        ) {
            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            double * raw_amp_in = extract_buffer <double> (
                amplitudes_in, "amplitudes_in", 1, temp_shape, {-1}
            );
            int64_t n_amp = temp_shape[0];

            double * raw_amp_out = extract_buffer <double> (
                amplitudes_out, "amplitudes_out", 1, temp_shape, {n_amp}
            );

            double * raw_offset_var = extract_buffer <double> (
                offset_var, "offset_var", 1, temp_shape, {n_amp}
            );

            if (offload) {
                #ifdef HAVE_OPENMP_TARGET

                double * dev_amp_in = omgr.device_ptr(raw_amp_in);
                double * dev_amp_out = omgr.device_ptr(raw_amp_out);
                double * dev_offset_var = omgr.device_ptr(raw_offset_var);

                # pragma omp target data \
                device(dev)              \
                map(to:                  \
                n_amp                    \
                )
                {
                    # pragma omp target \
                        is_device_ptr( \
                            dev_amp_in,              \
                            dev_amp_out,             \
                            dev_offset_var           \
                        )
                    {
                        # pragma omp parallel for default(shared)
                        for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                            dev_amp_out[iamp] = dev_amp_in[iamp];
                            dev_amp_out[iamp] *= dev_offset_var[iamp];
                        }
                    }
                }

                #endif // ifdef HAVE_OPENMP_TARGET
            } else {
                #pragma omp parallel for default(shared)
                for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                    raw_amp_out[iamp] = raw_amp_in[iamp];
                    raw_amp_out[iamp] *= raw_offset_var[iamp];
                }
            }
            return;
        });

    // m.def(
    //     "template_offset_add_to_signal", [](int64_t step_length, py::buffer
    // amplitudes,
    //                                         py::buffer data) {
    //         pybuffer_check_1D <double> (amplitudes);
    //         pybuffer_check_1D <double> (data);
    //         py::buffer_info info_amplitudes = amplitudes.request();
    //         py::buffer_info info_data = data.request();
    //         int64_t n_amp = info_amplitudes.size;
    //         int64_t n_data = info_data.size;
    //         double * raw_amplitudes = reinterpret_cast <double *>
    // (info_amplitudes.ptr);
    //         double * raw_data = reinterpret_cast <double *> (info_data.ptr);
    //         toast::template_offset_add_to_signal(step_length, n_amp, raw_amplitudes,
    //                                              n_data, raw_data);
    //         return;
    //     }, py::arg("step_length"), py::arg("amplitudes"), py::arg(
    //         "data"), R"(
    //     Accumulate offset amplitudes to timestream data.

    //     Each amplitude value is accumulated to `step_length` number of samples.  The
    //     final offset will be at least this many samples, but may be more if the step
    //     size does not evenly divide into the number of samples.

    //     Args:
    //         step_length (int64):  The minimum number of samples for each offset.
    //         amplitudes (array):  The float64 amplitude values.
    //         data (array):  The float64 timestream values to accumulate.

    //     Returns:
    //         None.

    // )");

    // m.def(
    //     "template_offset_project_signal", [](int64_t step_length, py::buffer data,
    //                                          py::buffer amplitudes) {
    //         pybuffer_check_1D <double> (amplitudes);
    //         pybuffer_check_1D <double> (data);
    //         py::buffer_info info_amplitudes = amplitudes.request();
    //         py::buffer_info info_data = data.request();
    //         int64_t n_amp = info_amplitudes.size;
    //         int64_t n_data = info_data.size;
    //         double * raw_amplitudes = reinterpret_cast <double *>
    // (info_amplitudes.ptr);
    //         double * raw_data = reinterpret_cast <double *> (info_data.ptr);
    //         toast::template_offset_project_signal(step_length, n_data, raw_data,
    //                                               n_amp, raw_amplitudes);
    //         return;
    //     }, py::arg("step_length"), py::arg("data"), py::arg(
    //         "amplitudes"), R"(
    //     Accumulate timestream data into offset amplitudes.

    //     Chunks of `step_length` number of samples are accumulated into the offset
    //     amplitudes.  If step_length does not evenly divide into the total number of
    //     samples, the final amplitude will be extended to include the remainder.

    //     Args:
    //         step_length (int64):  The minimum number of samples for each offset.
    //         data (array):  The float64 timestream values.
    //         amplitudes (array):  The float64 amplitude values.

    //     Returns:
    //         None.

    // )");

    return;
}
