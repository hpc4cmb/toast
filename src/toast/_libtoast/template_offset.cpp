
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <accelerator.hpp>

#include <intervals.hpp>

// FIXME:  docstrings need to be updated if we keep these versions of the code.

void init_template_offset(py::module &m)
{
    m.def(
        "template_offset_add_to_signal", [](
                                             int64_t step_length,
                                             int64_t amp_offset,
                                             py::buffer n_amp_views,
                                             py::buffer amplitudes,
                                             int32_t data_index,
                                             py::buffer det_data,
                                             py::buffer intervals,
                                             bool use_accel)
        {
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

            int64_t * amp_view_off = new int64_t(n_view);
            amp_view_off[0] = 0;
            for (int64_t iview = 1; iview < n_view; iview++) {
                amp_view_off[iview] = raw_n_amp_views[iview - 1];
            }

            if (offload) {
#ifdef HAVE_OPENMP_TARGET

                double * dev_det_data = omgr.device_ptr(raw_det_data);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                double * dev_amplitudes = omgr.device_ptr(raw_amplitudes);

// Calculate the maximum interval size on the CPU
int64_t max_interval_size = 0;
for (int64_t iview = 0; iview < n_view; iview++) {
    int64_t interval_size = raw_intervals[iview].last - raw_intervals[iview].first + 1;
    if (interval_size > max_interval_size) {
        max_interval_size = interval_size;
    }
}

#pragma omp target data map(to : n_view,     \
                                n_samp,      \
                                data_index,  \
                                step_length, \
                                amp_offset,  \
                                amp_view_off[0 : n_view])
{
#pragma omp target teams distribute parallel for collapse(2)
    for (int64_t iview = 0; iview < n_view; iview++) {
        for (int64_t isamp = 0; isamp < max_interval_size; isamp++) {
            // Adjust for the actual start of the interval
            int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

            // Check if the value is out of range for the current interval
            if (adjusted_isamp > dev_intervals[iview].last) {
                continue;
            }

            int64_t d = data_index * n_samp + adjusted_isamp;
            int64_t amp = amp_offset + amp_view_off[iview] + (int64_t)(
                (adjusted_isamp - dev_intervals[iview].first) / step_length
            );
            dev_det_data[d] += dev_amplitudes[amp];
        }
    }
}

#endif // ifdef HAVE_OPENMP_TARGET
            } else {
                for (int64_t iview = 0; iview < n_view; iview++) {
#pragma omp parallel for default(shared)
                    for (
                        int64_t isamp = raw_intervals[iview].first;
                        isamp <= raw_intervals[iview].last;
                        isamp++
                    ) {
                        int64_t d = data_index * n_samp + isamp;
                        int64_t amp = amp_offset + amp_view_off[iview] + (int64_t)(
                            (isamp - raw_intervals[iview].first) / step_length
                        );
                        raw_det_data[d] += raw_amplitudes[amp];
                    }
                }
            }

            delete amp_view_off;

            return; });

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
                                              bool use_accel)
        {
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

            int64_t * amp_view_off = new int64_t(n_view);
            amp_view_off[0] = 0;
            for (int64_t iview = 1; iview < n_view; iview++) {
                amp_view_off[iview] = raw_n_amp_views[iview - 1];
            }

            // Optionally use flags
            bool use_flags = false;
            uint8_t * raw_det_flags = omgr.null_ptr <uint8_t> ();
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

// Calculate the maximum interval size on the CPU
int64_t max_interval_size = 0;
for (int64_t iview = 0; iview < n_view; iview++) {
    int64_t interval_size = raw_intervals[iview].last - raw_intervals[iview].first + 1;
    if (interval_size > max_interval_size) {
        max_interval_size = interval_size;
    }
}

#pragma omp target data map(to : n_view,                  \
                                n_samp,                   \
                                data_index,               \
                                flag_index,               \
                                step_length,              \
                                amp_offset,               \
                                amp_view_off[0 : n_view], \
                                use_flags)
{
#pragma omp target teams distribute parallel for collapse(2)
    for (int64_t iview = 0; iview < n_view; iview++) {
        for (int64_t isamp = 0; isamp < max_interval_size; isamp++) {
            // Adjust for the actual start of the interval
            int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

            // Check if the value is out of range for the current interval
            if (adjusted_isamp > dev_intervals[iview].last) {
                continue;
            }

            int64_t d = data_index * n_samp + adjusted_isamp;
            int64_t amp = amp_offset + amp_view_off[iview] + (int64_t)(isamp / step_length);
            double contrib = 0.0;
            if (use_flags) {
                int64_t f = flag_index * n_samp + adjusted_isamp;
                uint8_t check = dev_det_flags[f] & flag_mask;
                if (check == 0) {
                    contrib = dev_det_data[d];
                }
            } else {
                contrib = dev_det_data[d];
            }
#pragma omp atomic update
            dev_amplitudes[amp] += contrib;
        }
    }
}

#endif // ifdef HAVE_OPENMP_TARGET
            } else {
                for (int64_t iview = 0; iview < n_view; iview++) {
#pragma omp parallel for default(shared)
                    for (
                        int64_t isamp = raw_intervals[iview].first;
                        isamp <= raw_intervals[iview].last;
                        isamp++
                    ) {
                        int64_t d = data_index * n_samp + isamp;
                        int64_t amp = amp_offset + amp_view_off[iview] + (int64_t)(
                            (isamp - raw_intervals[iview].first) / step_length
                        );
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
#pragma omp atomic update
                        raw_amplitudes[amp] += contrib;
                    }
                }
            }
            delete amp_view_off;
            return; });

    m.def(
        "template_offset_apply_diag_precond", [](
                                                  py::buffer offset_var,
                                                  py::buffer amplitudes_in,
                                                  py::buffer amplitudes_out,
                                                  bool use_accel)
        {
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

#pragma omp target data \
map(to : n_amp)
                {
#pragma omp target       \
is_device_ptr(           \
        dev_amp_in,      \
            dev_amp_out, \
            dev_offset_var)
                    {
#pragma omp parallel default(shared)
                        {
#pragma omp for
                            for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                                dev_amp_out[iamp] = dev_amp_in[iamp];
                                dev_amp_out[iamp] *= dev_offset_var[iamp];
                            }
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
            return; });

    return;
}
