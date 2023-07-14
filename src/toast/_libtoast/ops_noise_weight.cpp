// Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <intervals.hpp>

#include <accelerator.hpp>

void init_ops_noise_weight(py::module &m)
{
    m.def(
        "noise_weight", [](
                            py::buffer det_data,
                            py::buffer data_index,
                            py::buffer intervals,
                            py::buffer detector_weights,
                            bool use_accel)
        {
            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_data_index = extract_buffer <int32_t> (
                data_index, "data_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            double * raw_det_data = extract_buffer <double> (
                det_data, "det_data", 2, temp_shape, {-1, -1}
            );
            int64_t n_samp = temp_shape[1];

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            double * raw_det_weights = extract_buffer <double> (
                detector_weights, "detector_weights", 1, temp_shape, {n_det}
            );

            if (offload) {
#ifdef HAVE_OPENMP_TARGET

                double * dev_det_data = omgr.device_ptr(raw_det_data);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);

// Calculate the maximum interval size on the CPU
int64_t max_interval_size = 0;
for (int64_t iview = 0; iview < n_view; iview++) {
    int64_t interval_size = raw_intervals[iview].last - raw_intervals[iview].first + 1;
    if (interval_size > max_interval_size) {
        max_interval_size = interval_size;
    }
}

#pragma omp target data map(to : raw_data_index[0 : n_det], \
                                raw_det_weights[0 : n_det], \
                                n_view,                     \
                                n_det,                      \
                                n_samp)
{
#pragma omp target teams distribute parallel for collapse(3)
    for (int64_t idet = 0; idet < n_det; idet++) {
        for (int64_t iview = 0; iview < n_view; iview++) {
            for (int64_t isamp = 0; isamp < max_interval_size; isamp++) {
                // Adjust for the actual start of the interval
                int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                // Check if the value is out of range for the current interval
                if (adjusted_isamp > dev_intervals[iview].last) {
                    continue;
                }

                int32_t d_indx = raw_data_index[idet];
                int64_t off_d = d_indx * n_samp + adjusted_isamp;
                dev_det_data[off_d] *= raw_det_weights[idet];
            }
        }
    }
}

#endif // ifdef HAVE_OPENMP_TARGET
            } else {
                for (int64_t idet = 0; idet < n_det; idet++) {
                    for (int64_t iview = 0; iview < n_view; iview++) {
#pragma omp parallel for default(shared)
                        for (
                            int64_t isamp = raw_intervals[iview].first;
                            isamp <= raw_intervals[iview].last;
                            isamp++
                        ) {
                            int32_t d_indx = raw_data_index[idet];
                            int64_t off_d = d_indx * n_samp + isamp;
                            raw_det_data[off_d] *= raw_det_weights[idet];
                        }
                    }
                }
            }

            return; });
}
