// Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <intervals.hpp>

#include <accelerator.hpp>


void init_ops_noise_weight(py::module & m) {
    m.def(
        "noise_weight", [](
            py::buffer det_data,
            py::buffer data_index,
            py::buffer intervals,
            py::buffer detector_weights,
            bool use_accel
        ) {
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

                # pragma omp target data  \
                device(dev)               \
                map(to:                   \
                raw_data_index[0:n_det],  \
                raw_det_weights[0:n_det], \
                n_view,                   \
                n_det,                    \
                n_samp                    \
                )
                {
                    # pragma omp target teams distribute collapse(2) \
                    is_device_ptr(                                   \
                    dev_det_data,                                    \
                    dev_intervals                                    \
                    )
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
                            # pragma omp parallel
                            {
                                # pragma omp for default(shared)
                                for (
                                    int64_t isamp = dev_intervals[iview].first;
                                    isamp <= dev_intervals[iview].last;
                                    isamp++
                                ) {
                                    int32_t d_indx = raw_data_index[idet];
                                    int64_t off_d = d_indx * n_samp + isamp;
                                    dev_det_data[off_d] *= raw_det_weights[idet];
                                }
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

            return;
        });
}
