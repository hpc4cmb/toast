// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <intervals.hpp>

#include <qarray.hpp>

#include <accelerator.hpp>


void init_ops_pointing_detector(py::module & m) {
    // FIXME:  We are temporarily passing in an array of detector quaternions,
    // but eventually should support passing the core focalplane table.

    m.def(
        "pointing_detector", [](
            py::buffer focalplane,
            py::buffer boresight,
            py::buffer quat_index,
            py::buffer quats,
            py::buffer intervals,
            py::buffer shared_flags,
            uint8_t shared_flag_mask,
            bool use_accel
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <size_t> temp_shape(3);

            int32_t * raw_quat_index = extract_buffer <int32_t> (
                quat_index, "quat_index", 1, temp_shape, {-1}
            );
            size_t n_det = temp_shape[0];

            double * raw_focalplane = extract_buffer <double> (
                focalplane, "focalplane", 2, temp_shape, {n_det, 4}
            );

            double * raw_boresight = extract_buffer <double> (
                boresight, "boresight", 2, temp_shape, {-1, 4}
            );
            size_t n_samp = temp_shape[0];

            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {n_det, n_samp, 4}
            );

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            size_t n_view = temp_shape[0];

            uint8_t * raw_flags = extract_buffer <uint8_t> (
                shared_flags, "flags", 1, temp_shape, {-1}
            );
            bool use_flags = false;
            if (temp_shape[0] == n_samp) {
                use_flags = true;
            }

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (! omgr.device_is_host()) && use_accel;

            double * dev_boresight = raw_boresight;
            double * dev_quats = raw_quats;
            Interval * dev_intervals = raw_intervals;
            if (offload) {
                dev_boresight = (double*)omgr.device_ptr((void*)raw_boresight);
                dev_quats = (double*)omgr.device_ptr((void*)raw_quats);
                dev_intervals = (Interval*)omgr.device_ptr(
                    (void*)raw_intervals
                );
            }

            if (use_flags) {
                uint8_t * dev_flags = raw_flags;
                if (offload) {
                    dev_flags = (uint8_t*)omgr.device_ptr((void*)raw_flags);
                }
                #pragma omp target data \
                    device(dev) \
                    map(to: \
                        raw_focalplane[0:n_det], \
                        raw_quat_index[0:n_det], \
                        shared_flag_mask, \
                        n_view, \
                        n_det, \
                        n_samp, \
                    ) \
                    use_device_ptr(dev_boresight, dev_quats, dev_intervals, dev_flags) \
                    if(offload)
                {
                    #pragma omp target teams distribute collapse(2) if(offload)
                    for (size_t iview = 0; iview < n_view; iview++) {
                        for (size_t idet = 0; idet < n_det; idet++) {
                            #pragma omp parallel for
                            for (
                                size_t isamp = dev_intervals[iview].first;
                                isamp <= dev_intervals[iview].last;
                                isamp++
                            ) {
                                int32_t qidx = raw_quat_index[idet];
                                double temp_bore[4];
                                if ((dev_flags[isamp] & shared_flag_mask) == 0) {
                                    temp_bore[0] = dev_boresight[4 * isamp];
                                    temp_bore[1] = dev_boresight[4 * isamp + 1];
                                    temp_bore[2] = dev_boresight[4 * isamp + 2];
                                    temp_bore[3] = dev_boresight[4 * isamp + 3];
                                } else {
                                    temp_bore[0] = 0.0;
                                    temp_bore[1] = 0.0;
                                    temp_bore[2] = 0.0;
                                    temp_bore[3] = 1.0;
                                }
                                qa_mult(
                                    temp_bore,
                                    &(raw_focalplane[4 * idet]),
                                    &(dev_quats[(qidx * 4 * n_samp) + 4 * isamp])
                                );
                            }
                        }
                    }
                }
            } else {
                #pragma omp target data \
                    device(dev) \
                    map(to: \
                        raw_focalplane[0:n_det], \
                        raw_quat_index[0:n_det], \
                        n_view, \
                        n_det, \
                        n_samp, \
                    ) \
                    use_device_ptr(dev_boresight, dev_quats, dev_intervals) \
                    if(offload)
                {
                    #pragma omp target teams distribute collapse(2) if(offload)
                    for (size_t iview = 0; iview < n_view; iview++) {
                        for (size_t idet = 0; idet < n_det; idet++) {
                            #pragma omp parallel for
                            for (
                                size_t isamp = dev_intervals[iview].first;
                                isamp <= dev_intervals[iview].last;
                                isamp++
                            ) {
                                int32_t qidx = raw_quat_index[idet];
                                double temp_bore[4];
                                temp_bore[0] = dev_boresight[4 * isamp];
                                temp_bore[1] = dev_boresight[4 * isamp + 1];
                                temp_bore[2] = dev_boresight[4 * isamp + 2];
                                temp_bore[3] = dev_boresight[4 * isamp + 3];
                                qa_mult(
                                    temp_bore,
                                    &(raw_focalplane[4 * idet]),
                                    &(dev_quats[(qidx * 4 * n_samp) + 4 * isamp])
                                );
                            }
                        }
                    }
                }
            }
            return;
        });

}
