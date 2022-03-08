// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>

#include <intervals.hpp>

#include <accelerator.hpp>


void init_ops_stokes_weights(py::module & m) {
    // FIXME:  For now, we are passing in the epsilon array.  Once the full
    // focalplane table is staged to GPU, change this code to use that.

    m.def(
        "stokes_weights_IQU", [](
            py::buffer quat_index,
            py::buffer quats,
            py::buffer weight_index,
            py::buffer weights,
            py::buffer hwp,
            py::buffer intervals,
            py::buffer epsilon,
            double cal,
            bool use_accel
        ) {
            // NOTE:  Flags are not needed here, since the quaternions
            // have already had bad samples converted to null rotations.

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_quat_index = extract_buffer <int32_t> (
                quat_index, "quat_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            int32_t * raw_weight_index = extract_buffer <int32_t> (
                weight_index, "weight_index", 1, temp_shape, {n_det}
            );

            double * raw_weights = extract_buffer <double> (
                weights, "weights", 3, temp_shape, {n_det, -1, 3}
            );
            int64_t n_samp = temp_shape[1];

            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {n_det, n_samp, 4}
            );

            double * raw_hwp = extract_buffer <double> (
                hwp, "hwp", 1, temp_shape, {n_samp}
            );

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            double * raw_epsilon = extract_buffer <double> (
                epsilon, "epsilon", 1, temp_shape, {n_det}
            );

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (! omgr.device_is_host()) && use_accel;

            double * dev_weights = raw_weights;
            double * dev_hwp = raw_hwp;
            double * dev_quats = raw_quats;
            Interval * dev_intervals = raw_intervals;
            if (offload) {
                dev_weights = (double*)omgr.device_ptr((void*)raw_weights);
                dev_hwp = (double*)omgr.device_ptr((void*)raw_hwp);
                dev_quats = (double*)omgr.device_ptr((void*)raw_quats);
                dev_intervals = (Interval*)omgr.device_ptr(
                    (void*)raw_intervals
                );
            }

            #pragma omp target data \
                device(dev) \
                map(to: \
                    raw_weight_index[0:n_det], \
                    raw_quat_index[0:n_det], \
                    raw_epsilon[0:n_det], \
                    n_view, \
                    n_det, \
                    n_samp \
                ) \
                use_device_ptr(dev_weights, dev_quats, dev_intervals, dev_hwp) \
                if(offload)
            {
                #pragma omp target teams distribute collapse(2) if(offload)
                for (int64_t idet = 0; idet < n_det; idet++) {
                    for (int64_t iview = 0; iview < n_view; iview++) {
                        #pragma omp parallel for
                        for (
                            int64_t isamp = dev_intervals[iview].first;
                            isamp <= dev_intervals[iview].last;
                            isamp++
                        ) {
                            const double xaxis[3] = {1.0, 0.0, 0.0};
                            const double zaxis[3] = {0.0, 0.0, 1.0};
                            double eta = (1.0 - raw_epsilon[idet]) / (1.0 + raw_epsilon[idet]);
                            int32_t q_indx = raw_quat_index[idet];
                            int32_t w_indx = raw_weight_index[idet];

                            double dir[3];
                            double orient[3];

                            int64_t off = (q_indx * 4 * n_samp) + 4 * isamp;
                            qa_rotate(&(dev_quats[off]), zaxis, dir);
                            qa_rotate(&(dev_quats[off]), xaxis, orient);

                            double y = orient[0] * dir[1] - orient[1] * dir[0];
                            double x = orient[0] * (-dir[2] * dir[0]) +
                                    orient[1] * (-dir[2] * dir[1]) +
                                    orient[2] * (dir[0] * dir[0] + dir[1] * dir[1]);
                            double ang = atan2(y, x);

                            ang += 2.0 * dev_hwp[isamp];
                            ang *= 2.0;
                            double cang = cos(ang);
                            double sang = sin(ang);

                            off = (w_indx * 3 * n_samp) + 3 * isamp;
                            dev_weights[off] = cal;
                            dev_weights[off + 1] = cang * eta * cal;
                            dev_weights[off + 2] = sang * eta * cal;
                        }
                    }
                }
            }
            return;
        });

    m.def(
        "stokes_weights_I", [](
            py::buffer weight_index,
            py::buffer weights,
            py::buffer intervals,
            double cal,
            bool use_accel
        ) {
            // NOTE:  Flags are not needed here, since the quaternions
            // have already had bad samples converted to null rotations.

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_weight_index = extract_buffer <int32_t> (
                weight_index, "weight_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            double * raw_weights = extract_buffer <double> (
                weights, "weights", 2, temp_shape, {n_det, -1}
            );
            int64_t n_samp = temp_shape[1];

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (! omgr.device_is_host()) && use_accel;

            double * dev_weights = raw_weights;
            Interval * dev_intervals = raw_intervals;
            if (offload) {
                dev_weights = (double*)omgr.device_ptr((void*)raw_weights);
                dev_intervals = (Interval*)omgr.device_ptr(
                    (void*)raw_intervals
                );
            }

            #pragma omp target data \
                device(dev) \
                map(to: \
                    raw_weight_index[0:n_det], \
                    n_view, \
                    n_det, \
                    n_samp \
                ) \
                use_device_ptr(dev_weights, dev_intervals) \
                if(offload)
            {
                #pragma omp target teams distribute collapse(2) if(offload)
                for (int64_t idet = 0; idet < n_det; idet++) {
                    for (int64_t iview = 0; iview < n_view; iview++) {
                        #pragma omp parallel for
                        for (
                            int64_t isamp = dev_intervals[iview].first;
                            isamp <= dev_intervals[iview].last;
                            isamp++
                        ) {
                            int32_t w_indx = raw_weight_index[idet];
                            int64_t off = (w_indx * n_samp) + isamp;
                            dev_weights[off] = cal;
                        }
                    }
                }
            }
            return;
        });


}
