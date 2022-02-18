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
            py::array_t <double> const & focalplane,
            py::array_t <double> const & boresight,
            py::array_t <int32_t> const & quat_index,
            py::array_t <double> & quats,
            py::array_t <Interval> const & intervals,
            py::array_t <uint8_t> const & shared_flags,
            uint8_t shared_flag_mask,
            bool use_accel
        ) {
            size_t n_det = quat_index.shape(0);
            assert_shape <2> (focalplane, "focalplane", {n_det, 4});

            size_t n_samp = boresight.shape(0);
            assert_shape <2> (boresight, "boresight", {n_samp, 4});

            size_t n_view = intervals.shape(0);

            bool use_flags = false;
            if (shared_flags.shape(0) == n_samp) {
                use_flags = true;
            }

            assert_shape <3> (quats, "quats", {n_det, n_samp, 4});

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = ! omgr.device_is_host() && use_accel;

            double * dev_boresight = (double*)omgr.device_ptr((void*)boresight.data());
            double * dev_quats = (double*)omgr.device_ptr((void*)quats.data());
            Interval * dev_intervals = (Interval*)omgr.device_ptr(
                (void*)intervals.data()
            );
            double * fp = focalplane.data();
            int32_t * qindx = quat_index.data();

            if (use_flags) {
                uint8_t * dev_flags = (uint8_t*)omgr.device_ptr(
                    (void*)shared_flags.data()
                );
                #pragma omp target data \
                    device(dev) \
                    map(to: fp[0:n_det], qindx[0:n_det]) \
                    use_device_ptr(dev_boresight, dev_quats, dev_intervals, dev_flags) \
                    if(offload)
                #pragma omp target teams distribute \
                    parallel for collapse(3) \
                    if(offload)
                for (size_t iview = 0; iview < n_view; iview++) {
                    for (size_t idet = 0; idet < n_det; idet++) {
                        int32_t qidx = qindx[idet];
                        for (size_t isamp = dev_intervals[iview].first; isamp <= dev_intervals[iview].last; isamp++) {
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
                                &(fp[4 * idet]),
                                &(dev_quats[(qidx * 4 * n_samp) + 4 * isamp])
                            );
                        }
                    }
                }
            } else {
                #pragma omp target data \
                    device(dev) \
                    map(to: fp[0:n_det], qindx[0:n_det]) \
                    use_device_ptr(dev_boresight, dev_quats, dev_intervals) \
                    if(offload)
                #pragma omp target teams distribute \
                    parallel for collapse(3) \
                    if(offload)
                for (size_t iview = 0; iview < n_view; iview++) {
                    for (size_t idet = 0; idet < n_det; idet++) {
                        int32_t qidx = qindx[idet];
                        for (size_t isamp = dev_intervals[iview].first; isamp <= dev_intervals[iview].last; isamp++) {
                            double temp_bore[4];
                            temp_bore[0] = dev_boresight[4 * isamp];
                            temp_bore[1] = dev_boresight[4 * isamp + 1];
                            temp_bore[2] = dev_boresight[4 * isamp + 2];
                            temp_bore[3] = dev_boresight[4 * isamp + 3];
                            qa_mult(
                                temp_bore,
                                &(fp[4 * idet]),
                                &(dev_quats[(qidx * 4 * n_samp) + 4 * isamp])
                            );
                        }
                    }
                }
            }
            return;
        });

}
