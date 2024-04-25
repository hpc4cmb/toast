// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>

#include <intervals.hpp>

#include <accelerator.hpp>

#ifdef HAVE_OPENMP_TARGET
# pragma omp declare target
#endif // ifdef HAVE_OPENMP_TARGET

// FIXME:  this ridiculous code duplication is due to nvc++
// not supporting loadable device objects in shared libraries.
// So we must duplicate this across compilation units.

void stokes_weights_qa_rotate(
    double const * q_in,
    double const * v_in,
    double * v_out
) {
    // The input quaternion has already been normalized on the host.

    double xw = q_in[3] * q_in[0];
    double yw = q_in[3] * q_in[1];
    double zw = q_in[3] * q_in[2];
    double x2 = -q_in[0] * q_in[0];
    double xy = q_in[0] * q_in[1];
    double xz = q_in[0] * q_in[2];
    double y2 = -q_in[1] * q_in[1];
    double yz = q_in[1] * q_in[2];
    double z2 = -q_in[2] * q_in[2];

    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                    (yw + xz) * v_in[2]) + v_in[0];

    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                    (yz - xw) * v_in[2]) + v_in[1];

    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                    (x2 + y2) * v_in[2]) + v_in[2];

    return;
}

void stokes_weights_alpha(
    double const * quats,
    double & alpha
) {
    const double xaxis[3] = {1.0, 0.0, 0.0};
    const double zaxis[3] = {0.0, 0.0, 1.0};
    double vd[3];
    double vo[3];

    stokes_weights_qa_rotate(quats, zaxis, vd);
    stokes_weights_qa_rotate(quats, xaxis, vo);

    double ang_xy = ::atan2(vd[1], vd[0]);
    double vm_x = vd[2] * ::cos(ang_xy);
    double vm_y = vd[2] * ::sin(ang_xy);
    double vm_z = -::sqrt(1.0 - vd[2] * vd[2]);

    double alpha_y = (
        vd[0] * (vm_y * vo[2] - vm_z * vo[1]) - vd[1] * (vm_x * vo[2] - vm_z * vo[0]) +
        vd[2] * (vm_x * vo[1] - vm_y * vo[0])
    );
    double alpha_x = (vm_x * vo[0] + vm_y * vo[1] + vm_z * vo[2]);

    alpha = ::atan2(alpha_y, alpha_x);
    return;
}

void stokes_weights_IQU_inner(
    double const * cal,
    int32_t const * quat_index,
    int32_t const * weight_index,
    double const * quats,
    double const * epsilon,
    int64_t const & isamp,
    int64_t const & n_samp,
    int64_t const & idet,
    double const & U_sign,
    double * weights
) {
    double eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet]);
    int32_t q_indx = quat_index[idet];
    int32_t w_indx = weight_index[idet];
    int64_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;

    double alpha;
    stokes_weights_alpha(&(quats[qoff]), alpha);

    alpha *= 2.0;
    double cang = ::cos(alpha);
    double sang = ::sin(alpha);

    int64_t woff = (w_indx * 3 * n_samp) + 3 * isamp;
    weights[woff] = cal[idet];
    weights[woff + 1] = cang * eta * cal[idet];
    weights[woff + 2] = sang * eta * cal[idet] * U_sign;
    return;
}

void stokes_weights_IQU_inner_hwp(
    double const * cal,
    int32_t const * quat_index,
    int32_t const * weight_index,
    double const * quats,
    double const * hwp,
    double const * epsilon,
    double const * gamma,
    int64_t const & isamp,
    int64_t const & n_samp,
    int64_t const & idet,
    double const & U_sign,
    double * weights
) {
    double eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet]);
    int32_t q_indx = quat_index[idet];
    int32_t w_indx = weight_index[idet];
    int64_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;

    double alpha;
    stokes_weights_alpha(&(quats[qoff]), alpha);

    double ang = 2.0 * (2.0 * (gamma[idet] - hwp[isamp]) - alpha);
    double cang = ::cos(ang);
    double sang = ::sin(ang);

    int64_t woff = (w_indx * 3 * n_samp) + 3 * isamp;
    weights[woff] = cal[idet];
    weights[woff + 1] = cang * eta * cal[idet];
    weights[woff + 2] = -sang * eta * cal[idet] * U_sign;

    return;
}

#ifdef HAVE_OPENMP_TARGET
# pragma omp end declare target
#endif // ifdef HAVE_OPENMP_TARGET

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
            py::buffer gamma,
            py::buffer cal,
            bool IAU,
            bool use_accel
        ) {
            // NOTE:  Flags are not needed here, since the quaternions
            // have already had bad samples converted to null rotations.

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

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
                weights, "weights", 3, temp_shape, {-1, -1, 3}
            );
            int64_t n_samp = temp_shape[1];

            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {-1, n_samp, 4}
            );

            double * raw_hwp = extract_buffer <double> (
                hwp, "hwp", 1, temp_shape, {-1}
            );
            bool use_hwp = true;
            if (temp_shape[0] != n_samp) {
                // We are not using a HWP
                raw_hwp = omgr.null_ptr <double> ();
                use_hwp = false;
            }

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            double * raw_epsilon = extract_buffer <double> (
                epsilon, "epsilon", 1, temp_shape, {n_det}
            );

            double * raw_cal = extract_buffer <double> (
                cal, "cal", 1, temp_shape, {n_det}
            );

            double * raw_gamma = extract_buffer <double> (
                gamma, "gamma", 1, temp_shape, {n_det}
            );

            double U_sign = 1.0;
            if (IAU) {
                U_sign = -1.0;
            }

            if (offload) {
                #ifdef HAVE_OPENMP_TARGET

                double * dev_quats = omgr.device_ptr(raw_quats);
                double * dev_weights = omgr.device_ptr(raw_weights);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                double * dev_hwp = omgr.device_ptr(
                    raw_hwp);

                // Calculate the maximum interval size on the CPU
                int64_t max_interval_size = 0;
                for (int64_t iview = 0; iview < n_view; iview++) {
                    int64_t interval_size = raw_intervals[iview].last -
                                            raw_intervals[iview].first + 1;
                    if (interval_size > max_interval_size) {
                        max_interval_size = interval_size;
                    }
                }

                # pragma omp target data map(     \
                to : raw_weight_index[0 : n_det], \
                raw_quat_index[0 : n_det],        \
                raw_epsilon[0 : n_det],           \
                raw_gamma[0 : n_det],             \
                raw_cal[0 : n_det],               \
                U_sign,                           \
                n_view,                           \
                n_det,                            \
                n_samp,                           \
                max_interval_size                 \
                )
                {
                    if (!use_hwp) {
                        // No HWP
                        # pragma omp target teams distribute parallel for collapse(3) \
                        schedule(static,1)                                            \
                        is_device_ptr(                                                \
                        dev_weights,                                                  \
                        dev_quats,                                                    \
                        dev_intervals                                                 \
                        )
                        for (int64_t idet = 0; idet < n_det; idet++) {
                            for (int64_t iview = 0; iview < n_view; iview++) {
                                for (int64_t isamp = 0; isamp < max_interval_size;
                                     isamp++) {
                                    // adjust for the actual start of the interval
                                    int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                                    // check if the value is out of range for the
                                    // current interval
                                    if (adjusted_isamp > dev_intervals[iview].last) {
                                        continue;
                                    }

                                    stokes_weights_IQU_inner(
                                        raw_cal,
                                        raw_quat_index,
                                        raw_weight_index,
                                        dev_quats,
                                        raw_epsilon,
                                        adjusted_isamp,
                                        n_samp,
                                        idet,
                                        U_sign,
                                        dev_weights
                                    );
                                }
                            }
                        }
                    } else {
                        // We have a HWP
                        # pragma omp target teams distribute parallel for collapse(3) \
                        schedule(static,1)                                            \
                        is_device_ptr(                                                \
                        dev_weights,                                                  \
                        dev_quats,                                                    \
                        dev_hwp,                                                      \
                        dev_intervals                                                 \
                        )
                        for (int64_t idet = 0; idet < n_det; idet++) {
                            for (int64_t iview = 0; iview < n_view; iview++) {
                                for (int64_t isamp = 0; isamp < max_interval_size;
                                     isamp++) {
                                    // adjust for the actual start of the interval
                                    int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                                    // check if the value is out of range for the
                                    // current interval
                                    if (adjusted_isamp > dev_intervals[iview].last) {
                                        continue;
                                    }

                                    stokes_weights_IQU_inner_hwp(
                                        raw_cal,
                                        raw_quat_index,
                                        raw_weight_index,
                                        dev_quats,
                                        dev_hwp,
                                        raw_epsilon,
                                        raw_gamma,
                                        adjusted_isamp,
                                        n_samp,
                                        idet,
                                        U_sign,
                                        dev_weights
                                    );
                                }
                            }
                        }
                    }
                }

                #endif // ifdef HAVE_OPENMP_TARGET
            } else {
                if (!use_hwp) {
                    // No HWP
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
                            #pragma omp parallel for default(shared) schedule(static)
                            for (
                                int64_t isamp = raw_intervals[iview].first;
                                isamp <= raw_intervals[iview].last;
                                isamp++
                            ) {
                                stokes_weights_IQU_inner(
                                    raw_cal,
                                    raw_quat_index,
                                    raw_weight_index,
                                    raw_quats,
                                    raw_epsilon,
                                    isamp,
                                    n_samp,
                                    idet,
                                    U_sign,
                                    raw_weights
                                );
                            }
                        }
                    }
                } else {
                    // We are using a HWP
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
                            #pragma omp parallel for default(shared) schedule(static)
                            for (
                                int64_t isamp = raw_intervals[iview].first;
                                isamp <= raw_intervals[iview].last;
                                isamp++
                            ) {
                                stokes_weights_IQU_inner_hwp(
                                    raw_cal,
                                    raw_quat_index,
                                    raw_weight_index,
                                    raw_quats,
                                    raw_hwp,
                                    raw_epsilon,
                                    raw_gamma,
                                    isamp,
                                    n_samp,
                                    idet,
                                    U_sign,
                                    raw_weights
                                );
                            }
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
            py::buffer cal,
            bool use_accel
        ) {
            // NOTE:  Flags are not needed here, since the quaternions
            // have already had bad samples converted to null rotations.

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

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

            double * raw_cal = extract_buffer <double> (
                cal, "cal", 1, temp_shape, {n_det}
            );

            if (offload) {
                #ifdef HAVE_OPENMP_TARGET

                double * dev_weights = omgr.device_ptr(raw_weights);
                Interval * dev_intervals = omgr.device_ptr(
                    raw_intervals);

                // Calculate the maximum interval size on the CPU
                int64_t max_interval_size = 0;
                for (int64_t iview = 0; iview < n_view; iview++) {
                    int64_t interval_size = raw_intervals[iview].last -
                                            raw_intervals[iview].first + 1;
                    if (interval_size > max_interval_size) {
                        max_interval_size = interval_size;
                    }
                }

                # pragma omp target data map(     \
                to : raw_weight_index[0 : n_det], \
                n_view,                           \
                n_det,                            \
                n_samp,                           \
                raw_cal[0 : n_det]                \
                )
                {
                    # pragma omp target teams distribute parallel for collapse(3) \
                    schedule(static,1)                                            \
                    is_device_ptr(                                                \
                    dev_weights,                                                  \
                    dev_intervals                                                 \
                    )
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
                            for (int64_t isamp = 0; isamp < max_interval_size; isamp++) {
                                // Adjust for the actual start of the interval
                                int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                                // Check if the value is out of range for the current
                                // interval
                                if (adjusted_isamp > dev_intervals[iview].last) {
                                    continue;
                                }

                                int32_t w_indx = raw_weight_index[idet];
                                int64_t off = (w_indx * n_samp) + adjusted_isamp;
                                dev_weights[off] = raw_cal[idet];
                            }
                        }
                    }
                }

                #endif // ifdef HAVE_OPENMP_TARGET
            } else {
                for (int64_t idet = 0; idet < n_det; idet++) {
                    for (int64_t iview = 0; iview < n_view; iview++) {
                        #pragma omp parallel for default(shared) schedule(static)
                        for (
                            int64_t isamp = raw_intervals[iview].first;
                            isamp <= raw_intervals[iview].last;
                            isamp++
                        ) {
                            int32_t w_indx = raw_weight_index[idet];
                            int64_t off = (w_indx * n_samp) + isamp;
                            raw_weights[off] = raw_cal[idet];
                        }
                    }
                }
            }
            return;
        });
}
