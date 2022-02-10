// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>

#include <accelerator.hpp>


void init_ops_stokes_weights(py::module & m) {
    // FIXME:  For now, we are passing in the epsilon array.  Once the full
    // focalplane table is staged to GPU, change this code to use that.

    m.def(
        "stokes_weights_IQU", [](
            py::buffer quat_indx,
            py::buffer quats,
            py::buffer hwp,
            py::buffer weight_indx,
            py::buffer weights,
            py::buffer epsilon,
            double cal
        ) {
            // NOTE:  Flags are not needed here, since the quaternions
            // have already had bad samples converted to null rotations.

            auto info_quatindx = quat_indx.request();
            int32_t * raw_quatindx = reinterpret_cast <int32_t *> (info_quatindx.ptr);

            auto info_weightindx = weight_indx.request();
            int32_t * raw_weightindx = reinterpret_cast <int32_t *> (info_weightindx.ptr);

            auto info_weights = weights.request();
            double * raw_weights = reinterpret_cast <double *> (info_weights.ptr);

            auto info_quats = quats.request();
            double * raw_quats = reinterpret_cast <double *> (info_quats.ptr);

            auto info_hwp = hwp.request();
            double * raw_hwp = reinterpret_cast <double *> (info_hwp.ptr);

            auto info_epsilon = epsilon.request();
            double * raw_epsilon = reinterpret_cast <double *> (info_epsilon.ptr);

            size_t n_det = info_weightindx.shape[0];
            if (info_quatindx.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "weight and quat indices do not have same number of dets";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_weights.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det weights have fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t n_samp = info_weights.shape[1];
            if (info_weights.shape[2] != 3) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "IQU weights array does not have 3 elements per sample";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_hwp.shape[0] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "HWP angle does not have same number of samples as weights";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_epsilon.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "epsilon array has different number of detectors than weights";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_quats.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det quats have fewer detectors than index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_quats.shape[1] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det quats do not have same number of samples as weights";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_quats.shape[2] != 4) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det quats do not have 4 elements per sample";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t len_weights = info_weights.shape[0] * n_samp * 3;
            size_t len_hwp = n_samp;
            size_t len_quats = info_quats.shape[0] * n_samp * 4;

            // #pragma \
            // acc data copyin(n_det, n_samp, raw_epsilon[:n_det], cal, raw_quatindx[:n_det], raw_weightindx[:n_det]) present(raw_weights[:len_weights], raw_hwp[:len_hwp], raw_quats[:len_quats])
            // {
            //     if (fake_openacc()) {
            //         // Set all "present" data to point at the fake device pointers
            //         auto & fake = FakeMemPool::get();
            //         raw_weights = (double *)fake.device_ptr(raw_weights);
            //         raw_hwp = (double *)fake.device_ptr(raw_hwp);
            //         raw_quats = (double *)fake.device_ptr(raw_quats);
            //     }
            //     #pragma acc parallel
            //     #pragma acc loop independent
            //     for (size_t idet = 0; idet < n_det; idet++) {
            //         double eta = (1.0 - raw_epsilon[idet]) / (1.0 + raw_epsilon[idet]);
            //         int32_t q_indx = raw_quatindx[idet];
            //         int32_t w_indx = raw_weightindx[idet];
            //         #pragma acc loop independent
            //         for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //             const double xaxis[3] = {1.0, 0.0, 0.0};
            //             const double zaxis[3] = {0.0, 0.0, 1.0};
            //             double dir[3];
            //             double orient[3];

            //             size_t off = (q_indx * 4 * n_samp) + 4 * isamp;
            //             qa_rotate(&(raw_quats[off]), zaxis, dir);
            //             qa_rotate(&(raw_quats[off]), xaxis, orient);

            //             double y = orient[0] * dir[1] - orient[1] * dir[0];
            //             double x = orient[0] * (-dir[2] * dir[0]) +
            //                        orient[1] * (-dir[2] * dir[1]) +
            //                        orient[2] * (dir[0] * dir[0] + dir[1] * dir[1]);
            //             double ang = atan2(y, x);

            //             ang += 2.0 * raw_hwp[isamp];
            //             ang *= 2.0;
            //             double cang = cos(ang);
            //             double sang = sin(ang);

            //             off = (w_indx * 3 * n_samp) + 3 * isamp;
            //             raw_weights[off] = cal;
            //             raw_weights[off + 1] = cang * eta * cal;
            //             raw_weights[off + 2] = sang * eta * cal;
            //             // std::cout << "stokes IQU " << isamp << ": "
            //             // << raw_weights[off + 0] << " "
            //             // << raw_weights[off + 1] << " "
            //             // << raw_weights[off + 2] << " "
            //             // << std::endl;
            //         }
            //     }
            // }
            return;
        });

    m.def(
        "stokes_weights_I", [](
            py::buffer weight_indx,
            py::buffer weights,
            double cal
        ) {
            // NOTE:  Flags are not needed here, since the quaternions
            // have already had bad samples converted to null rotations.

            auto info_weightindx = weight_indx.request();
            int32_t * raw_weightindx = reinterpret_cast <int32_t *> (info_weightindx.ptr);

            auto info_weights = weights.request();
            double * raw_weights = reinterpret_cast <double *> (info_weights.ptr);

            size_t n_det = info_weightindx.shape[0];
            if (info_weights.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det weights have fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t n_samp = info_weights.shape[1];

            if (info_weights.ndim > 2 && info_weights.shape[2] != 1) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "I weights array does not have 1 element per sample";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t len_weights = info_weights.shape[0] * n_samp;

            // #pragma \
            // acc data copyin(n_det, n_samp, cal, raw_weightindx[:n_det]) present(raw_weights[:len_weights])
            // {
            //     if (fake_openacc()) {
            //         // Set all "present" data to point at the fake device pointers
            //         auto & fake = FakeMemPool::get();
            //         raw_weights = (double *)fake.device_ptr(raw_weights);
            //     }
            //     #pragma acc parallel
            //     #pragma acc loop independent
            //     for (size_t idet = 0; idet < n_det; idet++) {
            //         int32_t w_indx = raw_weightindx[idet];
            //         #pragma acc loop independent
            //         for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //             raw_weights[(w_indx * n_samp) + isamp] = cal;
            //             // std::cout << "stokes I " << isamp << ": "
            //             // << raw_weights[(w_indx * n_samp) + isamp]
            //             // << std::endl;
            //         }
            //     }
            // }
            return;
        });

}
