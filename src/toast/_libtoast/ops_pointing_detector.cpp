// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>

#ifdef HAVE_OPENACC
# include <openacc.h>
#endif // ifdef HAVE_OPENACC


void init_ops_pointing_detector(py::module & m) {
    // FIXME:  We are temporarily passing in an array of detector quaternions,
    // but eventually should support passing the core focalplane table.

    m.def(
        "pointing_detector", [](
            py::buffer focalplane,
            py::buffer boresight,
            py::buffer quat_indx,
            py::buffer quats,
            py::buffer shared_flags,
            uint8_t shared_flag_mask
        ) {
            auto info_fp = focalplane.request();
            double * raw_fp = reinterpret_cast <double *> (info_fp.ptr);

            auto info_qindx = quat_indx.request();
            int32_t * raw_qindx = reinterpret_cast <int32_t *> (info_qindx.ptr);

            auto info_bore = boresight.request();
            double * raw_bore = reinterpret_cast <double *> (info_bore.ptr);

            auto info_quats = quats.request();
            double * raw_quats = reinterpret_cast <double *> (info_quats.ptr);

            auto info_flags = shared_flags.request();
            uint8_t * raw_flags = reinterpret_cast <uint8_t *> (info_flags.ptr);

            size_t n_det = info_qindx.shape[0];

            if (info_fp.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "focalplane quats have different num dets than quat index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_fp.shape[1] != 4) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "focalplane quats do not have 4 elements";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t n_samp = info_bore.shape[0];
            if (info_bore.shape[1] != 4) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "boresight quats do not have 4 elements";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_flags.shape[0] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "shared flags do not have same number of samples as boresight";
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
                o << "det quats do not have same number of samples as boresight";
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
            size_t len_fp = n_det * 4;
            size_t len_bore = n_samp * 4;
            size_t len_flags = n_samp;
            size_t len_quats = info_quats.shape[0] * n_samp * 4;

            #pragma \
            acc data copyin(shared_flag_mask, n_det, n_samp, raw_fp[:len_fp], raw_qindx[:n_det]) present(raw_bore[:len_bore], raw_flags[:len_flags], raw_quats[:len_quats])
            {
                if (fake_openacc()) {
                    // Set all "present" data to point at the fake device pointers
                    auto & fake = FakeMemPool::get();
                    raw_bore = (double *)fake.device_ptr(raw_bore);
                    raw_flags = (uint8_t *)fake.device_ptr(raw_flags);
                    raw_quats = (double *)fake.device_ptr(raw_quats);
                    // for (size_t isamp = 0; isamp < n_samp; isamp++) {
                    //     std::cout << "bore: " << isamp << ": " << raw_bore[4*isamp] << ", " << raw_bore[4*isamp+1] << ", " << raw_bore[4*isamp+2] << ", " << raw_bore[4*isamp+3] << " flag = " << (int)raw_flags[isamp] << " mask = " << (int)shared_flag_mask << std::endl;
                    // }
                }
                #pragma acc parallel
                #pragma acc loop independent
                for (size_t idet = 0; idet < n_det; idet++) {
                    int32_t q_indx = raw_qindx[idet];
                    #pragma acc loop independent
                    for (size_t isamp = 0; isamp < n_samp; isamp++) {
                        double temp_bore[4];
                        if ((raw_flags[isamp] & shared_flag_mask) == 0) {
                            temp_bore[0] = raw_bore[4 * isamp];
                            temp_bore[1] = raw_bore[4 * isamp + 1];
                            temp_bore[2] = raw_bore[4 * isamp + 2];
                            temp_bore[3] = raw_bore[4 * isamp + 3];
                        } else {
                            temp_bore[0] = 0.0;
                            temp_bore[1] = 0.0;
                            temp_bore[2] = 0.0;
                            temp_bore[3] = 1.0;
                        }
                        qa_mult(
                            temp_bore,
                            &(raw_fp[4 * idet]),
                            &(raw_quats[(q_indx * 4 * n_samp) + 4 * isamp])
                        );
                        // std::cout << "detpt " << isamp << ": "
                        // << raw_quats[(q_indx * 4 * n_samp) + 4 * isamp] << " "
                        // << raw_quats[(q_indx * 4 * n_samp) + 4 * isamp + 1] << " "
                        // << raw_quats[(q_indx * 4 * n_samp) + 4 * isamp + 2] << " "
                        // << raw_quats[(q_indx * 4 * n_samp) + 4 * isamp + 3] << " "
                        // << std::endl;
                    }
                }
                // if (fake_openacc()) {
                //     for (size_t idet = 0; idet < n_det; idet++) {
                //         for (size_t isamp = 0; isamp < n_samp; isamp++) {
                //             std::cout << "quat " << idet << ": " << isamp << ": " << raw_quats[4*(idet * n_samp + isamp)] << ", " << raw_quats[4*(idet * n_samp + isamp)+1] << ", " << raw_quats[4*(idet * n_samp + isamp)+2] << ", " << raw_quats[4*(idet * n_samp + isamp)+3] << std::endl;
                //         }
                //     }
                // }
            }
            return;
        });

}
