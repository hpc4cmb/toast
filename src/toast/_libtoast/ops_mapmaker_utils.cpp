// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#ifdef HAVE_OPENACC
# include <openacc.h>
#endif // ifdef HAVE_OPENACC


void init_ops_mapmaker_utils(py::module & m) {
    m.def(
        "build_noise_weighted", [](
            py::buffer global2local,
            py::buffer zmap,
            int32_t n_global_submap,
            py::buffer pixel_indx,
            py::buffer pixels,
            py::buffer weight_indx,
            py::buffer weights,
            py::buffer data_indx,
            py::buffer det_data,
            py::buffer flag_indx,
            py::buffer det_flags,
            py::buffer det_scale,
            uint8_t det_flag_mask
        ) {
            // NOTE:  shared flags should already be applied to pointing, and any
            // bad samples will have pixels < 0.

            auto info_weightindx = weight_indx.request();
            int32_t * raw_weightindx = reinterpret_cast <int32_t *> (info_weightindx.ptr);

            auto info_pixindx = pixel_indx.request();
            int32_t * raw_pixindx = reinterpret_cast <int32_t *> (info_pixindx.ptr);

            auto info_dataindx = data_indx.request();
            int32_t * raw_dataindx = reinterpret_cast <int32_t *> (info_dataindx.ptr);

            auto info_flagindx = flag_indx.request();
            int32_t * raw_flagindx = reinterpret_cast <int32_t *> (info_flagindx.ptr);

            auto info_zmap = zmap.request();
            double * raw_zmap = reinterpret_cast <double *> (info_zmap.ptr);

            auto info_pixels = pixels.request();
            int64_t * raw_pixels =
                reinterpret_cast <int64_t *> (info_pixels.ptr);

            auto info_weights = weights.request();
            double * raw_weights =
                reinterpret_cast <double *> (info_weights.ptr);

            auto info_detdata = det_data.request();
            double * raw_data = reinterpret_cast <double *> (info_detdata.ptr);

            auto info_flags = det_flags.request();
            uint8_t * raw_flags = reinterpret_cast <uint8_t *> (info_flags.ptr);

            auto info_detscale = det_scale.request();
            double * raw_scale = reinterpret_cast <double *> (info_detscale.ptr);

            auto info_glob2loc = global2local.request();
            int64_t * raw_glob2loc =
                reinterpret_cast <int64_t *> (info_glob2loc.ptr);

            if (info_glob2loc.shape[0] != n_global_submap) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "global2local mapping does not have a value for every submap";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            size_t n_det = info_pixindx.shape[0];
            if (info_weightindx.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "pixel and weight indices do not have same number of dets";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_dataindx.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "pixel and data indices do not have same number of dets";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_flagindx.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "pixel and flag indices do not have same number of dets";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            if (info_detdata.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det data has fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t n_samp = info_detdata.shape[1];

            if (info_detscale.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det scale does not have same number of detectors as det_data";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_pixels.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "pixels has fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_pixels.shape[1] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "pixels do not have same number of samples as det_data";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_flags.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det_flags has fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_flags.shape[1] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det_flags do not have same number of samples as pixels";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_weights.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "weights has fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_weights.shape[1] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "weights do not have same number of samples as pixels";
                log.error(o.str().c_str());
                throw std::runtime_error(
                          o.str().c_str());
            }
            int64_t n_weight;
            if (info_weights.ndim == 2) {
                n_weight = 1;
            } else {
                n_weight = info_weights.shape[2];
            }
            int64_t n_local_submap = info_zmap.shape[0];
            int64_t n_pix_submap = info_zmap.shape[1];
            int64_t n_value_pix = info_zmap.shape[2];

            size_t len_pixels = info_pixels.shape[0] * n_samp;
            size_t len_weights = info_weights.shape[0] * n_samp * n_weight;
            size_t len_flags = info_flags.shape[0] * n_samp;
            size_t len_data = info_detdata.shape[0] * n_samp;
            size_t len_zmap = n_local_submap * n_pix_submap * n_value_pix;

            // std::cout << "zmap pointer " << raw_zmap << " with dims " << n_local_submap << " x " << n_pix_submap << " x " << n_value_pix << std::endl;

            #pragma \
            acc data copyin(det_flag_mask, n_det, n_samp, n_weight, n_local_submap, n_pix_submap, n_value_pix, raw_scale[:n_det], raw_pixindx[:n_det], raw_weightindx[:n_det], raw_flagindx[:n_det], raw_dataindx[:n_det], raw_glob2loc[:n_global_submap]) present(raw_pixels[:len_pixels], raw_weights[:len_weights], raw_flags[:len_flags], raw_data[:len_data], raw_zmap[:len_zmap])
            {
                if (fake_openacc()) {
                    // Set all "present" data to point at the fake device pointers
                    auto & fake = FakeMemPool::get();
                    raw_pixels = (int64_t *)fake.device_ptr(raw_pixels);
                    raw_weights = (double *)fake.device_ptr(raw_weights);
                    raw_flags = (uint8_t *)fake.device_ptr(raw_flags);
                    raw_data = (double *)fake.device_ptr(raw_data);
                    raw_zmap = (double *)fake.device_ptr(raw_zmap);
                }
                #pragma acc parallel
                #pragma acc loop independent
                for (size_t idet = 0; idet < n_det; idet++) {
                    int32_t p_indx = raw_pixindx[idet];
                    int32_t w_indx = raw_weightindx[idet];
                    int32_t d_indx = raw_dataindx[idet];
                    int32_t f_indx = raw_flagindx[idet];
                    double npix_submap_inv = 1.0 / (double)(n_pix_submap);

                    //#pragma acc loop independent
                    for (size_t isamp = 0; isamp < n_samp; isamp++) {
                        size_t off_p = p_indx * n_samp + isamp;
                        size_t off_w = w_indx * n_samp + isamp;
                        size_t off_d = d_indx * n_samp + isamp;
                        size_t off_f = f_indx * n_samp + isamp;
                        int64_t isubpix;
                        int64_t zoff;
                        int64_t off_wt;
                        double scaled_data;
                        int64_t local_submap;
                        int64_t global_submap;

                        // std::cout << "DBG samp " << isamp << ": indx " << p_indx << ", " << w_indx << ", " << d_indx << ", " << f_indx << ":  off " << off_p << ", " << off_w << ", " << off_d << ", " << off_f << std::endl;

                        // off_wt = n_weight * off_w;
                        // std::cout << "         pix " << raw_pixels[off_p] << std::endl;

                        // std::cout << "         weight (" << raw_weights[off_wt + 0] << " " << raw_weights[off_wt + 1] << " " << raw_weights[off_wt + 2] << ")" << std::endl;

                        // std::cout << "         data " << raw_data[off_d] << std::endl;

                        // std::cout << "         flags " << (int)raw_flags[off_f] << std::endl;

                        if (
                            (raw_pixels[off_p] >= 0) &&
                            ((raw_flags[off_f] & det_flag_mask) == 0)
                            ) {
                            // Good data, accumulate
                            global_submap = (int64_t)(raw_pixels[off_p] * npix_submap_inv);

                            // std::cout << "         global submap " << global_submap << std::endl;

                            local_submap = raw_glob2loc[global_submap];
                            // std::cout << "         local submap " << local_submap << std::endl;

                            isubpix = raw_pixels[off_p] - global_submap * n_pix_submap;
                            zoff = n_value_pix * (local_submap * n_pix_submap + isubpix);
                            // std::cout << "         isubpix " << isubpix << std::endl;
                            // std::cout << "         zoff " << zoff << std::endl;

                            off_wt = n_weight * off_w;

                            scaled_data = raw_data[off_d] * raw_scale[idet];
                            for (int64_t iweight = 0; iweight < n_weight; iweight++) {
                                raw_zmap[zoff + iweight] += scaled_data *
                                                            raw_weights[off_wt + iweight];
                                // std::cout << "         z[" << zoff << "+" << iweight << "] = " << raw_zmap[zoff + iweight] << std::endl;
                            }
                        }
                    }
                }
            }
            return;
        });

}
