// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <accelerator.hpp>

#include <intervals.hpp>

#ifdef HAVE_OPENMP_TARGET
# pragma omp declare target
#endif // ifdef HAVE_OPENMP_TARGET

void build_noise_weighted_inner(
    int32_t const * pixel_index,
    int32_t const * weight_index,
    int32_t const * flag_index,
    int32_t const * data_index,
    int64_t const * global2local,
    double const * data,
    uint8_t const * det_flags,
    uint8_t const * shared_flags,
    int64_t const * pixels,
    double const * weights,
    double const * det_scale,
    double * zmap_val,
    int64_t * zoff,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    int64_t nnz,
    uint8_t det_mask,
    uint8_t shared_mask,
    int64_t n_pix_submap,
    bool use_shared_flags,
    bool use_det_flags
) {
    int32_t w_indx = weight_index[idet];
    int32_t p_indx = pixel_index[idet];
    int32_t f_indx = flag_index[idet];
    int32_t d_indx = data_index[idet];

    int64_t off_p = p_indx * n_samp + isamp;
    int64_t off_w = w_indx * n_samp + isamp;
    int64_t off_d = d_indx * n_samp + isamp;
    int64_t off_f = f_indx * n_samp + isamp;
    int64_t isubpix;
    int64_t off_wt;
    double scaled_data;
    int64_t local_submap;
    int64_t global_submap;

    uint8_t det_check = 0;
    if (use_det_flags) {
        det_check = det_flags[off_f] & det_mask;
    }
    uint8_t shared_check = 0;
    if (use_shared_flags) {
        shared_check = shared_flags[isamp] & shared_mask;
    }

    if (
        (pixels[off_p] >= 0) &&
        (det_check == 0) &&
        (shared_check == 0)
    ) {
        // Good data, accumulate
        global_submap = (int64_t)(pixels[off_p] / n_pix_submap);

        local_submap = global2local[global_submap];

        isubpix = pixels[off_p] - global_submap * n_pix_submap;
        (*zoff) = nnz * (local_submap * n_pix_submap + isubpix);

        off_wt = nnz * off_w;

        scaled_data = data[off_d] * det_scale[idet];

        for (int64_t iweight = 0; iweight < nnz; iweight++) {
            zmap_val[iweight] = scaled_data * weights[off_wt + iweight];
        }
    } else {
        (*zoff) = -1;
        for (int64_t iweight = 0; iweight < nnz; iweight++) {
            zmap_val[iweight] = 0.0;
        }
    }
    return;
}

#ifdef HAVE_OPENMP_TARGET
# pragma omp end declare target
#endif // ifdef HAVE_OPENMP_TARGET

void init_ops_mapmaker_utils(py::module & m) {
    m.def(
        "build_noise_weighted", [](
            py::buffer global2local,
            py::buffer zmap,
            py::buffer pixel_index,
            py::buffer pixels,
            py::buffer weight_index,
            py::buffer weights,
            py::buffer data_index,
            py::buffer det_data,
            py::buffer flag_index,
            py::buffer det_flags,
            py::buffer det_scale,
            uint8_t det_flag_mask,
            py::buffer intervals,
            py::buffer shared_flags,
            uint8_t shared_flag_mask,
            bool use_accel
        ) {
            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_pixel_index = extract_buffer <int32_t> (
                pixel_index, "pixel_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            int64_t * raw_pixels = extract_buffer <int64_t> (
                pixels, "pixels", 2, temp_shape, {-1, -1}
            );
            int64_t n_samp = temp_shape[1];

            int32_t * raw_weight_index = extract_buffer <int32_t> (
                weight_index, "weight_index", 1, temp_shape, {n_det}
            );

            double * raw_weights = extract_buffer <double> (
                weights, "weights", 3, temp_shape, {-1, n_samp, -1}
            );
            int64_t nnz = temp_shape[2];

            int32_t * raw_data_index = extract_buffer <int32_t> (
                data_index, "data_index", 1, temp_shape, {n_det}
            );
            double * raw_det_data = extract_buffer <double> (
                det_data, "det_data", 2, temp_shape, {-1, n_samp}
            );
            int32_t * raw_flag_index = extract_buffer <int32_t> (
                flag_index, "flag_index", 1, temp_shape, {n_det}
            );

            double * raw_det_scale = extract_buffer <double> (
                det_scale, "det_scale", 1, temp_shape, {n_det}
            );

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            int64_t * raw_global2local = extract_buffer <int64_t> (
                global2local, "global2local", 1, temp_shape, {-1}
            );
            int64_t n_global_submap = temp_shape[0];

            double * raw_zmap = extract_buffer <double> (
                zmap, "zmap", 3, temp_shape, {-1, -1, nnz}
            );
            int64_t n_local_submap = temp_shape[0];
            int64_t n_pix_submap = temp_shape[1];

            // Optionally use flags

            bool use_shared_flags = true;
            uint8_t * raw_shared_flags = extract_buffer <uint8_t> (
                shared_flags, "flags", 1, temp_shape, {-1}
            );
            if (temp_shape[0] != n_samp) {
                raw_shared_flags = (uint8_t *)omgr.null;
                use_shared_flags = false;
            }

            bool use_det_flags = true;
            uint8_t * raw_det_flags = extract_buffer <uint8_t> (
                det_flags, "det_flags", 2, temp_shape, {-1, -1}
            );
            if (temp_shape[1] != n_samp) {
                raw_det_flags = (uint8_t *)omgr.null;
                use_det_flags = false;
            }

            if (offload) {
                #ifdef HAVE_OPENMP_TARGET

                int64_t * dev_pixels = omgr.device_ptr(raw_pixels);
                double * dev_weights = omgr.device_ptr(raw_weights);
                double * dev_det_data = omgr.device_ptr(raw_det_data);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                double * dev_zmap = omgr.device_ptr(raw_zmap);
                uint8_t * dev_shared_flags = omgr.device_ptr(raw_shared_flags);
                uint8_t * dev_det_flags = omgr.device_ptr(raw_det_flags);

                # pragma omp target data             \
                device(dev)                          \
                map(to:                              \
                raw_weight_index[0:n_det],           \
                raw_pixel_index[0:n_det],            \
                raw_flag_index[0:n_det],             \
                raw_data_index[0:n_det],             \
                raw_det_scale[0:n_det],              \
                raw_global2local[0:n_global_submap], \
                n_view,                              \
                n_det,                               \
                n_samp,                              \
                nnz,                                 \
                n_pix_submap,                        \
                det_flag_mask,                       \
                shared_flag_mask,                    \
                use_shared_flags,                    \
                use_det_flags                        \
                )                                    \
                use_device_ptr(                      \
                raw_weight_index,                    \
                raw_pixel_index,                     \
                raw_flag_index,                      \
                raw_data_index,                      \
                raw_det_scale,                       \
                raw_global2local                     \
                )
                {
                    # pragma omp target teams distribute collapse(2) \
                        is_device_ptr( \
                            dev_pixels,                          \
                            dev_weights,                         \
                            dev_det_data,                        \
                            dev_det_flags,                       \
                            dev_intervals,                       \
                            dev_shared_flags,                    \
                            dev_zmap \
                        )
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
                            # pragma omp parallel for default(shared)
                            for (
                                int64_t isamp = dev_intervals[iview].first;
                                isamp <= dev_intervals[iview].last;
                                isamp++
                            ) {
                                double zmap_val[nnz];
                                int64_t zoff;
                                build_noise_weighted_inner(
                                    raw_pixel_index,
                                    raw_weight_index,
                                    raw_flag_index,
                                    raw_data_index,
                                    raw_global2local,
                                    dev_det_data,
                                    dev_det_flags,
                                    dev_shared_flags,
                                    dev_pixels,
                                    dev_weights,
                                    raw_det_scale,
                                    zmap_val,
                                    &zoff,
                                    isamp,
                                    n_samp,
                                    idet,
                                    nnz,
                                    det_flag_mask,
                                    shared_flag_mask,
                                    n_pix_submap,
                                    use_shared_flags,
                                    use_det_flags
                                );
                                for (int64_t iw = 0; iw < nnz; iw++) {
                                    # pragma omp atomic
                                    dev_zmap[zoff + iw] += zmap_val[iw];
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
                            double zmap_val[nnz];
                            int64_t zoff;
                            build_noise_weighted_inner(
                                raw_pixel_index,
                                raw_weight_index,
                                raw_flag_index,
                                raw_data_index,
                                raw_global2local,
                                raw_det_data,
                                raw_det_flags,
                                raw_shared_flags,
                                raw_pixels,
                                raw_weights,
                                raw_det_scale,
                                zmap_val,
                                &zoff,
                                isamp,
                                n_samp,
                                idet,
                                nnz,
                                det_flag_mask,
                                shared_flag_mask,
                                n_pix_submap,
                                use_shared_flags,
                                use_det_flags
                            );
                            #pragma omp critical
                            {
                                for (int64_t iw = 0; iw < nnz; iw++) {
                                    raw_zmap[zoff + iw] += zmap_val[iw];
                                }
                            }
                        }
                    }
                }
            }
            return;
        });
}
