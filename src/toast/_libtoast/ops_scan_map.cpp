// Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <accelerator.hpp>

#include <intervals.hpp>

#ifdef HAVE_OPENMP_TARGET
# pragma omp declare target
#endif // ifdef HAVE_OPENMP_TARGET

template <typename T>
void scan_map_inner(
    int32_t const * pixel_index,
    int32_t const * weight_index,
    int32_t const * data_index,
    int64_t const * global2local,
    double * data,
    int64_t const * pixels,
    double const * weights,
    T const * mapdata,
    double data_scale,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    int64_t nnz,
    int64_t n_pix_submap,
    bool should_zero,
    bool should_subtract,
    bool should_scale
) {
    int32_t w_indx = weight_index[idet];
    int32_t p_indx = pixel_index[idet];
    int32_t d_indx = data_index[idet];

    int64_t off_p = p_indx * n_samp + isamp;
    int64_t off_w = w_indx * n_samp + isamp;
    int64_t off_d = d_indx * n_samp + isamp;
    int64_t isubpix;
    int64_t off_wt;
    double tod_val;
    int64_t local_submap;
    int64_t global_submap;
    int64_t map_off;

    if (should_zero) {
        data[off_d] = 0.0;
    }

    if (pixels[off_p] >= 0) {
        // Good data, accumulate
        global_submap = (int64_t)(pixels[off_p] / n_pix_submap);

        local_submap = global2local[global_submap];

        isubpix = pixels[off_p] - global_submap * n_pix_submap;
        map_off = nnz * (local_submap * n_pix_submap + isubpix);

        off_wt = nnz * off_w;

        tod_val = 0.0;
        for (int64_t iweight = 0; iweight < nnz; iweight++) {
            tod_val += weights[off_wt + iweight] * mapdata[map_off + iweight];
        }
        tod_val *= data_scale;

        if (should_subtract) {
            data[off_d] -= tod_val;
        } else if (should_scale) {
            data[off_d] *= tod_val;
        } else {
            data[off_d] += tod_val;
        }
    }
    return;
}

#ifdef HAVE_OPENMP_TARGET
# pragma omp end declare target
#endif // ifdef HAVE_OPENMP_TARGET

template <typename T>
void register_ops_scan_map(py::module & m, char const * name) {
    m.def(name,
          [](
              py::buffer global2local,
              int64_t n_pix_submap,
              py::buffer mapdata,
              py::buffer det_data,
              py::buffer data_index,
              py::buffer pixels,
              py::buffer pixel_index,
              py::buffer weights,
              py::buffer weight_index,
              py::buffer intervals,
              double data_scale,
              bool should_zero,
              bool should_subtract,
              bool should_scale,
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

              // Handle the case of either 2 or 3 dims
              auto winfo = weights.request();
              double * raw_weights;
              int64_t nnz;
              if (winfo.ndim == 2) {
                  nnz = 1;
                  raw_weights = extract_buffer <double> (
                      weights, "weights", 2, temp_shape, {-1, n_samp}
                  );
              } else {
                  raw_weights = extract_buffer <double> (
                      weights, "weights", 3, temp_shape, {-1, n_samp, -1}
                  );
                  nnz = temp_shape[2];
              }

              int32_t * raw_data_index = extract_buffer <int32_t> (
                  data_index, "data_index", 1, temp_shape, {n_det}
              );
              double * raw_det_data = extract_buffer <double> (
                  det_data, "det_data", 2, temp_shape, {-1, n_samp}
              );

              Interval * raw_intervals = extract_buffer <Interval> (
                  intervals, "intervals", 1, temp_shape, {-1}
              );
              int64_t n_view = temp_shape[0];

              int64_t * raw_global2local = extract_buffer <int64_t> (
                  global2local, "global2local", 1, temp_shape, {-1}
              );
              int64_t n_global_submap = temp_shape[0];

              T * raw_mapdata = extract_buffer <T> (
                  mapdata, "mapdata", 3, temp_shape, {-1, n_pix_submap, nnz}
              );
              int64_t n_local_submap = temp_shape[0];

              if (offload) {
                  #ifdef HAVE_OPENMP_TARGET

                  int64_t * dev_pixels = omgr.device_ptr(raw_pixels);
                  double * dev_weights = omgr.device_ptr(raw_weights);
                  double * dev_det_data = omgr.device_ptr(raw_det_data);
                  Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                  T * dev_mapdata = omgr.device_ptr(raw_mapdata);

                  # pragma omp target data             \
                  device(dev)                          \
                  map(to:                              \
                  raw_weight_index[0:n_det],           \
                  raw_pixel_index[0:n_det],            \
                  raw_data_index[0:n_det],             \
                  raw_global2local[0:n_global_submap], \
                  n_view,                              \
                  n_det,                               \
                  n_samp,                              \
                  nnz,                                 \
                  n_pix_submap,                        \
                  data_scale,                          \
                  should_scale,                        \
                  should_subtract,                     \
                  should_zero                          \
                  )
                  {
                      # pragma omp target teams distribute collapse(2) \
                      is_device_ptr(                                   \
                      dev_pixels,                                      \
                      dev_weights,                                     \
                      dev_det_data,                                    \
                      dev_intervals,                                   \
                      dev_mapdata                                      \
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
                                      scan_map_inner <T> (
                                          raw_pixel_index,
                                          raw_weight_index,
                                          raw_data_index,
                                          raw_global2local,
                                          dev_det_data,
                                          dev_pixels,
                                          dev_weights,
                                          dev_mapdata,
                                          data_scale,
                                          isamp,
                                          n_samp,
                                          idet,
                                          nnz,
                                          n_pix_submap,
                                          should_zero,
                                          should_subtract,
                                          should_scale
                                      );
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
                              scan_map_inner <T> (
                                  raw_pixel_index,
                                  raw_weight_index,
                                  raw_data_index,
                                  raw_global2local,
                                  raw_det_data,
                                  raw_pixels,
                                  raw_weights,
                                  raw_mapdata,
                                  data_scale,
                                  isamp,
                                  n_samp,
                                  idet,
                                  nnz,
                                  n_pix_submap,
                                  should_zero,
                                  should_subtract,
                                  should_scale
                              );
                          }
                      }
                  }
              }

              return;
          });
    return;
}

void init_ops_scan_map(py::module & m) {
    register_ops_scan_map <double> (m, "ops_scan_map_float64");
    register_ops_scan_map <float> (m, "ops_scan_map_float32");
    register_ops_scan_map <int64_t> (m, "ops_scan_map_int64");
    register_ops_scan_map <int32_t> (m, "ops_scan_map_int32");
}
