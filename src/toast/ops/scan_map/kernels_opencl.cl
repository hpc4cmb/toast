// Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Kernels


// FIXME: macro-fy this across all types

__kernel void scan_map_d_to_d(
    int n_det,
    long n_sample,
    long first_sample,
    __global int const * pixels_index,
    __global long const * pixels,
    __global int const * weight_index,
    __global double const * weights,
    __global int const * det_data_index,
    __global double * det_data,
    __global double const * mapdata,
    __global long const * global2local,
    long nnz,
    long npix_submap,
    double data_scale,
    unsigned char should_zero,
    unsigned char should_subtract,
    unsigned char should_scale
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    int d_indx = det_data_index[idet];
    int p_indx = pixels_index[idet];
    int w_indx = weight_index[idet];

    size_t woff = nnz * (w_indx * n_sample + isamp);
    size_t poff = p_indx * n_sample + isamp;
    size_t doff = d_indx * n_sample + isamp;

    long global_submap;
    long local_submap_pix;
    long local_submap;
    long local_pix;

    double tod_val = 0.0;

    if (pixels[poff] >= 0) {
        global_submap = (long)(pixels[poff] / npix_submap);
        local_submap_pix = pixels[poff] - global_submap * npix_submap;
        local_submap = global2local[global_submap];
        local_pix = local_submap * npix_submap + local_submap_pix;

        for (long i = 0; i < nnz; i++) {
            tod_val += weights[woff + i] * mapdata[nnz * local_pix + i];
        }
        tod_val *= data_scale;

        if (should_zero) {
            det_data[doff] = 0;
        }
        if (should_subtract) {
            det_data[doff] -= tod_val;
        } else if (should_scale) {
            det_data[doff] *= tod_val;
        } else {
            det_data[doff] += tod_val;
        }
    }
    return;
}



