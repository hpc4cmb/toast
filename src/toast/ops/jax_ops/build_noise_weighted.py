# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import select_implementation, ImplementationType

#-------------------------------------------------------------------------------------------------
# JAX

def build_noise_weighted_single_sample_jax(zmap, pixel, weights, data, det_flag, det_scale, det_mask, shared_flag, shared_mask):
    """
    The update to be added to zmap.

    Args:
        zmap (array, double): size nnz
        pixel (double)
        weights (array, double): size nnz
        data (double)
        det_flag (uint8)
        det_scale (double)
        det_mask (uint8)
        shared_flag (uint8)
        shared_mask (uint8)

    Returns:
        zmap (array, double): size nnz
    """
    new_zmap = jnp.where((pixel >= 0) and ((det_flag & det_mask) == 0) and ((shared_flag & shared_mask) == 0), # if
                         zmap + (data * det_scale * weights), # then
                         zmap) # else
    return new_zmap

# vmap over samples
build_noise_weighted_single_detector_jax = jax.vmap(build_noise_weighted_single_sample_jax, in_axes=(0,0,0,0,0,None,None,0,None), out_axes=0)
# vmap over detectors
build_noise_weighted_interval_jax = jax.vmap(build_noise_weighted_single_detector_jax, in_axes=(0,0,0,0,0,0,None,None,None), out_axes=0)

def build_noise_weighted_unjitted_jax(zmap, pixels, weights, det_data, det_flags, det_scale, det_flag_mask, shared_flags, shared_flag_mask):
    """
    Process a full interval at once.
    NOTE: this function was added for debugging purposes, one could replace it with `build_noise_weighted_interval_jax`

    Args:
        zmap (array, double): size n_det*n_samp_interval*nnz
        pixels (array, double): size n_det*n_samp_interval
        weights (array, double): The flat packed detectors weights for the specified mode (size n_det*n_samp_interval*nnz)
        det_data (array, double): size n_det*n_samp_interval
        det_flags (array, uint8): size n_det*n_samp_interval
        det_scale (array, double): size n_det
        det_flag_mask (uint8)
        shared_flags (array, uint8): size n_samp_interval
        shared_flag_mask (uint8)

    Returns:
        zmap (array, double): size n_det*n_samp_interval*nnz
    """
    # display sizes
    print(f"DEBUG: jit compiling 'build_noise_weighted_interval_jax' with zmap_shape:{zmap.shape} n_det:{det_scale.size} n_samp_interval:{shared_flags.size} det_mask:{det_flag_mask} shared_flag_mask:{shared_flag_mask}")
    # does the computation
    return build_noise_weighted_interval_jax(zmap, pixels, weights, det_data, det_flags, det_scale, det_flag_mask, shared_flags, shared_flag_mask)

# jit compiling
build_noise_weighted_jitted_jax = jax.jit(build_noise_weighted_unjitted_jax, static_argnames=['det_flag_mask','shared_flag_mask'])

def build_noise_weighted_jax(global2local, zmap, pixel_index, pixels, weight_index, weights, data_index, det_data, flag_index, det_flags, det_scale, det_flag_mask, intervals, shared_flags, shared_flag_mask):
    """
    Args:
        global2local (array, int): size n_global_submap
        zmap (array, double): size n_local_submap*n_pix_submap*nnz
        pixel_index (array, double): size n_det
        pixels (array, double): size ???*n_samp
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, double): The flat packed detectors weights for the specified mode (size ???*n_samp*nnz)
        data_index (array, int): size n_det
        det_data (array, double): size ???*n_samp
        flag_index (array, int): size n_det
        det_flags (array, uint8): size ???*n_samp
        det_scale (array, double): size n_det
        det_flag_mask (uint8)
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)

    Returns:
        None (the result is put in zmap).
    """
    # precompute the section of zmap that will be worked on
    n_pix_submap = zmap.shape[1]
    npix_submap_inv = 1.0 / n_pix_submap
    pixels_indexed = pixels[pixel_index,:]
    i_global_submap = (pixels_indexed * npix_submap_inv).astype(int)
    i_local_submap = global2local[i_global_submap]
    i_pix_submap = pixels_indexed - i_global_submap * n_pix_submap
    subzmap = zmap[i_local_submap, i_pix_submap, :]

    # we loop over intervals
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        # extract interval slices
        pixels_interval = pixels_indexed[:, interval_start:interval_end]
        weights_interval = weights[weight_index, interval_start:interval_end, :]
        data_interval = det_data[data_index, interval_start:interval_end]
        det_flags_interval = det_flags[flag_index, interval_start:interval_end]
        shared_flags_interval = shared_flags[interval_start:interval_end]
        subzmap_interval = subzmap[:, interval_start:interval_end, :]
        # process the interval then updates zmap in place
        subzmap_interval[:] = build_noise_weighted_jitted_jax(subzmap_interval, pixels_interval, weights_interval, data_interval, det_flags_interval, det_scale, det_flag_mask, shared_flags_interval, shared_flag_mask)

#-------------------------------------------------------------------------------------------------
# NUMPY

def build_noise_weighted_inner_numpy(global2local, data, det_flag, shared_flag, pixel, weights, det_scale, zmap,
                                     det_mask, shared_mask, npix_submap_inv):
    """
    Args:
        global2local (array, int): size n_global_submap
        data (double)
        det_flag (uint8)
        shared_flag (uint8)
        pixels(double)
        weights (array, double): The flat packed detectors weights for the specified mode (nnz)
        det_scale (double)
        zmap (array, double): size n_local_submap*n_pix_submap*nnz
        det_mask (uint8)
        shared_mask (uint8)
        npix_submap_inv (double)

    Returns:
        None (the result is put in zmap).
    """
    if ( (pixel >= 0) and ((det_flag & det_mask) == 0) and ((shared_flag & shared_mask) == 0) ): 
        # Good data, accumulate
        scaled_data = data * det_scale
        # computes the index in zmap
        global_submap = int(pixel * npix_submap_inv)
        local_submap = global2local[global_submap]
        n_pix_submap = zmap.shape[1]
        isubpix = pixel - global_submap * n_pix_submap
        # accumulates
        zmap[local_submap, isubpix, :] += scaled_data * weights

def build_noise_weighted_numpy(global2local, zmap, pixel_index, pixels, weight_index, weights, data_index, det_data, flag_index, det_flags, det_scale, det_flag_mask, intervals, shared_flags, shared_flag_mask):
    """
    Args:
        global2local (array, int): size n_global_submap
        zmap (array, double): size n_local_submap*n_pix_submap*nnz
        pixel_index (array, double): size n_det
        pixels (array, double): size ???*n_samp
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, double): The flat packed detectors weights for the specified mode (size ???*n_samp*nnz)
        data_index (array, int): size n_det
        det_data (array, double): size ???*n_samp
        flag_index (array, int): size n_det
        det_flags (array, uint8): size ???*n_samp
        det_scale (array, double): size n_det
        det_flag_mask (uint8)
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)

    Returns:
        None (the result is put in zmap).
    """
    # problem size
    n_det = data_index.size
    (n_local_submap,n_pix_submap,nnz) = zmap.shape
    print(f"DEBUG: running 'build_noise_weighted_numpy' with n_view:{intervals.size} n_det:{n_det} n_samp:{shared_flags.size} n_local_submap:{n_local_submap} n_pix_submap:{n_pix_submap} nnz:{nnz} det_flag_mask:{det_flag_mask} shared_flag_mask:{shared_flag_mask}")

    npix_submap_inv = 1.0 / n_pix_submap

    # iterates on detectors and intervals
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval['first']
            interval_end = interval['last']
            for isamp in range(interval_start,interval_end+1):
                p_index = pixel_index[idet]
                w_index = weight_index[idet]
                f_index = flag_index[idet]
                d_index = data_index[idet]
                build_noise_weighted_inner_numpy(
                    global2local,
                    det_data[d_index,isamp],
                    det_flags[f_index,isamp],
                    shared_flags[isamp],
                    pixels[p_index,isamp],
                    weights[w_index,isamp,:],
                    det_scale[idet],
                    zmap,
                    det_flag_mask,
                    shared_flag_mask,
                    npix_submap_inv)

#-------------------------------------------------------------------------------------------------
# C++

"""
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
    double * zmap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    int64_t nnz,
    uint8_t det_mask,
    uint8_t shared_mask,
    int64_t n_pix_submap,
    double npix_submap_inv
) 
{
    int32_t w_indx = weight_index[idet];
    int32_t p_indx = pixel_index[idet];
    int32_t f_indx = flag_index[idet];
    int32_t d_indx = data_index[idet];

    int64_t off_p = p_indx * n_samp + isamp;
    int64_t off_w = w_indx * n_samp + isamp;
    int64_t off_d = d_indx * n_samp + isamp;
    int64_t off_f = f_indx * n_samp + isamp;
    int64_t isubpix;
    int64_t zoff;
    int64_t off_wt;
    double scaled_data;
    int64_t local_submap;
    int64_t global_submap;

    if (
        (pixels[off_p] >= 0) &&
        ((det_flags[off_f] & det_mask) == 0) &&
        ((shared_flags[off_p] & shared_mask) == 0)
    ) 
    {
        // Good data, accumulate
        global_submap = (int64_t)(pixels[off_p] * npix_submap_inv);

        local_submap = global2local[global_submap];

        isubpix = pixels[off_p] - global_submap * n_pix_submap;
        zoff = nnz * (local_submap * n_pix_submap + isubpix);

        off_wt = nnz * off_w;

        scaled_data = data[off_d] * det_scale[idet];

        for (int64_t iweight = 0; iweight < nnz; iweight++) 
        {
            zmap[zoff + iweight] += scaled_data * weights[off_wt + iweight];
        }
    }
}

void build_noise_weighted(
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
) 
{
    // This is used to return the actual shape of each buffer
    std::vector <int64_t> temp_shape(3);

    int32_t * raw_pixel_index = extract_buffer <int32_t> (pixel_index, "pixel_index", 1, temp_shape, {-1});
    int64_t n_det = temp_shape[0];

    int64_t * raw_pixels = extract_buffer <int64_t> (pixels, "pixels", 2, temp_shape, {-1, -1});
    int64_t n_samp = temp_shape[1];

    int32_t * raw_weight_index = extract_buffer <int32_t> (weight_index, "weight_index", 1, temp_shape, {n_det});

    double * raw_weights = extract_buffer <double> (weights, "weights", 3, temp_shape, {-1, n_samp, -1});
    int64_t nnz = temp_shape[2];

    int32_t * raw_data_index = extract_buffer <int32_t> (data_index, "data_index", 1, temp_shape, {n_det});
    double * raw_det_data = extract_buffer <double> (det_data, "det_data", 2, temp_shape, {-1, n_samp});
    int32_t * raw_flag_index = extract_buffer <int32_t> (flag_index, "flag_index", 1, temp_shape, {n_det});
    uint8_t * raw_det_flags = extract_buffer <uint8_t> (det_flags, "det_flags", 2, temp_shape, {-1, n_samp});

    double * raw_det_scale = extract_buffer <double> (det_scale, "det_scale", 1, temp_shape, {n_det});

    uint8_t * raw_shared_flags = extract_buffer <uint8_t> (shared_flags, "flags", 1, temp_shape, {n_samp});

    Interval * raw_intervals = extract_buffer <Interval> (intervals, "intervals", 1, temp_shape, {-1});
    int64_t n_view = temp_shape[0];

    int64_t * raw_global2local = extract_buffer <int64_t> (global2local, "global2local", 1, temp_shape, {-1});
    int64_t n_global_submap = temp_shape[0];

    double * raw_zmap = extract_buffer <double> (zmap, "zmap", 3, temp_shape, {-1, -1, nnz});
    int64_t n_local_submap = temp_shape[0];
    int64_t n_pix_submap = temp_shape[1];

    int64_t * dev_pixels = raw_pixels;
    double * dev_weights = raw_weights;
    double * dev_det_data = raw_det_data;
    uint8_t * dev_det_flags = raw_det_flags;
    Interval * dev_intervals = raw_intervals;
    uint8_t * dev_shared_flags = raw_shared_flags;
    double * dev_zmap = raw_zmap;

    double npix_submap_inv = 1.0 / (double)(n_pix_submap);

    for (int64_t idet = 0; idet < n_det; idet++) 
    {
        for (int64_t iview = 0; iview < n_view; iview++) 
        {
            #pragma omp parallel for
            for (int64_t isamp = dev_intervals[iview]['first']; isamp <= dev_intervals[iview]['last']; isamp++)
            {
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
                    dev_zmap,
                    isamp,
                    n_samp,
                    idet,
                    nnz,
                    det_flag_mask,
                    shared_flag_mask,
                    n_pix_submap,
                    npix_submap_inv
                );
            }
        }
    }
}
"""

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
build_noise_weighted = select_implementation(build_noise_weighted_numpy, 
                                             build_noise_weighted_numpy, 
                                             build_noise_weighted_jax, 
                                             default_implementationType=ImplementationType.NUMPY)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_mapmaker_utils"); toast.tests.run("ops_mapmaker_binning")'

# to bench:
# use scanmap config and check BuildNoiseWeighted field in timing.csv
# one problem: this includes call to other operators
