# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


def build_noise_weighted_inner(
    global2local,
    data,
    det_flag,
    shared_flag,
    pixel,
    weights,
    det_scale,
    zmap,
    det_mask,
    shared_mask,
):
    """
    Args:
        global2local (array, int): size n_global_submap
        data (double)
        det_flag (uint8): Optional, could be None
        shared_flag (uint8): Optional, could be None
        pixels(int)
        weights (array, double): The flat packed detectors weights for the specified mode (nnz)
        det_scale (double)
        zmap (array, double): size n_local_submap*n_pix_submap*nnz
        det_mask (uint8)
        shared_mask (uint8)

    Returns:
        None (the result is put in zmap).
    """
    det_check = True if (det_flag is None) else ((det_flag & det_mask) == 0)
    shared_check = True if (shared_flag is None) else ((shared_flag & shared_mask) == 0)
    if (pixel >= 0) and (det_check and shared_check):
        # Good data, accumulate
        scaled_data = data * det_scale
        # computes the index in zmap
        n_pix_submap = zmap.shape[1]
        global_submap = pixel // n_pix_submap
        local_submap = global2local[global_submap]
        isubpix = pixel - global_submap * n_pix_submap
        # accumulates
        # cannot use += due to duplicated indices
        np.add.at(zmap, (local_submap, isubpix), scaled_data * weights)


def build_noise_weighted(
    global2local,
    zmap,
    pixel_index,
    pixels,
    weight_index,
    weights,
    data_index,
    det_data,
    flag_index,
    det_flags,
    det_scale,
    det_flag_mask,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel,
):
    """
    Args:
        global2local (array, int): size n_global_submap
        zmap (array, double): size n_local_submap*n_pix_submap*nnz
        pixel_index (array, int): size n_det
        pixels (array, int): size ???*n_samp
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, double): The flat packed detectors weights for the specified mode (size ???*n_samp*nnz)
        data_index (array, int): size n_det
        det_data (array, double): size ???*n_samp
        flag_index (array, int): size n_det
        det_flags (array, uint8): size ???*n_samp or 1*1
        det_scale (array, double): size n_det
        det_flag_mask (uint8)
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        use_accel (Bool): should we use the accelerator?

    Returns:
        None (the result is put in zmap).
    """
    # should we use flags?
    n_samp = pixels.shape[1]
    use_det_flags = det_flags.shape[1] == n_samp
    use_shared_flags = shared_flags.size == n_samp

    # iterates on detectors and intervals
    n_det = data_index.size
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last
            for isamp in range(interval_start, interval_end + 1):
                p_index = pixel_index[idet]
                w_index = weight_index[idet]
                f_index = flag_index[idet]
                d_index = data_index[idet]
                build_noise_weighted_inner(
                    global2local,
                    det_data[d_index, isamp],
                    det_flags[f_index, isamp] if use_det_flags else None,
                    shared_flags[isamp] if use_shared_flags else None,
                    pixels[p_index, isamp],
                    weights[w_index, isamp, :],
                    det_scale[idet],
                    zmap,
                    det_flag_mask,
                    shared_flag_mask,
                )

def _py_build_noise_weighted(
        self,
        zmap,
        pixel_indx,
        pixel_data,
        weight_indx,
        weight_data,
        det_indx,
        det_data,
        flag_indx,
        flag_data,
        flag_mask,
        intr_data,
        shared_flags,
        shared_mask,
        det_scale,
    ):
        """Internal python implementation for comparison tests."""
        global2local = zmap.distribution.global_submap_to_local.array()
        npix_submap = zmap.distribution.n_pix_submap
        nnz = zmap.n_value
        for idet in range(len(det_indx)):
            didx = det_indx[idet]
            pidx = pixel_indx[idet]
            widx = weight_indx[idet]
            fidx = flag_indx[idet]
            for vw in intr_data:
                samples = slice(vw.first, vw.last + 1, 1)
                good = np.logical_and(
                    ((flag_data[fidx][samples] & flag_mask) == 0),
                    ((shared_flags[samples] & shared_mask) == 0),
                )
                pixel_buffer = pixel_data[pidx][samples]
                det_buffer = det_data[didx][samples]
                weight_buffer = weight_data[widx][samples]
                global_submap = pixel_buffer[good] // npix_submap
                submap_pix = pixel_buffer[good] - global_submap * npix_submap
                local_submap = np.array(
                    [global2local[x] for x in global_submap], dtype=np.int64
                )
                tempdata = np.multiply(
                    weight_buffer[good],
                    np.multiply(det_scale[idet], det_buffer[good])[:, np.newaxis],
                )
                np.add.at(
                    zmap.data,
                    (local_submap, submap_pix),
                    tempdata,
                )