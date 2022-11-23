# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

# TODO does not pass "ops_pointing_wcs"
def build_noise_weighted_ted(
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
    n_det = data_index.size
    n_pix_submap = zmap.shape[1]

    # iterates on detectors and intervals
    for idet in range(n_det):
        p_index = pixel_index[idet]
        w_index = weight_index[idet]
        f_index = flag_index[idet]
        d_index = data_index[idet]
        for interval in intervals:
            samples = slice(interval.first, interval.last + 1, 1)
            good = np.logical_and(
                ((det_flags[f_index][samples] & det_flag_mask) == 0),
                ((shared_flags[samples] & shared_flag_mask) == 0),
            )
            pixel_buffer = pixels[p_index][samples]
            det_buffer = det_data[d_index][samples]
            weight_buffer = weights[w_index][samples]
            global_submap = pixel_buffer[good] // n_pix_submap
            submap_pix = pixel_buffer[good] - global_submap * n_pix_submap
            local_submap = np.array(
                [global2local[x] for x in global_submap], dtype=np.int64
            )
            tempdata = np.multiply(
                weight_buffer[good],
                np.multiply(det_scale[idet], det_buffer[good])[:, np.newaxis],
            )
            np.add.at(
                zmap,
                (local_submap, submap_pix),
                tempdata,
            )

#-----


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
    skip_det_flags = det_flags.shape[1] != n_samp
    skip_shared_flags = shared_flags.size != n_samp

    # iterates on detectors and intervals
    n_det = data_index.size
    n_pix_submap = zmap.shape[1]
    for idet in range(n_det):
        p_index = pixel_index[idet]
        w_index = weight_index[idet]
        f_index = flag_index[idet]
        d_index = data_index[idet]
        det_scale_det = det_scale[idet]
        for interval in intervals:
            # get interval slices
            interval_start = interval.first
            interval_end = interval.last + 1
            data_samples = det_data[d_index, interval_start:interval_end]
            pixel_samples = pixels[p_index, interval_start:interval_end]
            weights_sample = weights[w_index, interval_start:interval_end, :]
            det_flags_samples = det_flags[f_index, interval_start:interval_end]
            shared_flags_samples = shared_flags[interval_start:interval_end]
            # keeps only good samples
            det_check = skip_det_flags or ((det_flags_samples & det_flag_mask) == 0)
            shared_check = skip_shared_flags or ((shared_flags_samples & shared_flag_mask) == 0)
            good_samples = (pixel_samples >= 0) & (det_check & shared_check)
            data_samples = data_samples[good_samples]
            pixel_samples = pixel_samples[good_samples]
            weights_sample = weights_sample[good_samples]
            # computes the indices in zmap
            global_submap = pixel_samples // n_pix_submap
            local_submap = global2local[global_submap]
            isubpix = pixel_samples - global_submap * n_pix_submap
            # accumulates
            # cannot use += due to duplicated indices, np.add being atomic
            scaled_data = data_samples * det_scale_det
            np.add.at(zmap, (local_submap, isubpix), scaled_data[:, np.newaxis] * weights_sample)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_sim_tod_conviqt"); toast.tests.run("ops_mapmaker_utils"); toast.tests.run("ops_mapmaker_binning"); toast.tests.run("ops_sim_tod_dipole"); toast.tests.run("ops_demodulate"); toast.tests.run("ops_pointing_wcs")'
