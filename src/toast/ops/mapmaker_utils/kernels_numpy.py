# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ...accelerator import ImplementationType, kernel


@kernel(impl=ImplementationType.NUMPY, name="build_noise_weighted")
def build_noise_weighted_numpy(
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
            shared_check = skip_shared_flags or (
                (shared_flags_samples & shared_flag_mask) == 0
            )
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
            np.add.at(
                zmap,
                (local_submap, isubpix),
                scaled_data[:, np.newaxis] * weights_sample,
            )


@kernel(impl=ImplementationType.NUMPY, name="cov_accum_diag_hits")
def cov_accum_diag_hits_numpy(nsub, nsubpix, nnz, submap, subpix, hits, use_accel):
    """
    Accumulate hit map.
    This uses a pointing matrix to accumulate the local pieces of the hit map.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only locally stored submaps) (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap (size nsamp).
        hits (array, int64):  The local hitmap buffer to accumulate (size ???).
        use_accel (Bool): should we use the accelerator?

    Returns:
        None (result is put in hits).
    """
    nsamp = submap.size
    hits = np.asarray(hits).reshape((-1, nsubpix))

    for i_samp in range(nsamp):
        isubmap = submap[i_samp]
        ipix = subpix[i_samp]
        if (isubmap >= 0) and (ipix >= 0):
            hits[isubmap, ipix] += 1


@kernel(impl=ImplementationType.NUMPY, name="cov_accum_diag_invnpp")
def cov_accum_diag_invnpp_numpy(
    nsub, nsubpix, nnz, submap, subpix, weights, scale, invnpp, use_accel
):
    """
    Accumulate block diagonal noise covariance.
    This uses a pointing matrix to accumulate the local pieces
    of the inverse diagonal pixel covariance.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index
            within the local map (i.e. including only locally stored submaps)
            (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index
            within the submap (size nsamp).
        weights (array, float64):  The pointing matrix weights for each time
            sample and map (shape nw*nnz).
        scale (float):  Optional scaling factor.
        invnpp (array, float64):  The local buffer of diagonal inverse pixel
            covariances, stored as the lower triangle for each pixel (shape ?*nsubpix*block with block=(nnz * (nnz + 1))/2).
        use_accel (Bool): should we use the accelerator?

    Returns:
        None (stores the result in invnpp).
    """
    nsamp = submap.size
    block = (nnz * (nnz + 1)) // 2
    weights = np.asarray(weights).reshape((-1, nnz))
    invnpp = np.asarray(invnpp).reshape((-1, nsubpix, block))

    for i_samp in range(nsamp):
        isubmap = submap[i_samp]
        ipix = subpix[i_samp]
        if (isubmap >= 0) and (ipix >= 0):
            for i_block in range(block):
                # converts i_block back to index into upper triangular matrix of side nnz
                # you can rederive the equations by knowing that i_block = col + row*nnz + row(row+1)/2
                # then assuming col=row (close enough since row <= col < nnz) and rounding down to get row
                row = int(2 * nnz + 1 - np.sqrt((2 * nnz + 1) ** 2 - 8 * i_block)) // 2
                col = i_block + (row * (row + 1)) // 2 - row * nnz
                invnpp[isubmap, ipix, i_block] += (
                    weights[i_samp, col] * weights[i_samp, row] * scale
                )
