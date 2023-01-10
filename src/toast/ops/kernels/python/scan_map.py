# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


def global_to_local(global_pixels, npix_submap, global2local):
    """
    Convert global pixel indices to local submaps and pixels within the submap.

    Args:
        global_pixels (array):  The global pixel indices (size nsamples).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.

    Returns:
        (tuple of array):  The (local submap, local pixels within submap) for each global pixel (each of size nsamples).

    NOTE: map_dist.global_pixel_to_submap(pixels) => global_to_local(pixels, map_dist._n_pix_submap, map_dist._glob2loc)
    """
    quotient, remainder = np.divmod(global_pixels, npix_submap)
    local_pixels = np.where(global_pixels < 0, -1, remainder)
    local_submaps = np.where(global_pixels < 0, -1, global2local[quotient])
    return (local_submaps, local_pixels)


def scan_map_interval(
    mapdata,
    npix_submap,
    global2local,
    pixels,
    weights,
    det_data,
    data_scale,
    should_zero,
    should_subtract,
):
    """
    Applies scan_map to a given interval.

    Args:
        mapdata (array, ?):  The local piece of the map (size ?*npix_submap*nmap).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.
        pixels (array, int): pixels (size nsample)
        weights (array, float64):  The pointing matrix weights (size: nsample*nmap).
        det_data (array, float64):  The timestream on which to accumulate the map values (size nsample).
        data_scale (float): unit scalling
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data

    Returns:
        det_data
    """
    # Get local submap and pixels
    submap, subpix = global_to_local(pixels, npix_submap, global2local)

    # uses only samples with valid indices
    valid_samples = (subpix >= 0) & (submap >= 0)
    valid_weights = weights[valid_samples]
    valid_submap = submap[valid_samples]
    valid_subpix = subpix[valid_samples]
    valid_mapdata = mapdata[valid_submap, valid_subpix, :]

    # updates det_data
    # Note that the map scanned timestream will have
    # zeros anywhere that the pointing is bad, but those samples (and
    # any other detector flags) should be handled at other steps of the
    # processing.
    if should_zero:
        det_data[:] = 0.0
    if should_subtract:
        det_data[valid_samples] -= (
            np.sum(valid_mapdata * valid_weights, axis=1) * data_scale
        )
    else:
        det_data[valid_samples] += (
            np.sum(valid_mapdata * valid_weights, axis=1) * data_scale
        )
    return det_data


def scan_map(
    mapdata,
    nmap,
    det_data,
    det_data_index,
    pixels,
    pixels_index,
    weights,
    weight_index,
    intervals,
    map_dist,
    data_scale,
    should_zero,
    should_subtract,
    use_accel,
):
    """
    Sample a map into a timestream.

    This uses a distributed map and the pointing matrix
    to generate timestream values.

    Args:
        mapdata (Pixels):  The local piece of the map.
        nmap (int): number of valid pixels
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        pixels (array, int): pixels (size ???*n_samp)
        pixels_index (array, int): The indexes of the pixels (size n_det)
        weights (optional array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (optional array, int): The indexes of the weights (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        map_dist (PixelDistribution): encapsulate information to translate the pixel mapping
        data_scale (float): unit scalling
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    # extracts pixel distribution information
    npix_submap = map_dist._n_pix_submap
    global2local = map_dist._glob2loc.array()
    # turns mapdata into a numpy array of shape ?*npix_submap*nmap
    mapdata = mapdata.raw.array()
    mapdata = np.reshape(mapdata, newshape=(-1, npix_submap, nmap))

    # iterates on detectors and intervals
    n_det = det_data_index.size
    for idet in range(n_det):
        p_index = pixels_index[idet]
        d_index = det_data_index[idet]
        w_index = None if (weight_index is None) else weight_index[idet]
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1
            # gets interval data
            pixels_interval = pixels[p_index, interval_start:interval_end]
            weights_interval = (
                np.ones_like(pixels_interval)
                if (weight_index is None)
                else weights[w_index, interval_start:interval_end]
            )
            det_data_interval = det_data[d_index, interval_start:interval_end]
            # process the interval
            new_det_data_interval = scan_map_interval(
                mapdata,
                npix_submap,
                global2local,
                pixels_interval,
                weights_interval,
                det_data_interval,
                data_scale,
                should_zero,
                should_subtract,
            )
            det_data[d_index, interval_start:interval_end] = new_det_data_interval
