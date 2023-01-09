# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ...._libtoast import scan_map_float32 as scan_map_interval_float32_compiled
from ...._libtoast import scan_map_float64 as scan_map_interval_float64_compiled


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
        weights (array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (array, int): The indexes of the weights (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        map_dist (PixelDistribution):
        data_scale (float): unit scalling
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place

    NOTE: Wraps implementations of scan_map to factor the datatype handling
    TODO: implement target offload version of the code and remove the RuntimeError
    """
    # raise an error as the C++ implementation does not implement target_offload yet
    if use_accel:
        raise RuntimeError(
            "scan_map_compiled: there is no GPU-accelerated C++ backend yet."
        )

    # picks the correct implementation as a function of the data-type used
    if mapdata.dtype.char == "d":
        scan_map_interval = scan_map_interval_float64_compiled
    elif mapdata.dtype.char == "f":
        scan_map_interval = scan_map_interval_float32_compiled
    else:
        raise RuntimeError("Projection supports only float32 and float64 binned maps")

    # iterates on detectors and intervals
    n_det = det_data_index.size
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1

            # gets interval pixels
            p_index = pixels_index[idet]
            pixels_interval = pixels[p_index, interval_start:interval_end]
            # gets interval weights
            if weight_index is None:
                weights_interval = np.ones_like(pixels_interval)
            else:
                w_index = weight_index[idet]
                weights_interval = weights[w_index, interval_start:interval_end]

            # Get local submap and pixels
            local_submap, local_pixels = map_dist.global_pixel_to_submap(
                pixels_interval
            )

            # runs the function (and flattens the weights)
            maptod = np.zeros_like(pixels_interval, dtype=det_data.dtype)
            scan_map_interval(
                mapdata.distribution.n_pix_submap,
                nmap,
                local_submap,
                local_pixels,
                mapdata.raw,
                weights_interval.reshape(-1),
                maptod,
            )

            # applies unit scaling
            maptod *= data_scale

            # Add or subtract to det_data, zero out if needed
            # Note that the map scanned timestream will have
            # zeros anywhere that the pointing is bad, but those samples (and
            # any other detector flags) should be handled at other steps of the
            # processing.
            d_index = det_data_index[idet]
            if should_zero:
                det_data[d_index, interval_start:interval_end] = 0.0
            if should_subtract:
                det_data[d_index, interval_start:interval_end] -= maptod
            else:
                det_data[d_index, interval_start:interval_end] += maptod


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_mapmaker_solve", "ops_scan_map")'
