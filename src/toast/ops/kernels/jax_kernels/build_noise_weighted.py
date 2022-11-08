# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from ....utils import Logger

from ....jax.mutableArray import MutableJaxArray
from ....jax.intervals import INTERVALS_JAX, JaxIntervals, ALL


def build_noise_weighted_interval(
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
    shared_flags,
    shared_flag_mask,
    interval_starts,
    interval_ends,
    intervals_max_length,
):
    """
    Process all the intervals as a block.

    Args:
        global2local (array, int): size n_global_submap
        zmap (array, double): size n_local_submap*n_pix_submap*nnz
        pixel_index (array, int): size n_det
        pixels (array, int): size n_det*n_samp
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, double): The flat packed detectors weights for the specified mode (size n_det*n_samp*nnz)
        data_index (array, int): size n_det
        det_data (array, double): size n_det*n_samp
        flag_index (array, int): size n_det
        det_flags (array, uint8): size n_det*n_samp
        det_scale (array, double): size n_det
        det_flag_mask (uint8)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        zmap
    """
    # debugging information
    log = Logger.get()
    log.debug(f"build_noise_weighted: jit-compiling.")

    # should we use flags?
    n_samp = pixels.shape[1]
    use_det_flags = det_flags.shape[1] == n_samp
    use_shared_flags = shared_flags.size == n_samp

    # extract interval slices
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive
    pixels_interval = JaxIntervals.get(
        pixels, (pixel_index, intervals)
    )  # pixels[pixel_index,intervals]
    weights_interval = JaxIntervals.get(
        weights, (weight_index, intervals, ALL)
    )  # weights[weight_index,intervals,:]
    data_interval = JaxIntervals.get(
        det_data, (data_index, intervals)
    )  # det_data[data_index,intervals]

    # setup det check
    if use_det_flags:
        det_flags_interval = JaxIntervals.get(
            det_flags, (flag_index, intervals)
        )  # det_flags[flag_index,intervals]
        det_check = (det_flags_interval & det_flag_mask) == 0
    else:
        det_check = True
    # setup shared check
    if use_shared_flags:
        shared_flags_interval = JaxIntervals.get(
            shared_flags, intervals
        )  # shared_flags[intervals]
        shared_check = (shared_flags_interval & shared_flag_mask) == 0
    else:
        shared_check = True
    # mask to identify valid samples
    valid_samples = (pixels_interval >= 0) & det_check & shared_check

    # computes the update to add to zmap
    scaled_data = data_interval * det_scale[:, jnp.newaxis, jnp.newaxis]
    update = jnp.where(
        valid_samples[:, :, :, jnp.newaxis],  # if
        weights_interval * scaled_data[:, :, :, jnp.newaxis],  # then
        0.0,
    )  # else

    # computes the index in zmap
    n_pix_submap = zmap.shape[1]
    global_submap = pixels_interval // n_pix_submap
    local_submap = global2local[global_submap]
    isubpix = pixels_interval - global_submap * n_pix_submap

    # masks padded value before applying the update
    update_masked = jnp.where(
        intervals.mask[jnp.newaxis, :, :, jnp.newaxis],
        zmap[local_submap, isubpix, :],
        update,
    )

    # updates zmap and returns
    zmap = zmap.at[local_submap, isubpix, :].add(update_masked)
    return zmap


# jit compiling
build_noise_weighted_interval = jax.jit(
    build_noise_weighted_interval,
    static_argnames=["det_flag_mask", "shared_flag_mask", "intervals_max_length"],
    donate_argnums=[1],
)  # donate zmap


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
        pixels (array, int): size n_det*n_samp
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, double): The flat packed detectors weights for the specified mode (size n_det*n_samp*nnz)
        data_index (array, int): size n_det
        det_data (array, double): size n_det*n_samp
        flag_index (array, int): size n_det
        det_flags (array, uint8): size n_det*n_samp
        det_scale (array, double): size n_det
        det_flag_mask (uint8)
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        use_accel (Bool): should we use the accelerator?

    Returns:
        None (the result is put in zmap).
    """
    # prepares inputs
    if intervals.size == 0:
        return  # deals with a corner case in tests
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    zmap_input = MutableJaxArray.to_array(zmap)
    global2local = MutableJaxArray.to_array(global2local)
    pixels = MutableJaxArray.to_array(pixels)
    pixel_index = MutableJaxArray.to_array(pixel_index)
    weights = MutableJaxArray.to_array(weights)
    weight_index = MutableJaxArray.to_array(weight_index)
    det_data = MutableJaxArray.to_array(det_data)
    data_index = MutableJaxArray.to_array(data_index)
    det_flags = MutableJaxArray.to_array(det_flags)
    flag_index = MutableJaxArray.to_array(flag_index)
    det_scale = MutableJaxArray.to_array(det_scale)
    shared_flags = MutableJaxArray.to_array(shared_flags)

    # runs computation
    zmap[:] = build_noise_weighted_interval(
        global2local,
        zmap_input,
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
        shared_flags,
        shared_flag_mask,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_sim_tod_conviqt", "ops_mapmaker_utils", "ops_mapmaker_binning", "ops_sim_tod_dipole", "ops_demodulate");'
