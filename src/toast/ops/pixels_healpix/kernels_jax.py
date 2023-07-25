# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import ALL, INTERVALS_JAX, JaxIntervals
from ...jax.math import healpix, qarray
from ...jax.mutableArray import MutableJaxArray
from ...utils import Logger


def pixels_healpix_inner(hpix, quats, nest):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        hpix (HPIX_JAX): Healpix projection object.
        quats (array, float64): Detector quaternion (size 4).
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        pixels (array, int64): The detector pixel indices to store the result.
    """
    # initialize dir
    dir = qarray.rotate_zaxis(quats)

    # pixel computation
    (phi, region, z, rtz) = healpix.vec2zphi(dir)
    if nest:
        pixel = healpix.zphi2nest(hpix, phi, region, z, rtz)
    else:
        pixel = healpix.zphi2ring(hpix, phi, region, z, rtz)

    return pixel


# maps over samples and detectors
# pixels_healpix_inner = jax_xmap(pixels_healpix_inner,
#                                 in_axes=[[...], # hpix
#                                          ['detectors','intervals','interval_size',...], # quats
#                                          [...]], # nest
#                                 out_axes=['detectors','intervals','interval_size'])
# TODO xmap is commented out for now due to a [bug with static argnum](https://github.com/google/jax/issues/10741)
pixels_healpix_inner = jax.vmap(
    pixels_healpix_inner, in_axes=[None, 0, None], out_axes=0
)  # loop on interval_size
pixels_healpix_inner = jax.vmap(
    pixels_healpix_inner, in_axes=[None, 0, None], out_axes=0
)  # loop on intervals
pixels_healpix_inner = jax.vmap(
    pixels_healpix_inner, in_axes=[None, 0, None], out_axes=0
)  # loop on detectors


def pixels_healpix_interval(
    quat_index,
    quats,
    flags,
    flag_mask,
    pixel_index,
    pixels,
    hit_submaps,
    n_pix_submap,
    nside,
    nest,
    interval_starts,
    interval_ends,
    intervals_max_length,
):
    """
    Process all the intervals as a block.

    Args:
        quat_index (array, int): size n_det
        quats (array, float64): The flat-packed array of detector quaternions (size ???*n_samp*4).
        flags (array, uint8): size n_samp (or you shouldn't use flags)
        flag_mask (uint8): integer used to select flags (not necesarely a boolean)
        pixel_index (array, int): size n_det
        pixels (array, int64): The detector pixel indices to store the result (size n_det*n_samp).
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (int):
        nside (int): Used to build the healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        (pixels, hit_submaps)
    """
    # debugging information
    log = Logger.get()
    log.debug(f"pixels_healpix: jit-compiling.")

    # flag
    n_samp = pixels.shape[1]
    use_flags = (flag_mask != 0) and (flags.size == n_samp)

    # used to extract interval slices
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive

    # computes the pixels and submap
    hpix = healpix.HPIX_JAX(nside)
    quats_interval = JaxIntervals.get(
        quats, (quat_index, intervals, ALL)
    )  # quats[quat_index,intervals,:]
    pixels_interval = pixels_healpix_inner(hpix, quats_interval, nest)
    sub_map = jnp.ravel(
        pixels_interval // n_pix_submap
    )  # flattened to index into the 1D hit_submaps
    previous_hit_submaps_unflattened = jnp.reshape(
        hit_submaps[sub_map], newshape=pixels_interval.shape
    )  # unflattened to apply a 2D mask

    # applies the flags
    if use_flags:
        # we pad with 1 such that values out of the interval will be flagged
        flags_interval = JaxIntervals.get(
            flags, intervals, padding_value=1
        )  # flags[intervals]
        is_flagged = (flags_interval & flag_mask) != 0
        pixels_interval = jnp.where(is_flagged, -1, pixels_interval)
        #
        new_hit_submap_unflattened = jnp.where(
            is_flagged, previous_hit_submaps_unflattened, 1
        )
    else:
        # masks the padded values in sub_map
        new_hit_submap_unflattened = jnp.where(
            intervals.mask, previous_hit_submaps_unflattened, 1
        )
    new_hit_submap = jnp.ravel(new_hit_submap_unflattened)

    # updates results and returns
    hit_submaps = hit_submaps.at[sub_map].set(new_hit_submap)
    pixels = JaxIntervals.set(
        pixels, (pixel_index, intervals), pixels_interval
    )  # pixels[pixel_index,intervals] = pixels_interval
    return pixels, hit_submaps


# jit compiling
pixels_healpix_interval = jax.jit(
    pixels_healpix_interval,
    static_argnames=[
        "flag_mask",
        "n_pix_submap",
        "nside",
        "nest",
        "intervals_max_length",
    ],
    donate_argnums=[5, 6],
)  # donates pixels and hit_submap


@kernel(impl=ImplementationType.JAX, name="pixels_healpix")
def pixels_healpix_jax(
    quat_index,
    quats,
    flags,
    flag_mask,
    pixel_index,
    pixels,
    intervals,
    hit_submaps,
    n_pix_submap,
    nside,
    nest,
    use_accel,
):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        quat_index (array, int): size n_det
        quats (array, float64): The flat-packed array of detector quaternions (size ???*n_samp*4).
        flags (array, uint8): size n_samp (or you shouldn't use flags)
        flag_mask (uint8): integer used to select flags (not necesarely a boolean)
        pixel_index (array, int): size n_det
        pixels (array, int64): The detector pixel indices to store the result (size n_det*n_samp).
        intervals (array, float64): size n_view
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (int):
        nside (int): Used to build the healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.
        use_accel (bool): should we use the accelerator

    Returns:
        None (results are stored in pixels and hit_submaps).
    """
    # prepares inputs
    if intervals.size == 0:
        return  # deals with a corner case in tests
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    quat_index_input = MutableJaxArray.to_array(quat_index)
    quats_input = MutableJaxArray.to_array(quats)
    flags_input = MutableJaxArray.to_array(flags)
    pixel_index_input = MutableJaxArray.to_array(pixel_index)
    pixels_input = MutableJaxArray.to_array(pixels)
    hit_submaps_input = MutableJaxArray.to_array(hit_submaps)

    # runs computation
    new_pixels, new_hit_submaps = pixels_healpix_interval(
        quat_index_input,
        quats_input,
        flags_input,
        flag_mask,
        pixel_index_input,
        pixels_input,
        hit_submaps_input,
        n_pix_submap,
        nside,
        nest,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )

    # modifies output buffers in place
    pixels[:] = new_pixels
    hit_submaps[:] = new_hit_submaps  # NOTE: this is a move back to CPU


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix", "ops_sim_ground", "ops_sim_satellite", "ops_demodulate");'
