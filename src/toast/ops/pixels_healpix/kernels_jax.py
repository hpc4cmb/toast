# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import INTERVALS_JAX
from ...jax.maps import imap
from ...jax.math import healpix, qarray
from ...jax.mutableArray import MutableJaxArray
from ...utils import Logger


def pixels_healpix_inner(
    quats, use_flags, flag, flag_mask, hit_submaps, n_pix_submap, hpix, nest
):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        quats (array, float64): Detector quaternion (size 4).
        use_flags (bool): should we use flags?
        flag (uint8):
        flag_mask (uint8): integer used to select flags (not necesarely a boolean)
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (int):
        hpix (HPIX_JAX): Healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        (pixel,submap,hit_submap) (int64,int64): The detector pixel indices to store the result.
    """
    # pixel computation
    dir = qarray.rotate_zaxis(quats)
    (phi, region, z, rtz) = healpix.vec2zphi(dir)
    if nest:
        pixel = healpix.zphi2nest(hpix, phi, region, z, rtz)
    else:
        pixel = healpix.zphi2ring(hpix, phi, region, z, rtz)

    # compute sub map
    sub_map = pixel // n_pix_submap

    # applies the flags
    if use_flags:
        is_flagged = (flag & flag_mask) != 0
        pixel = jnp.where(is_flagged, -1, pixel)
        hit_submap = jnp.where(is_flagged, hit_submaps[sub_map], 1)
    else:
        hit_submap = 1

    return pixel, sub_map, hit_submap


# maps over samples and detectors
pixels_healpix_inner = imap(
    pixels_healpix_inner,
    in_axes={
        "quats": ["n_det", "n_samp", ...],
        "use_flags": bool,
        "flags": ["n_samp"],
        "flag_mask": int,
        "hit_submaps": [...],
        "n_pix_submap": int,
        "hpix": healpix.HPIX_JAX,
        "nest": bool,
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
        "intervals_max_length": int,
        "outputs": (["n_det", "n_samp"], ["n_det", "n_samp"], ["n_det", "n_samp"]),
    },
    interval_axis="n_samp",
    interval_starts="interval_starts",
    interval_ends="interval_ends",
    interval_max_length="intervals_max_length",
    output_name="outputs",
)


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

    # create hpix object
    hpix = healpix.HPIX_JAX(nside)

    # extract indexes
    quats_indexed = quats[quat_index, :, :]
    pixels_indexed = pixels[pixel_index, :]
    dummy_sub_map = jnp.zeros_like(pixels_indexed)
    dummy_hit_submaps = hit_submaps[dummy_sub_map]

    # should we use flags?
    use_flags = flag_mask != 0
    n_samp = pixels.shape[1]
    if flags.size != n_samp:
        flags = jnp.empty(shape=(n_samp,))
        use_flags = False

    # does the computation
    outputs = (pixels_indexed, dummy_sub_map, dummy_hit_submaps)
    new_pixels_indexed, sub_map, new_hit_submaps = pixels_healpix_inner(
        quats_indexed,
        use_flags,
        flags,
        flag_mask,
        hit_submaps,
        n_pix_submap,
        hpix,
        nest,
        interval_starts,
        interval_ends,
        intervals_max_length,
        outputs,
    )

    # updates results and returns
    pixels = pixels.at[pixel_index, :].set(new_pixels_indexed)
    hit_submaps = hit_submaps.at[sub_map].set(new_hit_submaps)
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
# export TOAST_GPU_JAX=true; export TOAST_GPU_HYBRID_PIPELINES=true; export TOAST_LOGLEVEL=DEBUG; python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix"); toast.tests.run("ops_sim_ground"); toast.tests.run("ops_sim_satellite"); toast.tests.run("ops_demodulate");'
