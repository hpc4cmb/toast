# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import ALL, INTERVALS_JAX, JaxIntervals
from ...jax.math import qarray
from ...jax.mutableArray import MutableJaxArray
from ...utils import Logger


def pointing_detector_inner(focalplane, boresight, flag, mask):
    """
    Process a single detector and a single sample inside an interval.

    Args:
        focalplane (array, double): size 4
        boresight (array, double): size 4
        flag (uint8)
        mask (uint8)

    Returns:
        quats (array, double): size 4
    """
    quats = jnp.where(
        (flag & mask) == 0, qarray.mult_one_one(boresight, focalplane), focalplane
    )
    return quats


# maps over intervals and detectors
# pointing_detector_inner = jax_xmap(
#    pointing_detector_inner,
#    in_axes=[
#        ["detectors", ...],  # focalplane
#        ["intervals", "interval_size", ...],  # boresight
#        ["intervals", "interval_size"],  # flags
#        [...],
#    ],  # mask
#    out_axes=["detectors", "intervals", "interval_size", ...],
# )
# using vmap as the static arguments triggers the following error:
# "ShardingContext cannot be used with xmap"
# TODO revisit once this issue is solved [bug with static argnum](https://github.com/google/jax/issues/10741)
pointing_detector_inner = jax.vmap(
    pointing_detector_inner, in_axes=[None, 0, 0, None], out_axes=0
)  # interval_size
pointing_detector_inner = jax.vmap(
    pointing_detector_inner, in_axes=[None, 0, 0, None], out_axes=0
)  # intervals
pointing_detector_inner = jax.vmap(
    pointing_detector_inner, in_axes=[0, None, None, None], out_axes=0
)  # detectors


def pointing_detector_interval(
    focalplane,
    boresight,
    quat_index,
    quats,
    shared_flags,
    shared_flag_mask,
    interval_starts,
    interval_ends,
    intervals_max_length,
):
    """
    Process all the intervals as a block.

    Args:
        focalplane (array, double): size n_det*4
        boresight (array, double): size n_samp*4
        quat_index (array, int): size n_det
        quats (array, double): size n_det*n_samp*4
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        quats
    """
    # debugging information
    log = Logger.get()
    log.debug(f"pointing_detector: jit-compiling.")

    # extract interval slices
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive
    boresight_interval = JaxIntervals.get(
        boresight, (intervals, ALL)
    )  # boresight[intervals,:]
    shared_flags_interval = JaxIntervals.get(
        shared_flags, intervals
    )  # shared_flags[intervals]

    # process the interval then updates quats in place
    new_quats_interval = pointing_detector_inner(
        focalplane, boresight_interval, shared_flags_interval, shared_flag_mask
    )
    quats = JaxIntervals.set(
        quats, (quat_index, intervals, ALL), new_quats_interval
    )  # quats[quat_index,intervals,:] = new_quats_interval
    return quats


# jit compiling
pointing_detector_interval = jax.jit(
    pointing_detector_interval,
    static_argnames=["shared_flag_mask", "intervals_max_length"],
    donate_argnums=[3],
)  # donates quats


@kernel(impl=ImplementationType.JAX, name="pointing_detector")
def pointing_detector_jax(
    focalplane,
    boresight,
    quat_index,
    quats,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel=False,
):
    """
    Args:
        focalplane (array, double): size n_det*4
        boresight (array, double): size n_samp*4
        quat_index (array, int): size n_det
        quats (array, double): size n_det*n_samp*4
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in quats).
    """
    # prepares inputs
    if intervals.size == 0:
        return  # deals with a corner case in tests
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    focalplane_input = MutableJaxArray.to_array(focalplane)
    boresight_input = MutableJaxArray.to_array(boresight)
    quat_index_input = MutableJaxArray.to_array(quat_index)
    quats_input = MutableJaxArray.to_array(quats)
    shared_flags_input = MutableJaxArray.to_array(shared_flags)

    # runs computation
    quats[:] = pointing_detector_interval(
        focalplane_input,
        boresight_input,
        quat_index_input,
        quats_input,
        shared_flags_input,
        shared_flag_mask,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix", "ops_demodulate")'
