# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import INTERVALS_JAX
from ...jax.maps import imap
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
pointing_detector_inner = imap(
    pointing_detector_inner,
    in_axes={
        "focalplane": ["n_det", ...],
        "boresight": ["n_samp", ...],
        "quats": ["n_det", "n_samp", ...],
        "flag": ["n_samp"],
        "mask": int,
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
        "intervals_max_length": int,
    },
    interval_axis="n_samp",
    interval_starts="interval_starts",
    interval_ends="interval_ends",
    interval_max_length="intervals_max_length",
    output_name="quats",
)


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

    # indexes quats
    quats_indexed = quats[quat_index, :, :]

    # process the intervals
    new_quats_indexed = pointing_detector_inner(
        focalplane,
        boresight,
        quats_indexed,
        shared_flags,
        shared_flag_mask,
        interval_starts,
        interval_ends,
        intervals_max_length,
    )

    # updates quats at the index
    quats = quats.at[quat_index, :, :].set(new_quats_indexed)
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
