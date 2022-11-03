# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from ...jax.mutableArray import MutableJaxArray
from ...jax.intervals import INTERVALS_JAX, JaxIntervals
from ...jax.implementation_selection import select_implementation
from ...jax.data_localization import dataMovementTracker

# -------------------------------------------------------------------------------------------------
# JAX


def noise_weight_interval_jax(
    det_data,
    det_data_index,
    detector_weights,
    interval_starts,
    interval_ends,
    intervals_max_length,
):
    """
    multiplies det_data by the weighs in detector_weights, applied to all intervals as a single block

    Args:
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        detector_weights (list, double): The weight to be used for each detcetor (size n_det)
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        det_data
    """
    # display sizes
    print(
        f"DEBUG: jit-compiling 'noise_weight' with n_det:{det_data_index.size} n_view:{interval_starts.size} n_samp:{det_data.shape[-1]} intervals_max_length:{intervals_max_length}"
    )

    # turns detector_weights into a jax array
    detector_weights = jnp.array(detector_weights)

    # extract interval slice
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive
    det_data_interval = JaxIntervals.get(
        det_data, (det_data_index, intervals)
    )  # det_data[det_data_index, intervals]

    # does the computation
    new_det_data_interval = (
        det_data_interval * detector_weights[:, jnp.newaxis, jnp.newaxis]
    )

    # updates results and returns
    # det_data[det_data_index, intervals] = new_det_data_interval
    det_data = JaxIntervals.set(
        det_data, (det_data_index, intervals), new_det_data_interval
    )
    return det_data


# jit compiling
noise_weight_interval_jax = jax.jit(
    noise_weight_interval_jax,
    static_argnames=["intervals_max_length"],
    donate_argnums=[0],
)  # donates det_data


def noise_weight_jax(det_data, det_data_index, intervals, detector_weights, use_accel):
    """
    multiplies det_data by the weighs in detector_weights

    Args:
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        detector_weights (list, double): The weight to be used for each detcetor (size n_det)
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    # prepares inputs
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    det_data_input = MutableJaxArray.to_array(det_data)
    det_data_index = MutableJaxArray.to_array(det_data_index)

    # track data movement
    dataMovementTracker.add(
        "noise_weight",
        use_accel,
        [
            det_data_input,
            det_data_index,
            detector_weights,
            intervals.first,
            intervals.last,
        ],
        [det_data],
    )

    # performs computation and updates det_data in place
    det_data[:] = noise_weight_interval_jax(
        det_data_input,
        det_data_index,
        detector_weights,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


# -------------------------------------------------------------------------------------------------
# NUMPY


def noise_weight_numpy(
    det_data, det_data_index, intervals, detector_weights, use_accel
):
    """
    multiplies det_data by the weighs in detector_weights

    Args:
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        detector_weights (list, double): The weight to be used for each detcetor (size n_det)
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    # iterates on detectors and intervals
    n_det = det_data_index.size
    for idet in range(n_det):
        d_index = det_data_index[idet]
        detector_weight = detector_weights[idet]
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1
            # applies the multiplication
            det_data[d_index, interval_start:interval_end] *= detector_weight


# -------------------------------------------------------------------------------------------------
# COMPILED


def noise_weight_compiled(
    det_data, det_data_index, intervals, detector_weights, use_accel
):
    # TODO make a C++ implementation with support for OpenMP target offload
    if use_accel:
        raise RuntimeError(
            "noise_weight_compiled: there is currently no OpenMP offload version of noise_weight"
        )
    noise_weight_numpy(det_data, det_data_index, intervals, detector_weights, use_accel)


# -------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
noise_weight = select_implementation(
    noise_weight_compiled, noise_weight_numpy, noise_weight_jax
)
