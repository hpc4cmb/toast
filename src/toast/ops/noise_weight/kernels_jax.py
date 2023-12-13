# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import INTERVALS_JAX
from ...jax.maps import imap
from ...jax.mutableArray import MutableJaxArray
from ...utils import Logger


def noise_weight_inner(det_data, detector_weights):
    """
    multiplies det_data by the weighs in detector_weights

    Args:
        det_data (float)
        detector_weights (double): The weight to be used for the detector

    Returns:
        det_data
    """
    return det_data * detector_weights


# maps over intervals and detectors
noise_weight_inner = imap(
    noise_weight_inner,
    in_axes={
        "det_data": ["n_det", "n_samp"],
        "detector_weights": ["n_det"],
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
        "intervals_max_length": int,
    },
    interval_axis="n_samp",
    interval_starts="interval_starts",
    interval_ends="interval_ends",
    interval_max_length="intervals_max_length",
    output_name="det_data",
    output_as_input=True,
)


def noise_weight_interval(
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
        detector_weights (array, double): The weight to be used for each detcetor (size n_det)
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        det_data
    """
    # debugging information
    log = Logger.get()
    log.debug(f"noise_weight: jit-compiling.")

    # extract indexes
    det_data_indexed = det_data[det_data_index, :]

    # does the computation
    new_det_data_indexed = noise_weight_inner(
        det_data_indexed,
        detector_weights,
        interval_starts,
        interval_ends,
        intervals_max_length,
    )

    # updates results and returns
    det_data = det_data.at[det_data_index, :].set(new_det_data_indexed)
    return det_data


# jit compiling
noise_weight_interval = jax.jit(
    noise_weight_interval,
    static_argnames=["intervals_max_length"],
    donate_argnums=[0],
)  # donates det_data


@kernel(impl=ImplementationType.JAX, name="noise_weight")
def noise_weight_jax(det_data, det_data_index, intervals, detector_weights, use_accel):
    """
    multiplies det_data by the weighs in detector_weights

    Args:
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        detector_weights (array, double): The weight to be used for each detector (size n_det)
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    # prepares inputs
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    det_data_input = MutableJaxArray.to_array(det_data)
    det_data_index = MutableJaxArray.to_array(det_data_index)

    # performs computation and updates det_data in place
    det_data[:] = noise_weight_interval(
        det_data_input,
        det_data_index,
        detector_weights,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )
