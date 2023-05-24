# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
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


def stokes_weights_IQU_inner(eps, cal, pin, hwpang):
    """
    Compute the Stokes weights for one detector.

    Args:
        eps (float):  The cross polar response.
        cal (float):  A constant to apply to the pointing weights.
        pin (array, float64):  The array of detector quaternions (size 4).
        hwpang (float64):  The HWP angle (could be None).

    Returns:
        weights (array, float64):  The detector weights for the specified mode (size 3)
    """
    # applies quaternion rotations
    dir = qarray.rotate_zaxis(pin)
    orient = qarray.rotate_xaxis(pin)

    # computes by and bx
    by = orient[0] * dir[1] - orient[1] * dir[0]
    bx = (
        orient[0] * (-dir[2] * dir[0])
        + orient[1] * (-dir[2] * dir[1])
        + orient[2] * (dir[0] * dir[0] + dir[1] * dir[1])
    )

    # computes detang
    detang = jnp.arctan2(by, bx)
    detang = detang + 2.0 * hwpang
    detang = 2.0 * detang

    # puts values into weights
    eta = (1.0 - eps) / (1.0 + eps)
    weights = jnp.array([cal, jnp.cos(detang) * eta * cal, jnp.sin(detang) * eta * cal])
    return weights


# maps over samples, intervals and detectors
# stokes_weights_IQU_inner = jax_xmap(
#    stokes_weights_IQU_inner,
#    in_axes=[
#        ["detectors"],  # epsilon
#        [...],  # cal
#        ["detectors", "intervals", "interval_size", ...],  # quats
#        ["intervals", "interval_size"],
#    ],  # hwp
#    out_axes=["detectors", "intervals", "interval_size", ...],
# )
# using vmap as the static arguments triggers the following error:
# "ShardingContext cannot be used with xmap"
# TODO revisit once this issue is solved [bug with static argnum](https://github.com/google/jax/issues/10741)
stokes_weights_IQU_inner = jax.vmap(
    stokes_weights_IQU_inner, in_axes=[None, None, 0, 0], out_axes=0
)  # interval_size
stokes_weights_IQU_inner = jax.vmap(
    stokes_weights_IQU_inner, in_axes=[None, None, 0, 0], out_axes=0
)  # intervals
stokes_weights_IQU_inner = jax.vmap(
    stokes_weights_IQU_inner, in_axes=[0, None, 0, None], out_axes=0
)  # detectors


def stokes_weights_IQU_interval(
    quat_index,
    quats,
    weight_index,
    weights,
    hwp,
    epsilon,
    cal,
    interval_starts,
    interval_ends,
    intervals_max_length,
):
    """
    Process all the intervals as a block.

    Args:
        quat_index (array, int): size n_det
        quats (array, double): size ???*n_samp*4
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, float64): The flat packed detectors weights for the specified mode (size n_det*n_samp*3)
        hwp (array, float64):  The HWP angles (size n_samp).
        epsilon (array, float):  The cross polar response (size n_det).
        cal (float):  A constant to apply to the pointing weights.
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        weights
    """
    # debugging information
    log = Logger.get()
    log.debug(f"stokes_weights_IQU: jit-compiling.")

    # extract interval slices
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive
    quats_interval = JaxIntervals.get(
        quats, (quat_index, intervals, ALL)
    )  # quats[quat_index,intervals,:]

    # insures hwp is a non empty array
    if hwp.size == 0:
        nb_intervals = interval_starts.size
        hwp_interval = jnp.zeros((nb_intervals, intervals_max_length))
    else:
        hwp_interval = JaxIntervals.get(hwp, intervals)  # hwp[intervals]

    # does the computation
    new_weights_interval = stokes_weights_IQU_inner(
        epsilon, cal, quats_interval, hwp_interval
    )

    # updates results and returns
    # weights[weight_index,intervals,:] = new_weights_interval
    weights = JaxIntervals.set(
        weights, (weight_index, intervals, ALL), new_weights_interval
    )
    return weights


# jit compiling
stokes_weights_IQU_interval = jax.jit(
    stokes_weights_IQU_interval,
    static_argnames=["intervals_max_length"],
    donate_argnums=[3],
)  # donates weights


@kernel(impl=ImplementationType.JAX, name="stokes_weights_IQU")
def stokes_weights_IQU_jax(
    quat_index, quats, weight_index, weights, hwp, intervals, epsilon, cal, use_accel
):
    """
    Compute the Stokes weights for the "IQU" mode.

    Args:
        quat_index (array, int): size n_det
        quats (array, double): size ???*n_samp*4
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, float64): The flat packed detectors weights for the specified mode (size n_det*n_samp*3)
        hwp (array, float64):  The HWP angles (size n_samp).
        intervals (array, Interval): The intervals to modify (size n_view)
        epsilon (array, float):  The cross polar response (size n_det).
        cal (float):  A constant to apply to the pointing weights.
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in weights).
    """
    # prepares inputs
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    quat_index_input = MutableJaxArray.to_array(quat_index)
    quats_input = MutableJaxArray.to_array(quats)
    weight_index_input = MutableJaxArray.to_array(weight_index)
    weights_input = MutableJaxArray.to_array(weights)
    hwp_input = MutableJaxArray.to_array(hwp)
    epsilon_input = MutableJaxArray.to_array(epsilon)

    # runs computation
    weights[:] = stokes_weights_IQU_interval(
        quat_index_input,
        quats_input,
        weight_index_input,
        weights_input,
        hwp_input,
        epsilon_input,
        cal,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


def stokes_weights_I_interval(
    weight_index,
    weights,
    cal,
    interval_starts,
    interval_ends,
    intervals_max_length,
):
    """
    Process all the intervals as a block.

    Args:
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, float64): The flat packed detectors weights for the specified mode (size n_det*n_samp)
        cal (float):  A constant to apply to the pointing weights.
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        weights (array, float64)
    """
    # debugging information
    log = Logger.get()
    log.debug(f"stokes_weights_I: jit-compiling.")

    # extract interval slices
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive

    # updates results and returns
    # weights[weight_index,intervals] = cal
    weights = JaxIntervals.set(weights, (weight_index, intervals), cal)
    return weights


# jit compiling
stokes_weights_I_interval = jax.jit(
    stokes_weights_I_interval,
    static_argnames=["intervals_max_length"],
    donate_argnums=[1],
)  # donates weights


@kernel(impl=ImplementationType.JAX, name="stokes_weights_I")
def stokes_weights_I_jax(weight_index, weights, intervals, cal, use_accel):
    """
    Compute the Stokes weights for the "I" mode.

    Args:
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, float64): The flat packed detectors weights for the specified mode (size n_det*n_samp)
        intervals (array, Interval): The intervals to modify (size n_view)
        cal (float):  A constant to apply to the pointing weights.
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in weights).
    """
    # prepares inputs
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    weight_index_input = MutableJaxArray.to_array(weight_index)
    weights_input = MutableJaxArray.to_array(weights)

    # runs computation
    weights[:] = stokes_weights_I_interval(
        weight_index_input,
        weights_input,
        cal,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix"); toast.tests.run("ops_sim_tod_dipole")'
