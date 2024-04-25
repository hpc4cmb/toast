# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
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

# ----------------------------------------------------------------------------------------
# IQU


def stokes_weights_IQU_inner(pin, hwpang, eps, gamma, cal, IAU):
    """
    Compute the Stokes weights for one detector.

    Args:
        pin (array, float64):  The array of detector quaternions (size 4).
        hwpang (float64):  The HWP angle.
        eps (float):  The cross polar response.
        gamma (float):  Detector polarization angle.
        cal (float):  A constant to apply to the pointing weights.
        IAU (int):  Sign factor for U stokes term.

    Returns:
        weights (array, float64):  The detector weights for the specified mode (size 3)
    """
    eta = (1.0 - eps) / (1.0 + eps)

    # applies quaternion rotations
    vd = qarray.rotate_zaxis(pin)
    vo = qarray.rotate_xaxis(pin)

    # The vector orthogonal to the line of sight that is parallel
    # to the local meridian.
    dir_ang = jnp.arctan2(vd[1], vd[0])
    dir_r = jnp.sqrt(1.0 - vd[2] * vd[2])
    vm_z = -dir_r
    vm_x = vd[2] * jnp.cos(dir_ang)
    vm_y = vd[2] * jnp.sin(dir_ang)

    # Compute the rotation angle from the meridian vector to the
    # orientation vector.  The direction vector is normal to the plane
    # containing these two vectors, so the rotation angle is:
    # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
    alpha_y = (
        vd[0] * (vm_y * vo[2] - vm_z * vo[1])
        - vd[1] * (vm_x * vo[2] - vm_z * vo[0])
        + vd[2] * (vm_x * vo[1] - vm_y * vo[0])
    )
    alpha_x = vm_x * vo[0] + vm_y * vo[1] + vm_z * vo[2]
    alpha = jnp.arctan2(alpha_y, alpha_x)
    ang = 2.0 * (2.0 * (gamma - hwpang) - alpha)
    weights = jnp.array(
        [cal, jnp.cos(ang) * eta * cal, -jnp.sin(ang) * eta * cal * IAU]
    )
    return weights


# maps over samples, intervals and detectors
stokes_weights_IQU_inner = imap(
    stokes_weights_IQU_inner,
    in_axes={
        "quats": ["n_det", "n_samp", ...],
        "weights": ["n_det", "n_samp", ...],
        "hwp": ["n_samp"],
        "eps": ["n_det"],
        "gamma": ["n_det"],
        "cal": ["n_det"],
        "IAU": int,
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
        "intervals_max_length": int,
    },
    interval_axis="n_samp",
    interval_starts="interval_starts",
    interval_ends="interval_ends",
    interval_max_length="intervals_max_length",
    output_name="weights",
)


def stokes_weights_IQU_interval(
    quat_index,
    quats,
    weight_index,
    weights,
    hwp,
    epsilon,
    gamma,
    cal,
    IAU,
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
        gamma (array, float):  The polarization orientation angle of each detector (size n_det).
        cal (array, float64):  An array to apply to the pointing weights (size n_det).
        IAU (bool):  Whether to use IAU convention.
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        weights
    """
    # debugging information
    log = Logger.get()
    log.debug(f"stokes_weights_IQU: jit-compiling.")

    # extract indexes
    quats_indexed = quats[quat_index, :, :]
    weights_indexed = weights[weight_index, :, :]

    # convert IAU to an integer for easier handling
    IAU_sign = -1 if IAU else 1

    # Are we using a half wave plate?
    if hwp.size == 0:
        # No half wave plate
        n_samp = weights.shape()[1]
        hwp = jnp.zeros(shape=(n_samp,), dtype=float)
        gamma = jnp.zeros_like(gamma)
        # In this case the U stokes coefficient is negated
        IAU_sign *= -1

    # does the computation
    new_weights_indexed = stokes_weights_IQU_inner(
        quats_indexed,
        weights_indexed,
        hwp,
        epsilon,
        gamma,
        cal,
        IAU_sign,
        interval_starts,
        interval_ends,
        intervals_max_length,
    )

    # updates results and returns
    weights = weights.at[weight_index, :, :].set(new_weights_indexed)
    return weights


# jit compiling
stokes_weights_IQU_interval = jax.jit(
    stokes_weights_IQU_interval,
    static_argnames=["intervals_max_length", "IAU"],
    donate_argnums=[3],
)  # donates weights


@kernel(impl=ImplementationType.JAX, name="stokes_weights_IQU")
def stokes_weights_IQU_jax(
    quat_index,
    quats,
    weight_index,
    weights,
    hwp,
    intervals,
    epsilon,
    gamma,
    cal,
    IAU,
    use_accel,
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
        gamma (array, float):  The polarization orientation angle of each detector.
        cal (array, float64):  An array to apply to the pointing weights (size n_det).
        IAU (bool):  If True, use IAU convention.
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
    gamma_input = MutableJaxArray.to_array(gamma)
    cal_input = MutableJaxArray.to_array(cal)

    # runs computation
    weights[:] = stokes_weights_IQU_interval(
        quat_index_input,
        quats_input,
        weight_index_input,
        weights_input,
        hwp_input,
        epsilon_input,
        gamma_input,
        cal_input,
        IAU,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


# ----------------------------------------------------------------------------------------
# I


def stokes_weights_I_inner(cal):
    """
    Compute the Stokes weights for one detector.

    Args:
        cal (float):  A constant to apply to the pointing weights.

    Returns:
        weights (float64):  The detector weights for the specified mode
    """
    return cal


# maps over samples, intervals and detectors
stokes_weights_I_inner = imap(
    stokes_weights_I_inner,
    in_axes={
        "weights": ["n_det", "n_samp"],
        "cal": ["n_det"],
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
        "intervals_max_length": int,
    },
    interval_axis="n_samp",
    interval_starts="interval_starts",
    interval_ends="interval_ends",
    interval_max_length="intervals_max_length",
    output_name="weights",
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
        cal (array, float64):  An array to apply to the pointing weights (size n_det).
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        weights (array, float64)
    """
    # debugging information
    log = Logger.get()
    log.debug(f"stokes_weights_I: jit-compiling.")

    # extract indexed values
    weights_indexed = weights[weight_index, :]

    # does computation
    new_weights_indexed = stokes_weights_I_inner(
        weights_indexed, cal, interval_starts, interval_ends, intervals_max_length
    )

    # updates results and returns
    weights = weights.at[weight_index, :].set(new_weights_indexed)
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
        cal (array, float64):  An array to apply to the pointing weights (size n_det).
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
# export TOAST_GPU_JAX=true; export TOAST_GPU_HYBRID_PIPELINES=true; export TOAST_LOGLEVEL=DEBUG; python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix"); toast.tests.run("ops_sim_tod_dipole"); toast.tests.run("ops_stokes_weights");'
