# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..._libtoast import filter_poly2D as libtoast_filter_poly2D
from ..._libtoast import filter_polynomial as libtoast_filter_polynomial

from ...accelerator import kernel, ImplementationType, use_accel_jax

from .kernels_numpy import (
    filter_poly2D_numpy,
    filter_polynomial_numpy,
)

if use_accel_jax:
    from .kernels_jax import (
        filter_poly2D_jax,
        filter_polynomial_jax,
    )


@kernel(impl=ImplementationType.DEFAULT)
def stokes_weights_I(
    weight_index, 
    weights, 
    intervals, 
    cal, 
    use_accel=False,
):
    """Kernel for computing trivial intensity-only Stokes pointing weights.

    Args:
        weight_index (array):  The index into the weights array for each detector.
        weights (array):  The array of I, Q, and U weights at each sample for each 
            detector.
        intervals (array):  The array of sample intervals.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return stokes_weights_I(
        weight_index, 
        weights, 
        intervals, 
        cal,
        impl=ImplementationType.COMPILED,
        use_accel=use_accel,
    )


@kernel(impl=ImplementationType.DEFAULT)
def stokes_weights_IQU(
    quat_index, 
    quats, 
    weight_index, 
    weights, 
    hwp, 
    intervals, 
    epsilon, 
    cal,
    use_accel=False,
):
    """Kernel for computing the I/Q/U Stokes pointing weights.

    Args:
        quat_index (array):  The index into the detector quaternion array for each
            detector.
        quats (array):  The array of detector quaternions for each sample.
        weight_index (array):  The index into the weights array for each detector.
        weights (array):  The array of I, Q, and U weights at each sample for each 
            detector.
        hwp (array):  The array of orientation angles for an ideal half wave plate.
        intervals (array):  The array of sample intervals.
        epsilon (float):  The cross polar leakage.
        cal (float):  Calibration factor to apply to the weights.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return stokes_weights_IQU(
        quat_index, 
        quats, 
        weight_index, 
        weights, 
        hwp, 
        intervals, 
        epsilon, 
        cal,
        impl=ImplementationType.COMPILED,
        use_accel=use_accel,
    )



@kernel(impl=ImplementationType.COMPILED, name="stokes_weights_I")
def stokes_weights_I_compiled(*args, use_accel=False):
    return libtoast_stokes_weights_I(*args, use_accel)

@kernel(impl=ImplementationType.COMPILED, name="stokes_weights_IQU")
def stokes_weights_IQU_compiled(*args, use_accel=False):
    return libtoast_stokes_weights_IQU(*args, use_accel)

