# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ... import qarray as qa
from ..._libtoast import pointing_detector as libtoast_pointing_detector
from ...accelerator import ImplementationType, kernel, use_accel_jax
from .kernels_numpy import pointing_detector_numpy

if use_accel_jax:
    from .kernels_jax import pointing_detector_jax


@kernel(impl=ImplementationType.DEFAULT)
def pointing_detector(
    focalplane,
    boresight,
    quat_index,
    quats,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel=False,
):
    """Kernel for computing detector quaternion pointing.

    Args:
        focalplane (array):  Detector offset quaternions for each detector.
        boresight (array):  Boresight quaternion pointing for each sample.
        quat_index (array):  The index into the detector quaternion array for each
            detector in the focalplane array.
        quats (array):  The array of detector quaternions for each sample.
        intervals (array):  The array of sample intervals.
        shared_flags (array):  The array of common flags for each sample.
        shared_flag_mask (int):  The flag mask to apply.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return libtoast_pointing_detector(
        focalplane,
        boresight,
        quat_index,
        quats,
        intervals,
        shared_flags,
        shared_flag_mask,
        use_accel,
    )


@kernel(impl=ImplementationType.COMPILED, name="pointing_detector")
def pointing_detector_compiled(*args, use_accel=False):
    return libtoast_pointing_detector(*args, use_accel)
