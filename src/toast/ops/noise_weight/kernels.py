# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..._libtoast import noise_weight as libtoast_noise_weight
from ...accelerator import ImplementationType, kernel, use_accel_jax
from .kernels_numpy import noise_weight_numpy

if use_accel_jax:
    from .kernels_jax import noise_weight_jax


@kernel(impl=ImplementationType.COMPILED, name="noise_weight")
def noise_weight_compiled(*args, use_accel=False):
    return libtoast_noise_weight(*args, use_accel)


@kernel(impl=ImplementationType.DEFAULT)
def noise_weight(
    det_data,
    det_data_index,
    intervals,
    detector_weights,
    use_accel=False,
):
    """Kernel for applying noise weights to detector timestreams.

    Args:
        det_data (array):  The detector data at each sample for each detector.
        det_data_index (array):  The index into the data array for each detector.
        intervals (array):  The array of sample intervals.
        detector_weights (array):  Array of noise weights for each detector.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return libtoast_noise_weight(
        det_data,
        det_data_index,
        intervals,
        detector_weights,
        use_accel,
    )
