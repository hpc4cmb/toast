# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..._libtoast import template_offset_add_to_signal as libtoast_offset_add_to_signal
from ..._libtoast import (
    template_offset_apply_diag_precond as libtoast_offset_apply_diag_precond,
)
from ..._libtoast import (
    template_offset_project_signal as libtoast_offset_project_signal,
)
from ...accelerator import ImplementationType, kernel, use_accel_jax
from .kernels_numpy import (
    offset_add_to_signal_numpy,
    offset_apply_diag_precond_numpy,
    offset_project_signal_numpy,
)

if use_accel_jax:
    from .kernels_jax import (
        offset_add_to_signal_jax,
        offset_apply_diag_precond_jax,
        offset_project_signal_jax,
    )


@kernel(impl=ImplementationType.DEFAULT)
def offset_add_to_signal(
    step_length,
    amp_offset,
    n_amp_views,
    amplitudes,
    data_index,
    det_data,
    intervals,
    use_accel=False,
):
    """Kernel to accumulate offset amplitudes to timestream data.

    Each amplitude value is accumulated to `step_length` number of samples.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int64):  The first amplitude for this detector.
        n_amp_views (array, int):  The number of amplitudes for each interval.
        amplitudes (array, double): The amplitude data.
        data_index (int):  The detector to process.
        det_data (array, double):  The array of data for all detectors.
        intervals (array, Interval):  The intervals to process.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return libtoast_offset_add_to_signal(
        step_length,
        amp_offset,
        n_amp_views,
        amplitudes,
        data_index,
        det_data,
        intervals,
        use_accel,
    )


@kernel(impl=ImplementationType.DEFAULT)
def offset_project_signal(
    data_index,
    det_data,
    flag_index,
    flag_data,
    flag_mask,
    step_length,
    amp_offset,
    n_amp_views,
    amplitudes,
    intervals,
    use_accel=False,
):
    """Kernel to accumulate timestream data into offset amplitudes.

    Chunks of `step_length` number of samples from one detector are accumulated
    into the offset amplitudes.

    Args:
        data_index (int):  The detector index to process.
        det_data (array, double):  The data for all detectors.
        flag_index (int):  The detector index in the flag data.  A negative value
            indicates no flags.
        flag_data (array, uint8):  The flags for all detectors.
        flag_mask (uint8):  The flag mask to apply.
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int64):  The first amplitude for this detector.
        n_amp_views (array, int):  The number of amplitudes for each interval.
        amplitudes (array, double): The amplitude data.
        intervals (array, Interval):  The intervals to process.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return libtoast_offset_project_signal(
        data_index,
        det_data,
        flag_index,
        flag_data,
        flag_mask,
        step_length,
        amp_offset,
        n_amp_views,
        amplitudes,
        intervals,
        use_accel,
    )


@kernel(impl=ImplementationType.DEFAULT)
def offset_apply_diag_precond(
    offset_var,
    amplitudes_in,
    amplitudes_out,
    use_accel=False,
):
    """
    Args:
        offset_var (array, double):  The variance weight to apply to each amplitude.
        amplitudes_in (array, double):  Input amplitude data
        amplitudes_out (array, double):  Output amplitude data
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return libtoast_offset_apply_diag_precond(
        offset_var,
        amplitudes_in,
        amplitudes_out,
        use_accel,
    )


@kernel(impl=ImplementationType.COMPILED, name="offset_add_to_signal")
def offset_add_to_signal_compiled(*args, use_accel=False):
    return libtoast_offset_add_to_signal(*args, use_accel)


@kernel(impl=ImplementationType.COMPILED, name="offset_project_signal")
def offset_project_signal_compiled(*args, use_accel=False):
    return libtoast_offset_project_signal(*args, use_accel)


@kernel(impl=ImplementationType.COMPILED, name="offset_apply_diag_precond")
def offset_apply_diag_precond_compiled(*args, use_accel=False):
    return libtoast_offset_apply_diag_precond(*args, use_accel)
