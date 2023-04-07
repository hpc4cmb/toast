# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ...accelerator import ImplementationType, kernel


@kernel(impl=ImplementationType.NUMPY, name="offset_add_to_signal")
def offset_add_to_signal_numpy(
    step_length,
    amp_offset,
    n_amp_views,
    amplitudes,
    data_index,
    det_data,
    intervals,
    use_accel,
):
    """
    Accumulate offset amplitudes to timestream data.
    Each amplitude value is accumulated to `step_length` number of samples.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int)
        n_amp_views (array, int): size n_view
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        intervals (array, Interval): size n_view
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in det_data).
    """
    offset = amp_offset
    for interval, view_offset in zip(intervals, n_amp_views):
        samples = slice(interval.first, interval.last + 1, 1)
        sampidx = np.arange(0, interval.last - interval.first + 1, dtype=np.int64)
        amp_vals = np.array([amplitudes[offset + x] for x in (sampidx // step_length)])
        det_data[data_index, samples] += amp_vals
        offset += view_offset


@kernel(impl=ImplementationType.NUMPY, name="offset_project_signal")
def offset_project_signal_numpy(
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
):
    """
    Accumulate timestream data into offset amplitudes.
    Chunks of `step_length` number of samples are accumulated into the offset amplitudes.

    Args:
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        flag_index (int), strictly negative in the absence of a detector flag
        flag_data (array, bool) size n_all_det*n_samp
        flag_mask (int),
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int)
        n_amp_views (array, int): size n_view
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        intervals (array, Interval): size n_view
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in amplitudes).
    """
    offset = amp_offset
    for interval, view_offset in zip(intervals, n_amp_views):
        samples = slice(interval.first, interval.last + 1, 1)
        ampidx = (
            offset
            + np.arange(0, interval.last - interval.first + 1, dtype=np.int64)
            // step_length
        )
        ddata = det_data[data_index][samples]
        # skip sample if it is flagged
        if flag_index >= 0:
            # We have detector flags
            ddata = np.array(
                ((flag_data[flag_index] & flag_mask) == 0), dtype=np.float64
            )[samples]
            ddata *= det_data[data_index][samples]
        # updates amplitude
        # using np.add to insure atomicity
        np.add.at(amplitudes, ampidx, ddata)
        offset += view_offset


@kernel(impl=ImplementationType.NUMPY, name="offset_apply_diag_precond")
def offset_apply_diag_precond_numpy(
    offset_var, amplitudes_in, amplitudes_out, use_accel
):
    """
    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitudes_out (array, double): size n_amp
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in amplitudes_out).
    """
    amplitudes_out[:] = amplitudes_in * offset_var


# To test:
# python -c 'import toast.tests; toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker")'
