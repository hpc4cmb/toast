# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


def template_offset_add_to_signal(
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
        interval_start = interval.first
        interval_end = interval.last
        for isamp in range(interval_start, interval_end + 1):
            amp = offset + int(isamp / step_length)
            det_data[data_index, isamp] += amplitudes[amp]
        offset += view_offset

def _py_add_to_signal(
    self,
    step_length,
    amp_offset,
    n_amp_views,
    amplitudes,
    data_index,
    det_data,
    intr_data,
):
    """Internal python implementation for comparison testing."""
    offset = amp_offset
    for ivw, vw in enumerate(intr_data):
        samples = slice(vw.first, vw.last + 1, 1)
        sampidx = np.arange(vw.first, vw.last + 1, dtype=np.int64)
        amp_vals = np.array([amplitudes[offset + x] for x in (sampidx // step_length)])
        det_data[data_index[0], samples] += amp_vals
        offset += n_amp_views[ivw]


def template_offset_project_signal(
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
        flag_index (int), strictly negative in the absence of a detcetor flag
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
        interval_start = interval.first
        interval_end = interval.last + 1
        for isamp in range(interval_start, interval_end):
            det_data_samp = det_data[data_index, isamp]
            # skip sample if it is flagged
            if flag_index >= 0:
                flagged = (flag_data[flag_index, isamp] & flag_mask) != 0
                if flagged:
                    continue
            # updates amplitude
            amp = offset + isamp // step_length
            amplitudes[
                amp
            ] += det_data_samp  # WARNING: this has to be done one at a time to avoid conflicts
        offset += view_offset

def _py_project_signal(
    self,
    data_index,
    det_data,
    flag_index,
    flag_data,
    flag_mask,
    step_length,
    amp_offset,
    n_amp_views,
    amplitudes,
    intr_data,
):
    """Internal python implementation for comparison testing."""
    offset = amp_offset
    for ivw, vw in enumerate(intr_data):
        samples = slice(vw.first, vw.last + 1, 1)
        ampidx = (
            offset + np.arange(vw.first, vw.last + 1, dtype=np.int64) // step_length
        )
        ddata = det_data[data_index[0]][samples]
        if flag_index[0] >= 0:
            # We have detector flags
            ddata = np.array(
                ((flag_data[flag_index[0]] & flag_mask) == 0), dtype=np.float64
            )
            ddata *= det_data[data_index[0]][samples]
        np.add.at(amplitudes, ampidx, ddata)
        offset += n_amp_views[ivw]

def template_offset_apply_diag_precond(
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
