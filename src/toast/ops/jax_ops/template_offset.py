
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import ImplementationType, select_implementation

# -------------------------------------------------------------------------------------------------
# JAX

def template_offset_add_to_signal_jax(step_length, amp_offset, amplitudes, data_index, det_data, intervals):
    """
    Accumulate offset amplitudes to timestream data.
    Each amplitude value is accumulated to `step_length` number of samples.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int)
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        intervals (array, Interval): size n_view

    Returns:
        None (the result is put in det_data).
    """
    # problem size
    print(f"DEBUG: running 'template_offset_add_to_signal_jax' with n_view:{intervals.size} n_amp:{amplitudes.size} n_all_det:{det_data.shape[0]}")
    
    # loop over the intervals
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        # extract interval slices
        data_interval = det_data[data_index, interval_start:interval_end]
        amplitude_index = amp_offset + (np.arange(interval_start,interval_end) // step_length)
        amplitudes_interval = amplitudes[amplitude_index]
        # does the computation and puts the result in amplitudes
        # TODO this does not use JAX as there is too little computation
        data_interval[:] += amplitudes_interval

def template_offset_project_signal_jax(data_index, det_data, step_length, amp_offset, amplitudes, intervals):
    """
    Accumulate timestream data into offset amplitudes.
    Chunks of `step_length` number of samples are accumulated into the offset amplitudes.

    Args:
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int)
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        intervals (array, Interval): size n_view

    Returns:
        None (the result is put in amplitudes).
    """
    # problem size
    print(f"DEBUG: running 'template_offset_project_signal_jax' data_index:{data_index} with n_view:{intervals.size} n_amp:{amplitudes.size} n_all_det:{det_data.shape[0]} n_samp:{det_data.shape[1]} amp_offset:{amp_offset}")
    
    # loop over the intervals
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        # extract interval slices
        data_interval = det_data[data_index, interval_start:interval_end]
        amplitude_index = amp_offset + (np.arange(interval_start,interval_end) // step_length)
        amplitudes_interval = amplitudes[amplitude_index]
        # does the computation and puts the result in amplitudes
        # TODO this does not use JAX as there is too little computation
        amplitudes_interval[:] += data_interval

def template_offset_apply_diag_precond_jax(offset_var, amplitudes_in, amplitudes_out):
    """
    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitudes_out (array, double): size n_amp

    Returns:
        None (the result is put in amplitudes_out).
    """
    # problem size
    print(f"DEBUG: running 'template_offset_apply_diag_precond_jax' with n_amp:{amplitudes_in.size}")
    
    # runs the computation
    # TODO this does not use JAX as there is too little computation
    amplitudes_out[:] = amplitudes_in * offset_var

# -------------------------------------------------------------------------------------------------
# NUMPY

def template_offset_add_to_signal_numpy(step_length, amp_offset, amplitudes, data_index, det_data, intervals):
    """
    Accumulate offset amplitudes to timestream data.
    Each amplitude value is accumulated to `step_length` number of samples.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int)
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        intervals (array, Interval): size n_view

    Returns:
        None (the result is put in det_data).
    """
    print(f"DEBUG: running 'template_offset_add_to_signal_numpy' with n_view:{intervals.size} n_amp:{amplitudes.size} n_all_det:{det_data.shape[0]}")

    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']
        for isamp in range(interval_start,interval_end+1):
            amp = amp_offset + int(isamp / step_length)
            det_data[data_index,isamp] += amplitudes[amp]

def template_offset_project_signal_numpy(data_index, det_data, step_length, amp_offset, amplitudes, intervals):
    """
    Accumulate timestream data into offset amplitudes.
    Chunks of `step_length` number of samples are accumulated into the offset amplitudes.

    Args:
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int)
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        intervals (array, Interval): size n_view

    Returns:
        None (the result is put in amplitudes).
    """
    print(f"DEBUG: running 'template_offset_project_signal_numpy' data_index:{data_index} with n_view:{intervals.size} n_amp:{amplitudes.size} n_all_det:{det_data.shape[0]} amp_offset:{amp_offset}")

    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']
        for isamp in range(interval_start,interval_end+1):
            amp = amp_offset + int(isamp / step_length)
            amplitudes[amp] += det_data[data_index,isamp]

def template_offset_apply_diag_precond_numpy(offset_var, amplitudes_in, amplitudes_out):
    """
    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitudes_out (array, double): size n_amp

    Returns:
        None (the result is put in amplitudes_out).
    """
    print(f"DEBUG: running 'template_offset_apply_diag_precond_numpy' with n_amp:{amplitudes_in.size}")

    amplitudes_out[:] = amplitudes_in * offset_var

# -------------------------------------------------------------------------------------------------
# C++

"""
void template_offset_add_to_signal"(
    int64_t step_length,
    int64_t amp_offset,
    py::buffer amplitudes,
    int32_t data_index,
    py::buffer det_data,
    py::buffer intervals) 
{
    // This is used to return the actual shape of each buffer
    std::vector <int64_t> temp_shape(3);

    double * raw_amplitudes = extract_buffer <double> (amplitudes, "amplitudes", 1, temp_shape, {-1});
    int64_t n_amp = temp_shape[0];

    double * raw_det_data = extract_buffer <double> (det_data, "det_data", 2, temp_shape, {-1, -1});
    int64_t n_all_det = temp_shape[0];
    int64_t n_samp = temp_shape[1];

    Interval * raw_intervals = extract_buffer <Interval> (intervals, "intervals", 1, temp_shape, {-1});
    int64_t n_view = temp_shape[0];

    double * dev_amplitudes = raw_amplitudes;
    double * dev_det_data = raw_det_data;
    Interval * dev_intervals = raw_intervals;

    for (int64_t iview = 0; iview < n_view; iview++) 
    {
        #pragma omp parallel for
        for (int64_t isamp = dev_intervals[iview]['first']; isamp <= dev_intervals[iview]['last']; isamp++)
        {
            int64_t d = data_index * n_samp * isamp;
            int64_t amp = amp_offset + (int64_t)(isamp / step_length);
            dev_det_data[d] += dev_amplitudes[amp];
        }
    }
}

void template_offset_project_signal(
    int32_t data_index,
    py::buffer det_data,
    int64_t step_length,
    int64_t amp_offset,
    py::buffer amplitudes,
    py::buffer intervals) 
{
    // This is used to return the actual shape of each buffer
    std::vector <int64_t> temp_shape(3);

    double * raw_amplitudes = extract_buffer <double> (amplitudes, "amplitudes", 1, temp_shape, {-1});
    int64_t n_amp = temp_shape[0];

    double * raw_det_data = extract_buffer <double> (det_data, "det_data", 2, temp_shape, {-1, -1});
    int64_t n_all_det = temp_shape[0];
    int64_t n_samp = temp_shape[1];

    Interval * raw_intervals = extract_buffer <Interval> (intervals, "intervals", 1, temp_shape, {-1});
    int64_t n_view = temp_shape[0];

    double * dev_amplitudes = raw_amplitudes;
    double * dev_det_data = raw_det_data;
    Interval * dev_intervals = raw_intervals;

    for (int64_t iview = 0; iview < n_view; iview++) {
        #pragma omp parallel for
        for (int64_t isamp = dev_intervals[iview]['first']; isamp <= dev_intervals[iview]['last']; isamp++)
        {
            int64_t d = data_index * n_samp * isamp;
            int64_t amp = amp_offset + (int64_t)(isamp / step_length);
            dev_amplitudes[amp] += dev_det_data[d];
        }
    }
}

void template_offset_apply_diag_precond(
    py::buffer offset_var,
    py::buffer amplitudes_in,
    py::buffer amplitudes_out)
{
    // This is used to return the actual shape of each buffer
    std::vector <int64_t> temp_shape(3);

    double * raw_amp_in = extract_buffer <double> (amplitudes_in, "amplitudes_in", 1, temp_shape, {-1});
    int64_t n_amp = temp_shape[0];

    double * raw_amp_out = extract_buffer <double> (amplitudes_out, "amplitudes_out", 1, temp_shape, {n_amp});

    double * raw_offset_var = extract_buffer <double> (offset_var, "offset_var", 1, temp_shape, {n_amp});

    double * dev_amp_in = raw_amp_in;
    double * dev_amp_out = raw_amp_out;
    double * dev_offset_var = raw_offset_var;

    #pragma omp parallel for
    for (int64_t iamp = 0; iamp < n_amp; iamp++)
    {
        dev_amp_out[iamp] = dev_amp_in[iamp];
        dev_amp_out[iamp] *= dev_offset_var[iamp];
    }
}
"""

# -------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
template_offset_add_to_signal = select_implementation(template_offset_add_to_signal_numpy, 
                                                      template_offset_add_to_signal_numpy, 
                                                      template_offset_add_to_signal_jax, 
                                                      default_implementationType=ImplementationType.NUMPY)
template_offset_project_signal = select_implementation(template_offset_project_signal_numpy, 
                                                       template_offset_project_signal_numpy, 
                                                       template_offset_project_signal_jax, 
                                                       default_implementationType=ImplementationType.NUMPY)
template_offset_apply_diag_precond = select_implementation(template_offset_apply_diag_precond_numpy, 
                                                           template_offset_apply_diag_precond_numpy, 
                                                           template_offset_apply_diag_precond_jax, 
                                                           default_implementationType=ImplementationType.NUMPY)

# To test:
# python -c 'import toast.tests; toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker")'

# to bench:
# use template_offset config with template not disabled in slurm and check 
# (function) run_mapmaker|MapMaker._exec|solve|SolverLHS._exec|Pipeline._exec|TemplateMatrix._exec
# field (line 174 for add and project, line 150 for just add) in timing.csv
