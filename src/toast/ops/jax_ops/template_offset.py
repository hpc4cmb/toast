
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import assert_data_localization, dataMovementTracker, ImplementationType, select_implementation, MutableJaxArray
from .utils.intervals import JaxIntervals, ALL
from ..._libtoast import template_offset_add_to_signal as template_offset_add_to_signal_compiled, template_offset_project_signal as template_offset_project_signal_compiled, template_offset_apply_diag_precond as template_offset_apply_diag_precond_compiled

# -------------------------------------------------------------------------------------------------
# JAX

def template_offset_add_to_signal_intervals_jax(step_length, amplitudes, data_index, det_data,
                                                interval_starts, interval_ends, intervals_max_length,
                                                offset_starts, offset_ends, offsets_max_length):
    """
    Each amplitude value is accumulated to `step_length` number of samples.  The
    final offset will be at least this many samples, but may be more if the step
    size does not evenly divide into the number of samples.

    Process all the intervals as a block.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval
        offset_starts (array, int): size n_view
        offset_ends (array, int): size n_view
        offsets_max_length (int): maximum length of an interval

    Returns:
        det_data
    """
    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    nb_intervals = interval_starts.size
    nb_amplitudes = offsets_max_length
    print(f"DEBUG: jit-compiling 'template_offset_add_to_signal' with step_length:{step_length} nb_amplitudes:{nb_amplitudes} nb_samples:{det_data.shape[-1]}")
    
    # computes interval data
    intervals = JaxIntervals(interval_starts, interval_ends+1, intervals_max_length) # end+1 as the interval is inclusive
    offsets = JaxIntervals(offset_starts, offset_ends+1, offsets_max_length) # end+1 as the interval is inclusive
    amplitudes_interval = JaxIntervals.get(amplitudes, offsets) # amplitudes[offsets]
    data_interval = JaxIntervals.get(det_data, (data_index, intervals)) # det_data[data_index, intervals]

    # All but the last amplitude have step_length samples.
    data_first = data_interval[:, :(nb_amplitudes - 1) * step_length]
    data_first = jnp.reshape(data_first, newshape=(nb_intervals,-1,step_length))
    new_data_first = data_first + amplitudes_interval[:,:-1, jnp.newaxis]
    #data_first[:] += amplitudes[:-1, jnp.newaxis]
    data_interval = data_interval.at[:,:(nb_amplitudes - 1) * step_length].set(new_data_first.reshape((nb_intervals,-1)))

    # Now handle the final amplitude.
    #data_last = data[(nb_amplitudes - 1) * step_length:]
    #data_last[:] += amplitudes[-1]
    data_interval = data_interval.at[:,(nb_amplitudes - 1) * step_length:].add(amplitudes_interval[:,-1])

    # updates det_data and returns
    # det_data[data_index, intervals] = data_interval
    det_data = JaxIntervals.set(det_data, (data_index, intervals), data_interval)
    return det_data

# jit compilation
template_offset_add_to_signal_intervals_jax = jax.jit(template_offset_add_to_signal_intervals_jax, 
                                                      static_argnames=['step_length', 'intervals_max_length', 'offsets_max_length'], 
                                                      donate_argnums=[3]) # det_data

def template_offset_add_to_signal_jax(step_length, amp_offset, n_amp_views, amplitudes, data_index, det_data, intervals, use_accel):
    """
    Accumulate offset amplitudes to timestream data.
    Each amplitude value is accumulated to `step_length` number of samples.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amp_offset (int): starting offset
        n_amp_views (array, int): subsequent offsets (size n_view)
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        intervals (array, Interval): size n_view
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in det_data).
    """
    # make sure the data is where we expect it
    assert_data_localization('template_offset_add_to_signal', use_accel, [amplitudes, det_data], [det_data])

    # prepare inputs
    intervals_max_length = np.max(1 + intervals.last - intervals.first) # end+1 as the interval is inclusive
    det_data_input = MutableJaxArray.to_array(det_data)
    amplitudes = MutableJaxArray.to_array(amplitudes)

    # computes offsets
    # a cumulative sums of the n_amp_views starting at amp_offset
    offsets = np.roll(n_amp_views, 1)
    offsets[0] = amp_offset
    offsets = np.cumsum(offsets)
    # prepare offsets intervals
    offsets_start = offsets + intervals.first // step_length
    offsets_end = offsets + intervals.last // step_length
    offsets_max_length = np.max(1 + offsets_end - offsets_start)

    # track data movement
    dataMovementTracker.add("template_offset_add_to_signal", use_accel, [amplitudes, det_data_input, intervals.first, intervals.last, offsets_start, offsets_end], [det_data])

    # run computation
    det_data[:] = template_offset_add_to_signal_intervals_jax(step_length, amplitudes, data_index, det_data_input,
                                                              intervals.first, intervals.last, intervals_max_length,
                                                              offsets_start, offsets_end, offsets_max_length)

def template_offset_project_signal_intervals_jax(data_index, det_data, use_flag, flag_index, flag_data, flag_mask, step_length, amplitudes,
                                                 interval_starts, interval_ends, intervals_max_length,
                                                 offset_starts, offset_ends, offsets_max_length):
    """
    Chunks of `step_length` number of samples are accumulated into the offset
    amplitudes.  If step_length does not evenly divide into the total number of
    samples, the final amplitude will be extended to include the remainder.

    Process all the intervals as a block.

    Args:
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        use_flag (bool): should we use flags
        flag_index (int), strictly negative in the absence of a detcetor flag
        flag_data (array, bool) size n_all_det*n_samp
        flag_mask (int),
        step_length (int64):  The minimum number of samples for each offset.
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval
        offset_starts (array, int): size n_view
        offset_ends (array, int): size n_view
        offsets_max_length (int): maximum length of an interval

    Returns:
        amplitudes
    """
    # gets interval information
    nb_amplitudes = offsets_max_length
    nb_intervals = interval_starts.size
    print(f"DEBUG: jit-compiling 'template_offset_project_signal' n_samp_interval:{det_data.size} flag_mask:{flag_mask} step_length:{step_length} nb_amplitudes:{nb_amplitudes}")

    # computes interval data
    intervals = JaxIntervals(interval_starts, interval_ends+1, intervals_max_length) # end+1 as the interval is inclusive
    offsets = JaxIntervals(offset_starts, offset_ends+1, offsets_max_length) # end+1 as the interval is inclusive
    amplitudes_interval = JaxIntervals.get(amplitudes, offsets) # amplitudes[offsets]
    det_data_interval = JaxIntervals.get(det_data, (data_index,intervals), padding_value=0.0) # det_data[data_index,intervals]

    # skip flagged samples
    if use_flag:
        flags_interval = JaxIntervals.get(flag_data, (flag_index,intervals)) # flag_data[flag_index,intervals]
        flagged = (flags_interval & flag_mask) != 0
        det_data_interval = jnp.where(flagged, 0.0, det_data_interval)

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    data_first = det_data_interval[:, :(nb_amplitudes - 1) * step_length]
    data_first = jnp.reshape(data_first, newshape=(nb_intervals, -1, step_length))
    data_last = det_data_interval[:, (nb_amplitudes - 1) * step_length:]
        
    # All but the last amplitude have step_length samples.
    #amplitudes[:-1] += np.sum(data_first, axis=1)
    amplitudes_interval = amplitudes_interval.at[:, :-1].add(jnp.sum(data_first, axis=-1))

    # Now handle the final amplitude.
    #amplitudes[-1] += np.sum(data_last)
    amplitudes_interval = amplitudes_interval.at[:, -1].add(jnp.sum(data_last, axis=-1))
    
    # updates amplitudes and returns
    amplitudes = JaxIntervals.set(amplitudes, offsets, amplitudes_interval) # amplitudes[offsets] = amplitudes_interval
    return amplitudes

# jit compilation
template_offset_project_signal_intervals_jax = jax.jit(template_offset_project_signal_intervals_jax, 
                                                   static_argnames=['use_flag', 'flag_mask','step_length', 'intervals_max_length', 'offsets_max_length'],
                                                   donate_argnums=[7]) # donate amplitude

def template_offset_project_signal_jax(data_index, det_data, flag_index, flag_data, flag_mask, step_length, amp_offset, n_amp_views, amplitudes, intervals, use_accel):
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
    # make sure the data is where we expect it
    assert_data_localization('template_offset_project_signal', use_accel, [det_data, flag_data, amplitudes], [amplitudes])

    # prepare inputs
    intervals_max_length = np.max(1 + intervals.last - intervals.first) # end+1 as the interval is inclusive
    use_flag = flag_index >= 0
    det_data = MutableJaxArray.to_array(det_data)
    flag_data = MutableJaxArray.to_array(flag_data)
    amplitudes_input = MutableJaxArray.to_array(amplitudes)

    # computes offsets
    # a cumulative sums of the n_amp_views starting at amp_offset
    offsets = np.roll(n_amp_views, 1)
    offsets[0] = amp_offset
    offsets = np.cumsum(offsets)
    # prepare offsets intervals
    offsets_start = offsets + intervals.first // step_length
    offsets_end = offsets + intervals.last // step_length
    offsets_max_length = np.max(1 + offsets_end - offsets_start)

    # track data movement
    dataMovementTracker.add("template_offset_project_signal", use_accel, [det_data, flag_data, amplitudes_input, intervals.first, intervals.last, offsets_start, offsets_end], [amplitudes])

    # run computation
    amplitudes[:] = template_offset_project_signal_intervals_jax(data_index, det_data, use_flag, flag_index, flag_data, flag_mask, step_length, amplitudes_input,
                                                                 intervals.first, intervals.last, intervals_max_length,
                                                                 offsets_start, offsets_end, offsets_max_length)

def template_offset_apply_diag_precond_jax(offset_var, amplitudes_in, amplitudes_out, use_accel):
    """
    TODO this does not use JAX as there is too little computation

    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitudes_out (array, double): size n_amp
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in amplitudes_out).
    """
    # make sure the data is where we expect it
    assert_data_localization('template_offset_apply_diag_precond', use_accel, [amplitudes_in, offset_var], [amplitudes_out])
    
    # problem size
    #print(f"DEBUG: running 'template_offset_apply_diag_precond_jax' with n_amp:{amplitudes_in.size}")
    
    # track data movement
    dataMovementTracker.add("template_offset_apply_diag_precond", use_accel, [amplitudes_in, offset_var], [amplitudes_out])

    # runs the computation
    amplitudes_out[:] = amplitudes_in * offset_var

# -------------------------------------------------------------------------------------------------
# NUMPY

def template_offset_add_to_signal_numpy(step_length, amp_offset, n_amp_views, amplitudes, data_index, det_data, intervals, use_accel):
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
    print(f"DEBUG: running 'template_offset_add_to_signal_numpy' with n_view:{intervals.size} n_amp:{amplitudes.size} n_all_det:{det_data.shape[0]}")

    offset = amp_offset
    for interval, view_offset in zip(intervals, n_amp_views):
        interval_start = interval['first']
        interval_end = interval['last']
        for isamp in range(interval_start,interval_end+1):
            amp = offset + int(isamp / step_length)
            det_data[data_index,isamp] += amplitudes[amp]
        offset += view_offset

def template_offset_project_signal_numpy(data_index, det_data, flag_index, flag_data, flag_mask, step_length, amp_offset, n_amp_views, amplitudes, intervals, use_accel):
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
    print(f"DEBUG: running 'template_offset_project_signal_numpy' data_index:{data_index} flag_data:{flag_data.shape} flag_index:{flag_index} flag_mask:{flag_mask} n_view:{intervals.size} n_amp:{amplitudes.size} n_all_det:{det_data.shape[0]} n_samp:{det_data.shape[1]} amp_offset:{amp_offset}")

    offset = amp_offset
    for interval, view_offset in zip(intervals, n_amp_views):
        interval_start = interval['first']
        interval_end = interval['last']+1
        for isamp in range(interval_start,interval_end):
            det_data_samp = det_data[data_index,isamp]
            # skip sample if it is flagged
            if flag_index >= 0:
                flagged = (flag_data[flag_index,isamp] & flag_mask) != 0
                if flagged: continue
            # updates amplitude
            amp = offset + isamp // step_length
            amplitudes[amp] += det_data_samp # WARNING: this has to be done one at a time to avoid conflicts
        offset += view_offset

def template_offset_apply_diag_precond_numpy(offset_var, amplitudes_in, amplitudes_out, use_accel):
    """
    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitudes_out (array, double): size n_amp
        use_accel (bool): should we use the accelerator

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
    py::buffer n_amp_views,
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

    int64_t * raw_n_amp_views = extract_buffer <int64_t> (n_amp_views, "n_amp_views", 1, temp_shape, {n_view});

    int64_t offset = amp_offset;
    for (int64_t iview = 0; iview < n_view; iview++) 
    {
        for (int64_t isamp = raw_intervals[iview].first; isamp <= raw_intervals[iview].last; isamp++) 
        {
            int64_t d = data_index * n_samp + isamp;
            int64_t amp = offset + (int64_t)(isamp / step_length);
            raw_det_data[d] += raw_amplitudes[amp];
        }
        offset += raw_n_amp_views[iview];
    }
}

void template_offset_project_signal(
    int32_t data_index,
    py::buffer det_data,
    int32_t flag_index,
    py::buffer flag_data,
    uint8_t flag_mask,
    int64_t step_length,
    int64_t amp_offset,
    py::buffer n_amp_views,
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

    int64_t * raw_n_amp_views = extract_buffer <int64_t> (n_amp_views, "n_amp_views", 1, temp_shape, {n_view});

    // Optionally use flags
    bool use_flags = false;
    uint8_t * raw_det_flags = (uint8_t *)omgr.null;
    if (flag_index >= 0) 
    {
        raw_det_flags = extract_buffer <uint8_t> (flag_data, "flag_data", 2, temp_shape, {-1, n_samp});
        use_flags = true;
    }

    double * dev_amplitudes = raw_amplitudes;
    double * dev_det_data = raw_det_data;
    Interval * dev_intervals = raw_intervals;

    int64_t offset = amp_offset;
    for (int64_t iview = 0; iview < n_view; iview++) {
        #pragma omp parallel for
        for (int64_t isamp = dev_intervals[iview]['first']; isamp <= dev_intervals[iview]['last']; isamp++)
        {
            int64_t d = data_index * n_samp * isamp;
            int64_t amp = offset + (int64_t)(isamp / step_length);

            double contrib = 0.0;
            if (use_flags) 
            {
                int64_t f = flag_index * n_samp + isamp;
                uint8_t check = raw_det_flags[f] & flag_mask;
                if (check == 0) 
                {
                    contrib = raw_det_data[d];
                }
            } 
            else 
            {
                contrib = raw_det_data[d];
            }
            #pragma omp atomic
            dev_amplitudes[amp] += contrib;
        }
        offset += raw_n_amp_views[iview];
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
template_offset_add_to_signal = select_implementation(template_offset_add_to_signal_compiled, 
                                                      template_offset_add_to_signal_numpy, 
                                                      template_offset_add_to_signal_jax)
template_offset_project_signal = select_implementation(template_offset_project_signal_compiled, 
                                                       template_offset_project_signal_numpy, 
                                                       template_offset_project_signal_jax)
template_offset_apply_diag_precond = select_implementation(template_offset_apply_diag_precond_compiled, 
                                                           template_offset_apply_diag_precond_numpy, 
                                                           template_offset_apply_diag_precond_jax)

# To test:
# python -c 'import toast.tests; toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker")'

# to bench:
# use template_offset config with template not disabled in slurm and check 
# (function) run_mapmaker|MapMaker._exec|solve|SolverLHS._exec|Pipeline._exec|TemplateMatrix._exec
# field (line 174 for add and project, line 150 for just add) in timing.csv
