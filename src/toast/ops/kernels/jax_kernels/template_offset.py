# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from ....utils import Logger

from ....jax.mutableArray import MutableJaxArray
from ....jax.intervals import INTERVALS_JAX, JaxIntervals


def template_offset_add_to_signal_intervals(
    step_length,
    amplitudes,
    data_index,
    det_data,
    interval_starts,
    interval_ends,
    intervals_max_length,
    offset_starts,
    offset_ends,
    offsets_max_length,
):
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
    # debugging information
    log = Logger.get()
    log.debug(f"template_offset_add_to_signal: jit-compiling.")

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    nb_intervals = interval_starts.size
    nb_amplitudes = offsets_max_length

    # computes interval data
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive
    offsets = JaxIntervals(
        offset_starts, offset_ends + 1, offsets_max_length
    )  # end+1 as the interval is inclusive
    amplitudes_interval = JaxIntervals.get(amplitudes, offsets)  # amplitudes[offsets]
    data_interval = JaxIntervals.get(
        det_data, (data_index, intervals)
    )  # det_data[data_index, intervals]

    # All but the last amplitude have step_length samples.
    data_first = data_interval[:, : (nb_amplitudes - 1) * step_length]
    data_first = jnp.reshape(data_first, newshape=(nb_intervals, -1, step_length))
    new_data_first = data_first + amplitudes_interval[:, :-1, jnp.newaxis]
    # data_first[:] += amplitudes[:-1, jnp.newaxis]
    data_interval = data_interval.at[:, : (nb_amplitudes - 1) * step_length].set(
        new_data_first.reshape((nb_intervals, -1))
    )

    # Now handle the final amplitude.
    # data_last = data[(nb_amplitudes - 1) * step_length:]
    # data_last[:] += amplitudes[-1]
    data_interval = data_interval.at[:, (nb_amplitudes - 1) * step_length :].add(
        amplitudes_interval[:, -1]
    )

    # updates det_data and returns
    # det_data[data_index, intervals] = data_interval
    det_data = JaxIntervals.set(det_data, (data_index, intervals), data_interval)
    return det_data


# jit compilation
template_offset_add_to_signal_intervals = jax.jit(
    template_offset_add_to_signal_intervals,
    static_argnames=["step_length", "intervals_max_length", "offsets_max_length"],
    donate_argnums=[3],
)  # det_data


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
    # prepare inputs
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
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
    offsets_max_length = int(np.max(1 + offsets_end - offsets_start))

    # run computation
    det_data[:] = template_offset_add_to_signal_intervals(
        step_length,
        amplitudes,
        data_index,
        det_data_input,
        intervals.first,
        intervals.last,
        intervals_max_length,
        offsets_start,
        offsets_end,
        offsets_max_length,
    )


def template_offset_project_signal_intervals(
    data_index,
    det_data,
    use_flag,
    flag_index,
    flag_data,
    flag_mask,
    step_length,
    amplitudes,
    interval_starts,
    interval_ends,
    intervals_max_length,
    offset_starts,
    offset_ends,
    offsets_max_length,
):
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
    # debugging information
    log = Logger.get()
    log.debug(f"template_offset_project_signal: jit-compiling.")

    # gets interval information
    nb_amplitudes = offsets_max_length
    nb_intervals = interval_starts.size

    # computes interval data
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive
    offsets = JaxIntervals(
        offset_starts, offset_ends + 1, offsets_max_length
    )  # end+1 as the interval is inclusive
    amplitudes_interval = JaxIntervals.get(amplitudes, offsets)  # amplitudes[offsets]
    det_data_interval = JaxIntervals.get(
        det_data, (data_index, intervals), padding_value=0.0
    )  # det_data[data_index,intervals]

    # skip flagged samples
    if use_flag:
        flags_interval = JaxIntervals.get(
            flag_data, (flag_index, intervals)
        )  # flag_data[flag_index,intervals]
        flagged = (flags_interval & flag_mask) != 0
        det_data_interval = jnp.where(flagged, 0.0, det_data_interval)

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    data_first = det_data_interval[:, : (nb_amplitudes - 1) * step_length]
    data_first = jnp.reshape(data_first, newshape=(nb_intervals, -1, step_length))
    data_last = det_data_interval[:, (nb_amplitudes - 1) * step_length :]

    # All but the last amplitude have step_length samples.
    # amplitudes[:-1] += np.sum(data_first, axis=1)
    amplitudes_interval = amplitudes_interval.at[:, :-1].add(
        jnp.sum(data_first, axis=-1)
    )

    # Now handle the final amplitude.
    # amplitudes[-1] += np.sum(data_last)
    amplitudes_interval = amplitudes_interval.at[:, -1].add(jnp.sum(data_last, axis=-1))

    # updates amplitudes and returns
    amplitudes = JaxIntervals.set(
        amplitudes, offsets, amplitudes_interval
    )  # amplitudes[offsets] = amplitudes_interval
    return amplitudes


# jit compilation
template_offset_project_signal_intervals = jax.jit(
    template_offset_project_signal_intervals,
    static_argnames=[
        "use_flag",
        "flag_mask",
        "step_length",
        "intervals_max_length",
        "offsets_max_length",
    ],
    donate_argnums=[7],
)  # donate amplitude


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
    # prepare inputs
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
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
    offsets_max_length = int(np.max(1 + offsets_end - offsets_start))

    # run computation
    amplitudes[:] = template_offset_project_signal_intervals(
        data_index,
        det_data,
        use_flag,
        flag_index,
        flag_data,
        flag_mask,
        step_length,
        amplitudes_input,
        intervals.first,
        intervals.last,
        intervals_max_length,
        offsets_start,
        offsets_end,
        offsets_max_length,
    )


def template_offset_apply_diag_precond(
    offset_var, amplitudes_in, amplitudes_out, use_accel
):
    """
    NOTE this does not use JAX as there is too little computation

    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitudes_out (array, double): size n_amp
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in amplitudes_out).
    """
    # cast the data if needed
    offset_var = MutableJaxArray.to_array(offset_var)
    amplitudes_in = MutableJaxArray.to_array(amplitudes_in)

    # runs the computation
    amplitudes_out[:] = amplitudes_in * offset_var


# To test:
# python -c 'import toast.tests; toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker")'
