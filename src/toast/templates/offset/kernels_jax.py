# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp
import numpy as np

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import INTERVALS_JAX, JaxIntervals
from ...jax.mutableArray import MutableJaxArray
from ...utils import Logger


def offset_add_to_signal_intervals(
    step_length,
    amplitudes,
    data_index,
    det_data,
    amp_offset,
    n_amp_views,
    interval_starts,
    interval_ends,
    intervals_max_length,
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
        amp_offset (int): starting offset
        n_amp_views (array, int): subsequent offsets (size n_view)
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        det_data
    """
    # debugging information
    log = Logger.get()
    log.debug(f"offset_add_to_signal: jit-compiling.")

    # computes offsets
    # a cumulative sums of the n_amp_views starting at amp_offset
    offsets = jnp.roll(n_amp_views, 1)
    offsets = offsets.at[0].set(amp_offset)
    offsets = jnp.cumsum(offsets)
    # prepare offsets intervals
    offset_starts = offsets
    offset_ends = offsets + (interval_ends - interval_starts) // step_length
    # end+1 as the interval is inclusive
    offsets_max_length = 1 + (intervals_max_length - 1) // step_length

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    nb_intervals = interval_starts.size
    nb_amplitudes = offsets_max_length

    # pad the intervals to insure that are exactly nb_amplitudes*step_length long
    # meaning that we do not have to deal with leftovers
    intervals_max_length = nb_amplitudes * step_length

    # computes interval data
    intervals = JaxIntervals(
        interval_starts, interval_ends + 1, intervals_max_length
    )  # end+1 as the interval is inclusive
    offsets = JaxIntervals(
        offset_starts, offset_ends + 1, offsets_max_length
    )  # end+1 as the interval is inclusive
    amplitudes_interval = JaxIntervals.get(amplitudes, offsets)  # amplitudes[offsets]
    det_data_interval = JaxIntervals.get(
        det_data, (data_index, intervals)
    )  # det_data[data_index, intervals]

    # All amplitudes now have step_length samples.
    det_data_interval = jnp.reshape(
        det_data_interval, newshape=(nb_intervals, -1, step_length)
    )
    # det_data_interval += amplitudes[:, jnp.newaxis]
    det_data_interval = det_data_interval + amplitudes_interval[:, :, jnp.newaxis]
    det_data_interval = jnp.reshape(det_data_interval, newshape=(nb_intervals, -1))

    # updates det_data and returns
    # det_data[data_index, intervals] = det_data_interval
    det_data = JaxIntervals.set(det_data, (data_index, intervals), det_data_interval)
    return det_data


# jit compilation
offset_add_to_signal_intervals = jax.jit(
    offset_add_to_signal_intervals,
    static_argnames=["step_length", "intervals_max_length"],
    donate_argnums=[3],
)  # det_data


@kernel(impl=ImplementationType.JAX, name="offset_add_to_signal")
def offset_add_to_signal_jax(
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
    det_data_input = MutableJaxArray.to_array(det_data)
    amplitudes = MutableJaxArray.to_array(amplitudes)
    n_amp_views = MutableJaxArray.to_array(n_amp_views)
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)

    # run computation
    det_data[:] = offset_add_to_signal_intervals(
        step_length,
        amplitudes,
        data_index,
        det_data_input,
        amp_offset,
        n_amp_views,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


def offset_project_signal_intervals(
    data_index,
    det_data,
    use_flag,
    flag_index,
    flag_data,
    flag_mask,
    step_length,
    amplitudes,
    amp_offset,
    n_amp_views,
    interval_starts,
    interval_ends,
    intervals_max_length,
):
    """
    Chunks of `step_length` number of samples are accumulated into the offset
    amplitudes.  If step_length does not evenly divide into the total number of
    samples, the final amplitude will be padded with zeros.

    Process all the intervals as a block.

    Args:
        data_index (int)
        det_data (array, double): The float64 timestream values (size n_all_det*n_samp).
        use_flag (bool): should we use flags
        flag_index (int), strictly negative in the absence of a detector flag
        flag_data (array, bool) size n_all_det*n_samp
        flag_mask (int),
        step_length (int64):  The minimum number of samples for each offset.
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        amp_offset (int)
        n_amp_views (array, int): size n_view
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval

    Returns:
        amplitudes
    """
    # debugging information
    log = Logger.get()
    log.debug(f"offset_project_signal: jit-compiling.")

    # computes offsets
    # a cumulative sums of the n_amp_views starting at amp_offset
    offsets = jnp.roll(n_amp_views, 1)
    offsets = offsets.at[0].set(amp_offset)
    offsets = jnp.cumsum(offsets)
    # prepare offsets intervals
    offset_starts = offsets
    offset_ends = offsets + (interval_ends - interval_starts) // step_length
    # end+1 as the interval is inclusive
    offsets_max_length = 1 + (intervals_max_length - 1) // step_length

    # gets interval information
    nb_amplitudes = offsets_max_length
    nb_intervals = interval_starts.size

    # pad the intervals to insure that are exactly nb_amplitudes*step_length long
    # meaning that we do not have to deal with leftovers
    intervals_max_length = nb_amplitudes * step_length

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

    # Reshape such that all amplitudes have step_length samples.
    det_data_interval = jnp.reshape(
        det_data_interval, newshape=(nb_intervals, -1, step_length)
    )
    # amplitudes += np.sum(det_data_interval, axis=-1)
    amplitudes_interval = amplitudes_interval + jnp.sum(det_data_interval, axis=-1)

    # updates amplitudes and returns
    amplitudes = JaxIntervals.set(
        amplitudes, offsets, amplitudes_interval
    )  # amplitudes[offsets] = amplitudes_interval
    return amplitudes


# jit compilation
offset_project_signal_intervals = jax.jit(
    offset_project_signal_intervals,
    static_argnames=[
        "use_flag",
        "flag_mask",
        "step_length",
        "intervals_max_length",
    ],
    donate_argnums=[7],
)  # donate amplitude


@kernel(impl=ImplementationType.JAX, name="offset_project_signal")
def offset_project_signal_jax(
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
    use_flag = flag_index >= 0
    det_data = MutableJaxArray.to_array(det_data)
    flag_data = MutableJaxArray.to_array(flag_data)
    amplitudes_input = MutableJaxArray.to_array(amplitudes)
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)

    # run computation
    amplitudes[:] = offset_project_signal_intervals(
        data_index,
        det_data,
        use_flag,
        flag_index,
        flag_data,
        flag_mask,
        step_length,
        amplitudes_input,
        amp_offset,
        n_amp_views,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


def offset_apply_diag_precond_inner(offset_var, amplitudes_in, amplitudes_out):
    """
    Simple multiplication.

    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitudes_out (array, double): size n_amp

    Returns:
        amplitudes_out (array, double): size n_amp
    """
    return amplitudes_in * offset_var


# jit compilation
offset_apply_diag_precond_inner = jax.jit(
    offset_apply_diag_precond_inner,
    donate_argnums=[2],
)  # donate amplitudes_out


@kernel(impl=ImplementationType.JAX, name="offset_apply_diag_precond")
def offset_apply_diag_precond_jax(offset_var, amplitudes_in, amplitudes_out, use_accel):
    """
    Simple multiplication.

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
    amplitudes_out[:] = offset_apply_diag_precond_inner(
        offset_var, amplitudes_in, amplitudes_out
    )


# To test:
# python -c 'import toast.tests; toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker")'
