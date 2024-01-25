# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import INTERVALS_JAX
from ...jax.maps import imap, xmap
from ...jax.mutableArray import MutableJaxArray
from ...utils import Logger

# ----------------------------------------------------------------------------------------
# offset_add_to_signal


def offset_add_to_signal_inner(
    step_length,
    amplitudes,
    amplitude_flags,
    det_data,
    amplitude_offset,
    amplitude_view_offset,
    sample_index,
):
    """
    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amplitudes (array, double): The float64 amplitude values (size n_amp)
        amplitude_flags (array, int): flags for each amplitude value (size n_amp)
        det_data (double): timestream value
        amplitude_offset (int): starting offset
        amplitude_view_offset (int): offset for the view
        sample_index (int): index of the sample within the interval

    Returns:
       det_data (double)
    """
    # Computes the index of the amplitude
    amplitude_index = (
        amplitude_offset + amplitude_view_offset + (sample_index // step_length)
    )
    # Mask out contributions where amplitude_flags are non-zero
    amplitude = jnp.where(
        (amplitude_flags[amplitude_index] == 0), amplitudes[amplitude_index], 0.0
    )
    return det_data + amplitude


# maps over intervals
offset_add_to_signal_inner = imap(
    offset_add_to_signal_inner,
    in_axes={
        "step_length": int,
        "amplitudes": [...],
        "amplitude_flags": [...],
        "det_data": ["n_samp"],
        "amplitude_offset": int,
        "amplitude_view_offset": ["n_intervals"],
        "sample_index": ["intervals_max_length"],
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
        "intervals_max_length": int,
    },
    interval_axis="n_samp",
    interval_starts="interval_starts",
    interval_ends="interval_ends",
    interval_max_length="intervals_max_length",
    output_name="det_data",
    output_as_input=True,
)


def offset_add_to_signal_intervals(
    step_length,
    amplitudes,
    amplitude_flags,
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
        amplitude_flags (array, int): flags for each amplitude value (size n_amp)
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

    # get inputs
    det_data_indexed = det_data[data_index, :]
    amp_view_off = jnp.roll(n_amp_views, shift=1)
    amp_view_off = amp_view_off.at[0].set(0)
    sample_indices = jnp.arange(start=0, stop=intervals_max_length)

    # runs computation
    new_det_data_indexed = offset_add_to_signal_inner(
        step_length,
        amplitudes,
        amplitude_flags,
        det_data_indexed,
        amp_offset,
        amp_view_off,
        sample_indices,
        interval_starts,
        interval_ends,
        intervals_max_length,
    )

    # updates det_data and returns
    det_data = det_data.at[data_index, :].set(new_det_data_indexed)
    return det_data


# jit compilation
offset_add_to_signal_intervals = jax.jit(
    offset_add_to_signal_intervals,
    static_argnames=["step_length", "intervals_max_length"],
    donate_argnums=[4],
)  # det_data


@kernel(impl=ImplementationType.JAX, name="offset_add_to_signal")
def offset_add_to_signal_jax(
    step_length,
    amp_offset,
    n_amp_views,
    amplitudes,
    amplitude_flags,
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
        amplitude_flags (array, int): flags for each amplitude value (size n_amp)
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
    amplitude_flags = MutableJaxArray.to_array(amplitude_flags)
    n_amp_views = MutableJaxArray.to_array(n_amp_views)
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)

    # run computation
    det_data[:] = offset_add_to_signal_intervals(
        step_length,
        amplitudes,
        amplitude_flags,
        data_index,
        det_data_input,
        amp_offset,
        n_amp_views,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


# ----------------------------------------------------------------------------------------
# offset_project_signal


def offset_project_signal_sample(det_data, use_flag, flag_data, flag_mask):
    """
    Compute the contribution for a given sample

    Args:
        det_data (double): timestream value
        use_flag (bool): should we use flags
        flag_data (bool),
        flag_mask (int),

    Returns:
        contribution (double): value to add to the corresponding amplitude
    """
    if use_flag:
        valid_sample = (flag_data & flag_mask) == 0
        contribution = jnp.where(valid_sample, det_data, 0.0)
    else:
        contribution = det_data

    return contribution


# maps over samples in a block of size step_length
offset_project_signal_samples = xmap(
    offset_project_signal_sample,
    in_axes={
        "det_data": ["step_length"],
        "use_flag": bool,
        "flag_data": ["step_length"],
        "flag_mask": int,
    },
    out_axes=["step_length"],
)


def offset_project_signal_steplength_block(
    step_length,
    det_data,
    use_flag,
    flag_data,
    flag_mask,
    amplitude_offset,
    amplitude_view_offset,
    block_index,
    interval_start,
    interval_end,
):
    """
    Computes the contribution and index for a block of samples of size step_length

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        det_data (array[double]): timestream value
        use_flag (bool): should we use flags
        flag_data (array[bool]),
        flag_mask (int),
        amplitude_offset (int): starting offset
        amplitude_view_offset (int): offset for the view
        block_index (int): index of the sample within the interval
        interval_start (int): begining of the current interval
        interval_end (int): end of the current interval

    Returns:
        (amplitude_index, contribution) (int,double): index in amplitude and value to add (atomically) there
    """
    # indices and mask to insure we iterate inside the block / interval
    block_indices = (
        interval_start
        + block_index * step_length
        + jnp.arange(start=0, stop=step_length)
    )
    block_mask = block_indices <= interval_end

    # extract block data
    det_data = det_data[block_indices]
    flag_data = flag_data[block_indices]

    # computes and sums contribution within the inerval
    contributions = offset_project_signal_samples(
        det_data, use_flag, flag_data, flag_mask
    )
    contributions_masked = jnp.where(block_mask, contributions, 0.0)
    contribution = jnp.sum(contributions_masked)

    # computes amplitude index
    amplitude_index = amplitude_offset + amplitude_view_offset + block_index
    return (amplitude_index, contribution)


# maps over nb_intervals and blocks_per_interval (intervals_max_length // steplength)
offset_project_signal_steplength_blocks = xmap(
    offset_project_signal_steplength_block,
    in_axes={
        "step_length": int,
        "det_data": [...],  # n_samp
        "use_flag": bool,
        "flag_data": [...],  # n_samp
        "flag_mask": int,
        "amplitude_offset": int,
        "amplitude_view_offset": ["n_intervals"],
        "block_indices": ["blocks_per_interval"],
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
    },
    out_axes=(
        ["n_intervals", "blocks_per_interval"],
        ["n_intervals", "blocks_per_interval"],
    ),
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
    amplitude_flags,
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
        amplitude_flags (array, int): flags for each amplitude value (size n_amp)
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

    # get inputs
    det_data_indexed = det_data[data_index, :]
    flag_data_indexed = (
        flag_data[flag_index, :] if use_flag else jnp.empty_like(det_data_indexed)
    )
    amp_view_off = jnp.roll(n_amp_views, shift=1)
    amp_view_off = amp_view_off.at[0].set(0)
    # get number of step_length sized blocks per interval
    nb_blocks = 1 + (intervals_max_length - 1) // step_length
    block_indices = jnp.arange(start=0, stop=nb_blocks)

    # runs computation
    # NOTE: we work on blocks of size step_lengh (which will go to the same amplitude)
    #       we could simplify the code significantly by ignoring the block structure and using imap (exploiting `.add` being atomic)
    #       but it reduces performances significantly by creating contention on the atomic
    (amplitude_indices, contributions) = offset_project_signal_steplength_blocks(
        step_length,
        det_data_indexed,
        use_flag,
        flag_data_indexed,
        flag_mask,
        amp_offset,
        amp_view_off,
        block_indices,
        interval_starts,
        interval_ends,
    )

    # Mask out contributions where amplitude_flags are non-zero
    non_flagged_amplitudes = amplitude_flags[amplitude_indices] == 0
    contributions = jnp.where(non_flagged_amplitudes, contributions, 0.0)

    # updates det_data and returns
    # NOTE: add is atomic
    amplitudes = amplitudes.at[amplitude_indices].add(contributions)
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
    amplitude_flags,
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
        amplitude_flags (array, int): flags for each amplitude value (size n_amp)
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
    amplitude_flags = MutableJaxArray.to_array(amplitude_flags)
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
        amplitude_flags,
        amp_offset,
        n_amp_views,
        intervals.first,
        intervals.last,
        intervals_max_length,
    )


# ----------------------------------------------------------------------------------------
# offset_apply_diag_precond


def offset_apply_diag_precond_inner(
    offset_var, amplitudes_in, amplitude_flags, amplitudes_out
):
    """
    Simple multiplication with amplitude flags.

    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitude_flags (array, int): size n_amp
        amplitudes_out (array, double): size n_amp

    Returns:
        amplitudes_out (array, double): size n_amp
    """
    non_flagged_amplitudes = amplitude_flags == 0
    amplitudes_out = jnp.where(non_flagged_amplitudes, amplitudes_in * offset_var, 0.0)
    return amplitudes_out


# jit compilation
offset_apply_diag_precond_inner = jax.jit(
    offset_apply_diag_precond_inner,
    donate_argnums=[3],
)  # donate amplitudes_out


@kernel(impl=ImplementationType.JAX, name="offset_apply_diag_precond")
def offset_apply_diag_precond_jax(
    offset_var, amplitudes_in, amplitude_flags, amplitudes_out, use_accel
):
    """
    Simple multiplication with amplitude flags.

    Args:
        offset_var (array, double): size n_amp
        amplitudes_in (array, double): size n_amp
        amplitude_flags (array, int): size n_amp
        amplitudes_out (array, double): size n_amp
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in amplitudes_out).
    """
    # cast the data if needed
    offset_var = MutableJaxArray.to_array(offset_var)
    amplitudes_in = MutableJaxArray.to_array(amplitudes_in)
    amplitude_flags = MutableJaxArray.to_array(amplitude_flags)

    # runs the computation
    amplitudes_out[:] = offset_apply_diag_precond_inner(
        offset_var, amplitudes_in, amplitude_flags, amplitudes_out
    )


# To test:
# export TOAST_GPU_JAX=true; export TOAST_GPU_HYBRID_PIPELINES=true; export TOAST_LOGLEVEL=DEBUG; python -c 'import toast.tests; toast.tests.run("ops_pointing_wcs"); toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker");'
