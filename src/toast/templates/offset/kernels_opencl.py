# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np
import pyopencl as cl

from ...accelerator import ImplementationType, kernel
from ...opencl import (
    find_source,
    OpenCL,
    add_kernel_deps,
    get_kernel_deps,
    clear_kernel_deps,
)


@kernel(impl=ImplementationType.OPENCL, name="offset_add_to_signal")
def offset_add_to_signal_opencl(
    step_length,
    amp_offset,
    n_amp_views,
    amplitudes,
    amplitude_flags,
    data_index,
    det_data,
    intervals,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    if len(amplitude_flags) == len(amplitudes):
        use_amp_flags = np.uint8(1)
    else:
        use_amp_flags = np.uint8(0)

    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    ocl = OpenCL()

    devtype = ocl.default_device_type

    kernel = ocl.get_or_build_kernel(
        "offset",
        "offset_add_to_signal",
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_amplitudes = ocl.mem(amplitudes, device_type=devtype)
    dev_amplitude_flags = ocl.mem(amplitude_flags, device_type=devtype)
    dev_det_data = ocl.mem(det_data, device_type=devtype)

    # No temporaries needed for this kernel.

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"OFFADD: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = det_data.shape[0]
    n_samp = det_data.shape[1]
    view_offset = amp_offset
    for intr, view_amps in zip(intervals, n_amp_views):
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1
        ev = kernel(
            ocl.queue(device_type=devtype),
            (n_det, n_intr),
            None,
            np.int64(n_samp),
            np.int64(first_sample),
            np.int64(step_length),
            np.int64(view_offset),
            dev_amplitudes.data,
            dev_amplitude_flags.data,
            data_index,
            dev_det_data.data,
            use_amp_flags,
            wait_for=wait_for,
        )
        wait_for = [ev]
        view_offset += view_amps
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)


@kernel(impl=ImplementationType.OPENCL, name="offset_project_signal")
def offset_project_signal_opencl(
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
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    if len(amplitude_flags) == len(amplitudes):
        use_amp_flags = np.uint8(1)
    else:
        use_amp_flags = np.uint8(0)
    if len(flag_data) == det_data.shape[1]:
        use_det_flags = np.uint8(1)
    else:
        use_det_flags = np.uint8(0)

    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    ocl = OpenCL()

    devtype = ocl.default_device_type

    kernel = ocl.get_or_build_kernel(
        "offset",
        "offset_project_signal",
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_amplitudes = ocl.mem(amplitudes, device_type=devtype)
    dev_amplitude_flags = ocl.mem(amplitude_flags, device_type=devtype)
    dev_det_data = ocl.mem(det_data, device_type=devtype)
    dev_flag_data = ocl.mem(flag_data, device_type=devtype)

    # No temporaries needed for this kernel.

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"OFFPROJ: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = det_data.shape[0]
    n_samp = det_data.shape[1]
    view_offset = amp_offset
    for intr, view_amps in zip(intervals, n_amp_views):
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1
        ev = kernel(
            ocl.queue(device_type=devtype),
            (n_det, n_intr),
            None,
            np.int64(n_samp),
            np.int64(first_sample),
            np.int32(data_index),
            dev_det_data.data,
            np.int32(flag_index),
            dev_flag_data.data,
            np.uint8(flag_mask),
            np.int64(step_length),
            np.int64(view_offset),
            dev_amplitudes.data,
            dev_amplitude_flags.data,
            use_det_flags,
            use_amp_flags,
            wait_for=wait_for,
        )
        wait_for = [ev]
        view_offset += view_amps
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)


@kernel(impl=ImplementationType.OPENCL, name="offset_apply_diag_precond")
def offset_apply_diag_precond_opencl(
    offset_var,
    amplitudes_in,
    amplitude_flags,
    amplitudes_out,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    if len(amplitude_flags) == len(amplitudes_in):
        use_amp_flags = np.uint8(1)
    else:
        use_amp_flags = np.uint8(0)

    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    ocl = OpenCL()
    queue = ocl.queue()
    devtype = ocl.default_device_type

    kernel = ocl.get_or_build_kernel(
        "offset",
        "offset_apply_diag_precond",
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_amplitudes_in = ocl.mem(amplitudes_in, device_type=devtype)
    dev_amplitude_flags = ocl.mem(amplitude_flags, device_type=devtype)
    dev_amplitudes_out = ocl.mem(amplitudes_out, device_type=devtype)

    # Allocate temporaries
    dev_offset_var = ocl.mem_to_device(offset_var, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_offset_var.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"OFFPREC: {obs_name} got wait_for = {wait_for}", flush=True)

    n_amp = len(amplitudes_in)
    ev = kernel(
        ocl.queue(device_type=devtype),
        (n_amp),
        None,
        dev_amplitudes_in.data,
        dev_amplitudes_out.data,
        dev_offset_var.data,
        dev_amplitude_flags.data,
        use_amp_flags,
        wait_for=wait_for,
    )
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, ev)
