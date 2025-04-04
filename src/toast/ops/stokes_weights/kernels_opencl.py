# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np
import pyopencl as cl

from ... import qarray as qa
from ...accelerator import ImplementationType, kernel
from ...opencl import (
    find_source,
    OpenCL,
    add_kernel_deps,
    get_kernel_deps,
    clear_kernel_deps,
)


@kernel(impl=ImplementationType.OPENCL, name="stokes_weights_IQU")
def stokes_weights_IQU_opencl(
    quat_index,
    quats,
    weight_index,
    weights,
    hwp,
    intervals,
    epsilon,
    gamma,
    cal,
    IAU,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    if hwp is not None and len(hwp) != len(quats):
        hwp = None

    if IAU:
        U_sign = -1.0
    else:
        U_sign = 1.0

    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    ocl = OpenCL()

    devtype = ocl.default_device_type

    if hwp is None:
        stokes_weights_IQU = ocl.get_or_build_kernel(
            "stokes_weights",
            "stokes_weights_IQU",
            device_type=devtype,
            source=program_file,
        )
    else:
        stokes_weights_IQU = ocl.get_or_build_kernel(
            "stokes_weights",
            "stokes_weights_IQU_hwp",
            device_type=devtype,
            source=program_file,
        )

    # Get our device arrays
    dev_quats = ocl.mem(quats, device_type=devtype)
    dev_weights = ocl.mem(weights, device_type=devtype)
    if hwp is not None:
        dev_hwp = ocl.mem(hwp, device_type=devtype)

    # Allocate temporary device arrays
    dev_quat_index = ocl.mem_to_device(quat_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_quat_index.events)

    dev_weight_index = ocl.mem_to_device(weight_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_weight_index.events)

    dev_epsilon = ocl.mem_to_device(epsilon, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_epsilon.events)

    dev_gamma = ocl.mem_to_device(gamma, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_gamma.events)

    dev_cal = ocl.mem_to_device(cal, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_cal.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"STOKESIQU: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = len(quat_index)
    n_samp = quats.shape[1]
    for intr in intervals:
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1

        if hwp is None:
            ev = stokes_weights_IQU(
                ocl.queue(device_type=devtype),
                (n_det, n_intr),
                None,
                np.int32(n_det),
                np.int64(n_samp),
                np.int64(first_sample),
                dev_quat_index.data,
                dev_quats.data,
                dev_weight_index.data,
                dev_weights.data,
                dev_epsilon.data,
                dev_gamma.data,
                dev_cal.data,
                np.float64(U_sign),
                np.uint8(IAU),
                wait_for=wait_for,
            )
            wait_for = [ev]
        else:
            ev = stokes_weights_IQU(
                ocl.queue(device_type=devtype),
                (n_det, n_intr),
                None,
                np.int32(n_det),
                np.int64(n_samp),
                np.int64(first_sample),
                dev_quat_index.data,
                dev_quats.data,
                dev_weight_index.data,
                dev_weights.data,
                dev_hwp.data,
                dev_epsilon.data,
                dev_gamma.data,
                dev_cal.data,
                np.float64(U_sign),
                np.uint8(IAU),
                wait_for=wait_for,
            )
            wait_for = [ev]
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)

    # Free temporaries
    ocl.mem_remove(quat_index, device_type=devtype)
    ocl.mem_remove(weight_index, device_type=devtype)
    ocl.mem_remove(epsilon, device_type=devtype)
    ocl.mem_remove(gamma, device_type=devtype)
    ocl.mem_remove(cal, device_type=devtype)


@kernel(impl=ImplementationType.OPENCL, name="stokes_weights_I")
def stokes_weights_I_opencl(
    weight_index,
    weights,
    intervals,
    cal,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    ocl = OpenCL()
    queue = ocl.queue()
    devtype = ocl.default_device_type

    stokes_weights_I = ocl.get_or_build_kernel(
        "stokes_weights",
        "stokes_weights_I",
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_weights = ocl.mem(weights, device_type=devtype)

    # Allocate temporary device arrays

    dev_weight_index = ocl.mem_to_device(weight_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_weight_index.events)

    dev_cal = ocl.mem_to_device(cal, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_cal.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"STOKESI: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = len(weight_index)
    for intr in intervals:
        first_sample = intr.first
        n_samp = intr.last - intr.first + 1
        ev = stokes_weights_I(
            ocl.queue(device_type=devtype),
            (n_det, n_samp),
            None,
            np.int32(n_det),
            np.int64(n_samp),
            np.int64(first_sample),
            dev_weight_index.data,
            dev_weights.data,
            dev_cal.data,
            wait_for=wait_for,
        )
        wait_for = [ev]
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)

    # Free temporaries
    ocl.mem_remove(weight_index, device_type=devtype)
    ocl.mem_remove(cal, device_type=devtype)
