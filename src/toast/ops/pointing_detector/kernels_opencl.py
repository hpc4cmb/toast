# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
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


@kernel(impl=ImplementationType.OPENCL, name="pointing_detector")
def pointing_detector_opencl(
    focalplane,
    boresight,
    quat_index,
    quats,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    if len(shared_flags) == len(boresight):
        use_flags = np.uint8(1)
    else:
        use_flags = np.uint8(0)

    ocl = OpenCL()
    queue = ocl.queue()
    devtype = ocl.default_device_type

    # Get our kernel
    pointing_detector = ocl.get_or_build_kernel(
        "pointing_detector",
        "pointing_detector",
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_boresight = ocl.mem(boresight, device_type=devtype)
    dev_quats = ocl.mem(quats, device_type=devtype)
    if use_flags:
        dev_flags = ocl.mem(shared_flags, device_type=devtype)
    else:
        dev_flags = ocl.mem_null(shared_flags, device_type=devtype)

    # Allocate temporary arrays and copy to device
    dev_quat_index = ocl.mem_to_device(quat_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_quat_index.events)
    dev_fp = ocl.mem_to_device(focalplane, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_fp.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"PNTDET: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = len(quat_index)
    n_samp = quats.shape[1]
    for intr in intervals:
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1
        ev = pointing_detector(
            ocl.queue(device_type=devtype),
            (n_det, n_intr),
            None,
            np.int32(n_det),
            np.int64(n_samp),
            np.int64(first_sample),
            dev_fp.data,
            dev_boresight.data,
            dev_quat_index.data,
            dev_quats.data,
            dev_flags.data,
            np.uint8(shared_flag_mask),
            use_flags,
            wait_for=wait_for,
        )
        wait_for = [ev]
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)

    # Free temporaries
    ocl.mem_remove(quat_index, device_type=devtype)
    ocl.mem_remove(focalplane, device_type=devtype)
