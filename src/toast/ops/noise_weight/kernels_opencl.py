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


@kernel(impl=ImplementationType.OPENCL, name="noise_weight")
def noise_weight_opencl(
    det_data,
    det_data_index,
    intervals,
    detector_weights,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    ocl = OpenCL()
    queue = ocl.queue()
    devtype = ocl.default_device_type

    # Get our kernel
    noise_weight = ocl.get_or_build_kernel(
        "noise_weight",
        "noise_weight",
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_det_data = ocl.mem(det_data, device_type=devtype)

    # Allocate temporary arrays and copy to device
    dev_det_data_index = ocl.mem_to_device(
        det_data_index, device_type=devtype, async_=True
    )
    add_kernel_deps(state, obs_name, dev_det_data_index.events)
    dev_det_weights = ocl.mem_to_device(
        detector_weights, device_type=devtype, async_=True
    )
    add_kernel_deps(state, obs_name, dev_det_weights.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"NSEWEIGHT: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = len(det_data_index)
    n_samp = det_data.shape[1]
    for intr in intervals:
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1
        ev = noise_weight(
            ocl.queue(device_type=devtype),
            (n_det, n_intr),
            None,
            np.int32(n_det),
            np.int64(n_samp),
            np.int64(first_sample),
            dev_det_weights.data,
            dev_det_data_index.data,
            dev_det_data.data,
            wait_for=wait_for,
        )
        wait_for = [ev]
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)

    # Free temporaries
    ocl.mem_remove(det_data_index, device_type=devtype)
    ocl.mem_remove(detector_weights, device_type=devtype)
