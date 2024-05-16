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
    replace_kernel_deps,
    clear_kernel_deps,
)


@kernel(impl=ImplementationType.OPENCL, name="build_noise_weighted")
def build_noise_weighted_opencl(
    global2local,
    zmap,
    pixels_index,
    pixels,
    weight_index,
    weights,
    det_data_index,
    det_data,
    flag_index,
    det_flags,
    det_scale,
    det_flag_mask,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    if len(shared_flags) == det_data.shape[1]:
        use_shared_flags = np.uint8(1)
    else:
        use_shared_flags = np.uint8(0)

    if len(det_flags) == det_data.shape[1]:
        use_det_flags = np.uint8(1)
    else:
        use_det_flags = np.uint8(0)

    ocl = OpenCL()
    queue = ocl.queue()
    devtype = ocl.default_device_type

    kernel = ocl.get_or_build_kernel(
        "mapmaker_utils",
        "build_noise_weighted",
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_global2local = ocl.mem(global2local, device_type=devtype)
    dev_zmap = ocl.mem(zmap, device_type=devtype)
    dev_pixels = ocl.mem(pixels, device_type=devtype)
    dev_weights = ocl.mem(weights, device_type=devtype)
    dev_det_data = ocl.mem(det_data, device_type=devtype)
    dev_det_flags = ocl.mem(det_flags, device_type=devtype)
    dev_shared_flags = ocl.mem(shared_flags, device_type=devtype)

    # Allocate temporary device arrays

    dev_pixels_index = ocl.mem_to_device(pixels_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_pixels_index.events)

    dev_weight_index = ocl.mem_to_device(weight_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_weight_index.events)

    dev_det_data_index = ocl.mem_to_device(
        det_data_index, device_type=devtype, async_=True
    )
    add_kernel_deps(state, obs_name, dev_det_data_index.events)

    dev_flag_index = ocl.mem_to_device(flag_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_flag_index.events)

    dev_det_scale = ocl.mem_to_device(det_scale, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_det_scale.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    print(f"BLDNSEW: {obs_name} got wait_for = {wait_for}", flush=True)
    print(f"BLDNSEW: {obs_name} pixels={dev_pixels}, weights={dev_weights}, zmap={dev_zmap}", flush=True)

    n_det = len(det_data_index)
    n_samp = weights.shape[1]
    nnz = weights.shape[2]
    n_pix_submap = zmap.shape[1]

    for intr in intervals:
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1
        ev = kernel(
            ocl.queue(device_type=devtype),
            (n_det, n_intr),
            None,
            np.int32(n_det),
            np.int64(n_samp),
            np.int64(first_sample),
            dev_pixels_index.data,
            dev_pixels.data,
            dev_weight_index.data,
            dev_weights.data,
            dev_det_data_index.data,
            dev_det_data.data,
            dev_flag_index.data,
            dev_det_flags.data,
            dev_shared_flags.data,
            dev_zmap.data,
            dev_global2local.data,
            dev_det_scale.data,
            np.int64(nnz),
            np.int64(n_pix_submap),
            np.uint8(det_flag_mask),
            np.uint8(shared_flag_mask),
            use_shared_flags,
            use_det_flags,
            wait_for=wait_for,
        )
        wait_for = [ev]
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)

    # Free temporaries
    ocl.mem_remove(pixels_index, device_type=devtype)
    ocl.mem_remove(weight_index, device_type=devtype)
    ocl.mem_remove(det_data_index, device_type=devtype)
    ocl.mem_remove(flag_index, device_type=devtype)
    ocl.mem_remove(det_scale, device_type=devtype)
