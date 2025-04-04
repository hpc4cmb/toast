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


@kernel(impl=ImplementationType.OPENCL, name="scan_map")
def scan_map_opencl(
    global2local,
    n_pix_submap,
    mapdata,
    det_data,
    det_data_index,
    pixels,
    pixels_index,
    weights,
    weight_index,
    intervals,
    data_scale=1.0,
    should_zero=False,
    should_subtract=False,
    should_scale=False,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    # Select the kernel we will use based on input datatypes
    if mapdata.dtype.char == "d":
        kname = "scan_map_d_to_"
    elif mapdata.dtype.char == "f":
        kname = "scan_map_f_to_"
    elif mapdata.dtype.char == "i":
        kname = "scan_map_i_to_"
    elif mapdata.dtype.char == "l":
        kname = "scan_map_l_to_"
    else:
        msg = f"OpenCL version of scan_map does not support map "
        msg += f"dtype '{mapdata.dtype.char}'"
        raise NotImplementedError(msg)

    if det_data.dtype.char == "d":
        kname += "d"
    elif mapdata.dtype.char == "f":
        kname += "f"
    elif mapdata.dtype.char == "i":
        kname += "i"
    elif mapdata.dtype.char == "l":
        kname += "l"
    else:
        msg = f"OpenCL version of scan_map does not support det_data "
        msg += f"dtype '{det_data.dtype.char}'"
        raise NotImplementedError(msg)

    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    ocl = OpenCL()
    queue = ocl.queue()
    devtype = ocl.default_device_type

    scan_map_kernel = ocl.get_or_build_kernel(
        "scan_map",
        kname,
        device_type=devtype,
        source=program_file,
    )

    # Get our device arrays
    dev_global2local = ocl.mem(global2local, device_type=devtype)
    dev_mapdata = ocl.mem(mapdata, device_type=devtype)
    dev_pixels = ocl.mem(pixels, device_type=devtype)
    dev_weights = ocl.mem(weights, device_type=devtype)
    dev_det_data = ocl.mem(det_data, device_type=devtype)

    # Allocate temporary device arrays
    dev_pixels_index = ocl.mem_to_device(pixels_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_pixels_index.events)

    dev_weight_index = ocl.mem_to_device(weight_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_weight_index.events)

    dev_det_data_index = ocl.mem_to_device(
        det_data_index, device_type=devtype, async_=True
    )
    add_kernel_deps(state, obs_name, dev_det_data_index.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    # print(f"SCANMAP: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = len(det_data_index)
    n_samp = weights.shape[1]
    nnz = weights.shape[2]
    for intr in intervals:
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1
        ev = scan_map_kernel(
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
            dev_mapdata.data,
            dev_global2local.data,
            np.int64(nnz),
            np.int64(n_pix_submap),
            np.float64(data_scale),
            np.uint8(should_zero),
            np.uint8(should_subtract),
            np.uint8(should_scale),
            wait_for=wait_for,
        )
        wait_for = [ev]
    clear_kernel_deps(state, obs_name)
    add_kernel_deps(state, obs_name, wait_for)

    # Free temporaries
    ocl.mem_remove(pixels_index, device_type=devtype)
    ocl.mem_remove(weight_index, device_type=devtype)
    ocl.mem_remove(det_data_index, device_type=devtype)
