# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import healpy as hp
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


utab = None
ctab = None

# Initialize these constant arrays on host
if utab is None:
    utab = np.zeros(0x100, dtype=np.int64)
    ctab = np.zeros(0x100, dtype=np.int64)
    for m in range(0x100):
        utab[m] = (
            (m & 0x1)
            | ((m & 0x2) << 1)
            | ((m & 0x4) << 2)
            | ((m & 0x8) << 3)
            | ((m & 0x10) << 4)
            | ((m & 0x20) << 5)
            | ((m & 0x40) << 6)
            | ((m & 0x80) << 7)
        )
        ctab[m] = (
            (m & 0x1)
            | ((m & 0x2) << 7)
            | ((m & 0x4) >> 1)
            | ((m & 0x8) << 6)
            | ((m & 0x10) >> 2)
            | ((m & 0x20) << 5)
            | ((m & 0x40) >> 3)
            | ((m & 0x80) << 4)
        )


@kernel(impl=ImplementationType.OPENCL, name="pixels_healpix")
def pixels_healpix_opencl(
    quat_index,
    quats,
    shared_flags,
    shared_flag_mask,
    pixel_index,
    pixels,
    intervals,
    hit_submaps,
    n_pix_submap,
    nside,
    nest,
    compute_submaps,
    use_accel=False,
    obs_name=None,
    state=None,
    **kwargs,
):
    program_file = find_source(os.path.dirname(__file__), "kernels_opencl.cl")

    if len(shared_flags) == quats.shape[1]:
        use_flags = np.uint8(1)
    else:
        use_flags = np.uint8(0)

    ocl = OpenCL()
    queue = ocl.queue()
    devtype = ocl.default_device_type

    # Make sure that our small helper arrays are staged.  These are persistent
    # and we do not delete them until the program terminates.
    if not ocl.mem_present(utab, name="utab", device_type=devtype):
        dev_utab = ocl.mem_create(utab, name="utab", device_type=devtype)
        ocl.mem_update_device(utab, name="utab", device_type=devtype)
    else:
        dev_utab = ocl.mem(utab, name="utab", device_type=devtype)

    if not ocl.mem_present(ctab, name="ctab", device_type=devtype):
        dev_ctab = ocl.mem_create(ctab, name="ctab", device_type=devtype)
        ocl.mem_update_device(ctab, name="ctab", device_type=devtype)
    else:
        dev_ctab = ocl.mem(ctab, name="ctab", device_type=devtype)

    factor = 0
    while nside != 1 << factor:
        factor += 1

    if nest:
        pixels_healpix = ocl.get_or_build_kernel(
            "pixels_healpix",
            "pixels_healpix_nest",
            device_type=devtype,
            source=program_file,
        )
    else:
        pixels_healpix = ocl.get_or_build_kernel(
            "pixels_healpix",
            "pixels_healpix_ring",
            device_type=devtype,
            source=program_file,
        )

    # Get our device arrays
    dev_pixels = ocl.mem(pixels, device_type=devtype)
    dev_quats = ocl.mem(quats, device_type=devtype)
    if compute_submaps:
        dev_hit_submaps = ocl.mem(hit_submaps, device_type=devtype)
    else:
        dev_hit_submaps = ocl.mem_null(hit_submaps, device_type=devtype)
    if use_flags:
        dev_flags = ocl.mem(shared_flags, device_type=devtype)
    else:
        dev_flags = ocl.mem_null(shared_flags, device_type=devtype)

    # Allocate temporary device arrays
    dev_quat_index = ocl.mem_to_device(quat_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_quat_index.events)
    dev_pixel_index = ocl.mem_to_device(pixel_index, device_type=devtype, async_=True)
    add_kernel_deps(state, obs_name, dev_pixel_index.events)

    # All of the events that our kernels depend on
    wait_for = get_kernel_deps(state, obs_name)
    print(f"PIXHPX: {obs_name} got wait_for = {wait_for}", flush=True)

    n_det = len(pixel_index)
    n_samp = quats.shape[1]
    for intr in intervals:
        first_sample = intr.first
        n_intr = intr.last - intr.first + 1
        ev = pixels_healpix(
            ocl.queue(device_type=devtype),
            (n_det, n_intr),
            None,
            np.int32(n_det),
            np.int64(n_samp),
            np.int64(first_sample),
            np.int64(nside),
            np.int64(factor),
            np.int64(n_pix_submap),
            dev_utab.data,
            dev_quat_index.data,
            dev_quats.data,
            dev_pixel_index.data,
            dev_pixels.data,
            dev_hit_submaps.data,
            dev_flags.data,
            np.uint8(shared_flag_mask),
            use_flags,
            np.uint8(compute_submaps),
            wait_for=wait_for,
        )
        wait_for = [ev]
    clear_kernel_deps(state, obs_name)
    print(f"PIXHPX: {obs_name} export wait_for = {wait_for}", flush=True)
    add_kernel_deps(state, obs_name, wait_for)

    # Free temporaries
    ocl.mem_remove(quat_index, device_type=devtype)
    ocl.mem_remove(pixel_index, device_type=devtype)
