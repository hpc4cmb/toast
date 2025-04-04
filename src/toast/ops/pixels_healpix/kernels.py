# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from ... import qarray as qa
from ..._libtoast import pixels_healpix as libtoast_pixels_healpix
from ...accelerator import ImplementationType, kernel, use_accel_jax, use_accel_opencl
from .kernels_numpy import pixels_healpix_numpy

if use_accel_jax:
    from .kernels_jax import pixels_healpix_jax

if use_accel_opencl:
    from .kernels_opencl import pixels_healpix_opencl


@kernel(impl=ImplementationType.COMPILED, name="pixels_healpix")
def pixels_healpix_compiled(
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
    **kwargs,
):
    if hit_submaps is None:
        hit_submaps = np.zeros(1, dtype=np.uint8)
    return libtoast_pixels_healpix(
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
        use_accel,
    )


@kernel(impl=ImplementationType.DEFAULT)
def pixels_healpix(
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
    **kwargs,
):
    """Kernel for computing healpix pixelization.

    Args:
        quat_index (array):  The index into the detector quaternion array for each
            detector.
        quats (array):  The array of detector quaternions for each sample.
        shared_flags (array):  The array of common flags for each sample.
        shared_flag_mask (int):  The flag mask to apply.
        pixel_index (array):  The index into the detector pixel array for each
            detector.
        pixels (array):  The array of detector pixels for each sample.
        intervals (array):  The array of sample intervals.
        hit_submaps (device array):  Array of bytes to set to 1 if the submap is hit
            and zero if not hit.
        n_pix_submap (int):  The number of pixels in a submap.
        nside (int):  The Healpix NSIDE of the pixelization.
        nest (bool):  If true, use NESTED ordering, else use RING.
        compute_submaps (bool):  If True, compute the hit submaps.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    if hit_submaps is None:
        hit_submaps = np.zeros(1, dtype=np.uint8)
    return libtoast_pixels_healpix(
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
        use_accel,
    )
