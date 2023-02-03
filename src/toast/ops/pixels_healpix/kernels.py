# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..._libtoast import pixels_healpix as libtoast_pixels_healpix
from ...accelerator import kernel, ImplementationType, use_accel_jax

from ...healpix import Pixels
from ... import qarray as qa

if use_accel_jax:
    from .kernels_jax import pixels_healpix_jax


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
    use_accel=False,
):
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
    use_accel=False,
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
        hit_submaps (array):  Array of bytes to set to 1 if the submap is hit
            and zero if not hit.
        n_pix_submap (int):  The number of pixels in a submap.
        nside (int):  The Healpix NSIDE of the pixelization.
        nest (bool):  If true, use NESTED ordering, else use RING.        
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return pixels_healpix(
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
        impl=ImplementationType.COMPILED,
        use_accel=use_accel,
    )


@kernel(impl=ImplementationType.NUMPY, name="pixels_healpix")
def pixels_healpix_numpy(
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
    use_accel=False,
):
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    hpix = Pixels(nside)
    if nest:
        for idet in range(len(quat_index)):
            qidx = quat_index[idet]
            pidx = pixel_index[idet]
            for vw in intervals:
                samples = slice(vw.first, vw.last + 1, 1)
                dir = qa.rotate(quats[qidx][samples], zaxis)
                pixels[pidx][samples] = hpix.vec2nest(dir)
                good = (shared_flags[samples] & shared_flag_mask) == 0
                bad = np.logical_not(good)
                sub_maps = pixels[pidx][samples][good] // n_pix_submap
                hit_submaps[sub_maps] = 1
                pixels[pidx][samples][bad] = -1
    else:
        for idet in range(len(quat_index)):
            qidx = quat_index[idet]
            pidx = pixel_index[idet]
            for vw in intervals:
                samples = slice(vw.first, vw.last + 1, 1)
                dir = qa.rotate(quats[qidx][samples], zaxis)
                pixels[pidx][samples] = hpix.vec2ring(dir)
                good = (shared_flags[samples] & shared_flag_mask) == 0
                bad = np.logical_not(good)
                sub_maps = pixels[pidx][samples][good] // n_pix_submap
                hit_submaps[sub_maps] = 1
                pixels[pidx][samples][bad] = -1

