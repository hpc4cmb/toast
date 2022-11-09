# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .math import qarray, healpix


def pixels_healpix_inner(hpix, quats, nest):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        hpix (HPIX_NUMPY): Healpix projection object.
        quats (array, float64): Detector quaternion (size 4).
        hsub (array, uint8): The pointing flags (size ???).
        intervals (array, float64): size n_view
        n_pix_submap (array, float64):
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        pixels (array, int64): The detector pixel indices to store the result.
    """
    # constants
    zaxis = np.array([0.0, 0.0, 1.0])

    # initialize dir
    dir = qarray.rotate_one_one(quats, zaxis)

    # pixel computation
    (phi, region, z, rtz) = healpix.vec2zphi(dir)
    if nest:
        pixel = healpix.zphi2nest(hpix, phi, region, z, rtz)
    else:
        pixel = healpix.zphi2ring(hpix, phi, region, z, rtz)

    return pixel


def pixels_healpix(
    quat_index,
    quats,
    flags,
    flag_mask,
    pixel_index,
    pixels,
    intervals,
    hit_submaps,
    n_pix_submap,
    nside,
    nest,
    use_accel,
):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        quat_index (array, int): size n_det
        quats (array, float64): The flat-packed array of detector quaternions (size ???*n_samp*4).
        flags (array, uint8): size n_samp (or you shouldn't use flags)
        flag_mask (uint8)
        pixel_index (array, int): size n_det
        pixels (array, int64): The detector pixel indices to store the result (size ???*n_samp).
        intervals (array, float64): size n_view
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (array, float64):
        nside (int): Used to build the healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.
        use_accel (bool): should we use the accelerator

    Returns:
        None (results are stored in pixels and hit_submaps).
    """
    # constants
    hpix = healpix.HPIX_PYTHON(nside)
    n_samp = pixels.shape[1]
    use_flags = (flag_mask != 0) and (flags.size == n_samp)

    n_det = quat_index.size
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1
            for isamp in range(interval_start, interval_end):
                p_index = pixel_index[idet]
                q_index = quat_index[idet]
                is_flagged = (flags[isamp] & flag_mask) != 0
                if use_flags and is_flagged:
                    # masked pixel
                    pixels[p_index, isamp] = -1
                else:
                    # computes pixel value and saves it
                    pixel = pixels_healpix_inner(hpix, quats[q_index, isamp, :], nest)
                    pixels[p_index, isamp] = pixel
                    # modifies submap in place
                    sub_map = pixel // n_pix_submap
                    hit_submaps[sub_map] = 1


def _py_pixels_healpix(
    self,
    quat_indx,
    quat_data,
    flag_data,
    flag_mask,
    pix_indx,
    pix_data,
    intr_data,
    hit_submaps,
):
    """Internal python implementation for comparison tests."""
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    if self.nest:
        for idet in range(len(quat_indx)):
            qidx = quat_indx[idet]
            pidx = pix_indx[idet]
            for vw in intr_data:
                samples = slice(vw.first, vw.last + 1, 1)
                dir = qarray.rotate_one_one.rotate(quat_data[qidx][samples], zaxis)
                pix_data[pidx][samples] = self.hpix.vec2nest(dir)
                good = (flag_data[samples] & flag_mask) == 0
                bad = np.logical_not(good)
                sub_maps = pix_data[pidx][samples][good] // self._n_pix_submap
                hit_submaps[sub_maps] = 1
                pix_data[pidx][samples][bad] = -1
    else:
        for idet in range(len(quat_indx)):
            qidx = quat_indx[idet]
            pidx = pix_indx[idet]
            for vw in intr_data:
                samples = slice(vw.first, vw.last + 1, 1)
                dir = qarray.rotate_one_one.rotate(quat_data[qidx][samples], zaxis)
                pix_data[pidx][samples] = self.hpix.vec2ring(dir)
                good = (flag_data[samples] & flag_mask) == 0
                bad = np.logical_not(good)
                sub_maps = pix_data[pidx][samples][good] // self._n_pix_submap
                hit_submaps[sub_maps] = 1
                pix_data[pidx][samples][bad] = -1
