# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .math import qarray, healpix


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
    zaxis = np.array([0, 0, 1], dtype=np.float64)

    n_det = quat_index.size
    for idet in range(n_det):
        p_index = pixel_index[idet]
        q_index = quat_index[idet]
        for interval in intervals:
            samples = slice(interval.first, interval.last + 1, 1)
            dir = qarray.rotate(quats[q_index][samples], zaxis)
            # loops as the healpix operations are not vectorised to run on a batch of quaternions
            for isample in range(interval.first, interval.last + 1):
                (phi, region, z, rtz) = healpix.vec2zphi(dir[isample])
                pixels[p_index][isample] = (
                    healpix.zphi2nest(hpix, phi, region, z, rtz)
                    if nest
                    else healpix.zphi2ring(hpix, phi, region, z, rtz)
                )
            good = (flags[samples] & flag_mask) == 0
            bad = np.logical_not(good)
            sub_maps = pixels[p_index][samples][good] // n_pix_submap
            hit_submaps[sub_maps] = 1
            pixels[p_index][samples][bad] = -1


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix"); toast.tests.run("ops_sim_ground"); toast.tests.run("ops_sim_satellite"); toast.tests.run("ops_demodulate"); toast.tests.run("ops_sim_tod_conviqt");'
