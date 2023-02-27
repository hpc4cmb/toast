# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from ... import qarray as qa
from ...accelerator import ImplementationType, kernel


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
    for idet in range(len(quat_index)):
        qidx = quat_index[idet]
        pidx = pixel_index[idet]
        for vw in intervals:
            samples = slice(vw.first, vw.last + 1, 1)
            dir = qa.rotate(quats[qidx][samples], zaxis)
            pixels[pidx][samples] = hp.vec2pix(
                nside,
                dir[:, 0],
                dir[:, 1],
                dir[:, 2],
                nest,
            )
            good = (shared_flags[samples] & shared_flag_mask) == 0
            bad = np.logical_not(good)
            sub_maps = pixels[pidx][samples][good] // n_pix_submap
            hit_submaps[sub_maps] = 1
            pixels[pidx][samples][bad] = -1
