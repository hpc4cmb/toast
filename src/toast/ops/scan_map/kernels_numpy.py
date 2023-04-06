# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ...accelerator import ImplementationType, kernel


@kernel(impl=ImplementationType.NUMPY, name="scan_map")
def scan_map_numpy(
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
):
    nmap = weights.shape[-1]
    local_map = mapdata.reshape((-1, nmap))
    for idet in range(len(det_data_index)):
        pix = pixels[pixels_index[idet]]
        wts = weights[weight_index[idet]]
        tod = det_data[det_data_index[idet]]
        for view in intervals:
            vslice = slice(view.first, view.last + 1)

            # Get local submaps and pixel indices within each submap
            good = pix[vslice] >= 0
            local_submap_pix = pix[vslice] % n_pix_submap
            local_submap = global2local[pix[vslice] // n_pix_submap]

            # Indices into the local buffer
            map_pixels = local_submap * n_pix_submap + local_submap_pix

            # Local scanned TOD
            local_tod = data_scale * np.sum(
                local_map[map_pixels[good]] * wts[vslice][good], axis=1
            )

            # Accumulate
            if should_zero:
                tod[vslice] = 0
            if should_subtract:
                tod[vslice][good] -= local_tod
            elif should_scale:
                tod[vslice][good] *= local_tod
            else:
                tod[vslice][good] += local_tod
