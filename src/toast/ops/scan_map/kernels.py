# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from ... import qarray as qa
from ..._libtoast import AlignedF64, AlignedI64
from ..._libtoast import scan_map_float32 as libtoast_scan_map_float32
from ..._libtoast import scan_map_float64 as libtoast_scan_map_float64
from ..._libtoast import scan_map_int32 as libtoast_scan_map_int32
from ..._libtoast import scan_map_int64 as libtoast_scan_map_int64
from ...accelerator import ImplementationType, kernel, use_accel_jax
from .kernels_numpy import scan_map_numpy

if use_accel_jax:
    from .kernels_jax import scan_map_jax


@kernel(impl=ImplementationType.DEFAULT)
def scan_map(
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
    """Kernel for scanning a map into timestreams.

    This uses a local piece of a distributed map and the pointing matrix for local
    detectors to generate timestream values.

    Args:
        global2local (array):  The mapping from global submap to local submap index.
        n_pix_submap (int):  The number of pixels per submap.
        mapdata (array):  The local piece of the map.
        det_data (array):  The detector data at each sample for each detector.
        data_index (array):  The index into the data array for each detector.
        pixels (array):  The array of detector pixels for each sample.
        pixel_index (array):  The index into the detector pixel array for each
            detector.
        weights (array):  The array of I, Q, and U weights at each sample for each
            detector.
        weight_index (array):  The index into the weights array for each detector.
        intervals (array):  The array of sample intervals.
        data_scale (float):  Scale factor to apply to map data before accumulation.
        should_zero (bool):  If True, set the detector data to zero before accumulation.
        should_subtract (bool):  If True, subtract rather than add to detector data.
        should_scale (bool):  If True, scale detector data by the scanned map values.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None.

    """
    return scan_map_compiled(
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
        data_scale,
        should_zero,
        should_subtract,
        should_scale,
        use_accel,
    )


# FIXME:  This "compiled" kernel will migrate fully to the _libtoast extension.


@kernel(impl=ImplementationType.COMPILED, name="scan_map")
def scan_map_compiled(
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
    if mapdata.dtype.char == "d":
        fcomp = libtoast_scan_map_float64
    elif mapdata.dtype.char == "f":
        fcomp = libtoast_scan_map_float32
    elif mapdata.dtype.char == "i":
        fcomp = libtoast_scan_map_int32
    elif mapdata.dtype.char == "l":
        fcomp = libtoast_scan_map_int64
    else:
        msg = f"Compiled version of scan_map does not support array "
        msg += f"type '{mapdata.dtype.char}'"
        raise NotImplementedError(msg)

    nmap = 1
    if len(weights.shape) > 2:
        nmap = weights.shape[2]
    for idet in range(len(det_data_index)):
        pix = pixels[pixels_index[idet]]
        wts = weights[weight_index[idet]]
        tod = det_data[det_data_index[idet]]
        for view in intervals:
            view_samples = view.last + 1 - view.first
            vslice = slice(view.first, view.last + 1)

            # Temp buffers
            local_tod_raw = AlignedF64.zeros(view_samples)
            local_tod = local_tod_raw.array()
            local_submap_pix_raw = AlignedI64.zeros(view_samples)
            local_submap_pix = local_submap_pix_raw.array()
            local_submap_pix[:] = -1
            local_submap_raw = AlignedI64.zeros(view_samples)
            local_submap = local_submap_raw.array()
            local_submap[:] = -1

            # Get local submaps and pixel indices within each submap
            good = pix[vslice] >= 0
            local_submap_pix[good] = pix[vslice][good] % n_pix_submap
            local_submap[good] = global2local[pix[vslice][good] // n_pix_submap]

            fcomp(
                n_pix_submap,
                nmap,
                local_submap,
                local_submap_pix,
                mapdata.reshape((-1,)),
                wts[vslice].reshape((-1,)),
                local_tod,
            )

            # Accumulate
            if should_zero:
                tod[vslice] = 0
            if should_subtract:
                tod[vslice][good] -= local_tod[good]
            elif should_scale:
                tod[vslice][good] *= local_tod[good]
            else:
                tod[vslice][good] += local_tod[good]

            del local_submap
            del local_submap_raw
            del local_submap_pix
            del local_submap_pix_raw
            del local_tod
            del local_tod_raw
