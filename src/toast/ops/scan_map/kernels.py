# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from ... import qarray as qa
from ..._libtoast import ops_scan_map_float32 as libtoast_scan_map_float32
from ..._libtoast import ops_scan_map_float64 as libtoast_scan_map_float64
from ..._libtoast import ops_scan_map_int32 as libtoast_scan_map_int32
from ..._libtoast import ops_scan_map_int64 as libtoast_scan_map_int64
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
    data_scale,
    should_zero,
    should_subtract,
    should_scale,
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
    return libtoast_scan_map(
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
        use_accel=use_accel,
    )


@kernel(impl=ImplementationType.COMPILED, name="scan_map")
def scan_map_compiled(*args, use_accel=False):
    return libtoast_scan_map(*args, use_accel)


def libtoast_scan_map(
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

    wt_view = weights
    if len(weights.shape) == 2:
        # One value per sample, but our kernel always assumes a
        # 3D array.
        wdets = weights.shape[0]
        wsamps = weights.shape[1]
        wt_view = weights.reshape((wdets, wsamps, -1))

    fcomp(
        global2local,
        n_pix_submap,
        mapdata,
        det_data,
        det_data_index,
        pixels,
        pixels_index,
        wt_view,
        weight_index,
        intervals,
        data_scale,
        should_zero,
        should_subtract,
        should_scale,
        use_accel,
    )
