# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp

from ...accelerator import ImplementationType, kernel
from ...jax.intervals import INTERVALS_JAX
from ...jax.maps import imap
from ...jax.mutableArray import MutableJaxArray
from ...utils import Logger


def global_to_local(global_pixels, npix_submap, global2local):
    """
    Convert global pixel indices to local submaps and pixels within the submap.

    Args:
        global_pixels (int):  The global pixel indice.
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.

    Returns:
        (local_submaps, local_pixels) (int,int): (local submap, pixel within submap) for each global pixel.
    """
    quotient, remainder = jnp.divmod(global_pixels, npix_submap)
    local_pixels = jnp.where(global_pixels < 0, -1, remainder.astype(int))
    local_submaps = jnp.where(global_pixels < 0, -1, global2local[quotient.astype(int)])
    return (local_submaps, local_pixels)


def scan_map_inner(
    mapdata,
    npix_submap,
    global2local,
    det_data,
    pixels,
    weights,
    data_scale,
    should_zero,
    should_subtract,
    should_scale,
):
    """
    Applies scan_map to a given interval.

    Args:
        mapdata (array, ?):  The local piece of the map (size ?*npix_submap*nmap).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.
        det_data (float64):  The timestream on which to accumulate the map values.
        pixels (int): pixels
        weights (array, float64):  The pointing matrix weights (size: nmap).
        data_scale (float): unit rescaling
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        should_scale (bool): should we scale the detector data by the map values

    Returns:
        det_data (float64)
    """
    # Get local submap and pixels
    submap, subpix = global_to_local(pixels, npix_submap, global2local)
    # by default JAX will put any value where the indices were invalid instead of erroring out
    mapdata = mapdata[submap, subpix, :]

    # computes the update term
    update = jnp.sum(mapdata * weights) * data_scale

    # removes potentially useless dimension inroduced by imap
    det_data = jnp.squeeze(det_data)
    # updates det_data
    if should_zero:
        det_data = jnp.zeros_like(det_data)
    if should_subtract:
        new_det_data = det_data - update
    elif should_scale:
        new_det_data = det_data * update
    else:
        new_det_data = det_data + update

    # mask invalid values and returns
    valid_samples = (subpix >= 0) & (submap >= 0)
    return jnp.where(valid_samples, new_det_data, det_data)


# maps over intervals and detectors
scan_map_inner = imap(
    scan_map_inner,
    in_axes={
        "mapdata": [..., ..., ...],
        "npix_submap": int,
        "global2local": [...],
        "det_data": ["n_det", "n_samp"],
        "pixels": ["n_det", "n_samp"],
        "weights": ["n_det", "n_samp", ...],
        "data_scale": float,
        "should_zero": bool,
        "should_subtract": bool,
        "should_scale": bool,
        "interval_starts": ["n_intervals"],
        "interval_ends": ["n_intervals"],
        "intervals_max_length": int,
    },
    interval_axis="n_samp",
    interval_starts="interval_starts",
    interval_ends="interval_ends",
    interval_max_length="intervals_max_length",
    output_name="det_data",
    output_as_input=True,
)


def scan_map_interval(
    mapdata,
    npix_submap,
    global2local,
    det_data,
    det_data_index,
    pixels,
    pixels_index,
    weights,
    weight_index,
    interval_starts,
    interval_ends,
    intervals_max_length,
    data_scale,
    should_zero,
    should_subtract,
    should_scale,
):
    """
    Process all the intervals as a block.

    Args:
        mapdata (array, ?):  The local piece of the map (size ?*npix_submap*nmap).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        pixels (array, int): pixels (size ???*n_samp)
        pixels_index (array, int): The indexes of the pixels (size n_det)
        weights (optional array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (optional array, int): The indexes of the weights (size n_det)
        interval_starts (array, int): size n_view
        interval_ends (array, int): size n_view
        intervals_max_length (int): maximum length of an interval
        data_scale (float): unit scaling
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        should_scale (bool): should we scale the detector data by the map values

    Returns:
        det_data (array, float): size ???*n_samp
    """
    # debugging information
    log = Logger.get()
    log.debug(f"scan_map: jit-compiling.")

    # extract indexes
    pixels_indexed = pixels[pixels_index, :]
    det_data_indexed = det_data[det_data_index, :]
    if weight_index is None:
        weights_indexed = jnp.ones_like(pixels_indexed)[:, :, jnp.newaxis]
    elif weights.ndim == 2:
        weights_indexed = weights[weight_index, :, jnp.newaxis]
    else:
        weights_indexed = weights[weight_index, :, :]

    # does the computation
    new_det_data_indexed = scan_map_inner(
        mapdata,
        npix_submap,
        global2local,
        det_data_indexed,
        pixels_indexed,
        weights_indexed,
        data_scale,
        should_zero,
        should_subtract,
        should_scale,
        interval_starts,
        interval_ends,
        intervals_max_length,
    )

    # updates results and returns
    det_data = det_data.at[det_data_index, :].set(new_det_data_indexed)
    return det_data


# jit compiling
scan_map_interval = jax.jit(
    scan_map_interval,
    static_argnames=[
        "npix_submap",
        "intervals_max_length",
        "should_zero",
        "should_subtract",
        "should_scale",
    ],
    donate_argnums=[3],
)  # donates det_data


@kernel(impl=ImplementationType.JAX, name="scan_map")
def scan_map_jax(
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
    """
    Kernel for scanning a map into timestreams.

    This uses a local piece of a distributed map and the pointing matrix for local
    detectors to generate timestream values.

    Args:
        global2local (array):  The mapping from global submap to local submap index.
        n_pix_submap (int):  The number of pixels per submap.
        mapdata (array, ?):  The local piece of the map.
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        pixels (array, int): pixels (size ???*n_samp)
        pixels_index (array, int): The indexes of the pixels (size n_det)
        weights (optional array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (optional array, int): The indexes of the weights (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        data_scale (float): unit scalling
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        should_scale (bool): should we scale
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    # prepares inputs
    intervals_max_length = INTERVALS_JAX.compute_max_intervals_length(intervals)
    mapdata = MutableJaxArray.to_array(mapdata)
    global2local = MutableJaxArray.to_array(global2local)
    det_data_input = MutableJaxArray.to_array(det_data)
    det_data_index = MutableJaxArray.to_array(det_data_index)
    pixels = MutableJaxArray.to_array(pixels)
    pixels_index = MutableJaxArray.to_array(pixels_index)
    weights = MutableJaxArray.to_array(weights)
    weight_index = MutableJaxArray.to_array(weight_index)

    # performs computation and updates det_data in place
    det_data[:] = scan_map_interval(
        mapdata,
        n_pix_submap,
        global2local,
        det_data_input,
        det_data_index,
        pixels,
        pixels_index,
        weights,
        weight_index,
        intervals.first,
        intervals.last,
        intervals_max_length,
        data_scale,
        should_zero,
        should_subtract,
        should_scale,
    )


# To test:
# export TOAST_GPU_JAX=true; export TOAST_GPU_HYBRID_PIPELINES=true; export TOAST_LOGLEVEL=DEBUG; python -c 'import toast.tests; toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_scan_map");'
