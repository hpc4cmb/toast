# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from .utils import assert_data_localization, dataMovementTracker, MutableJaxArray, select_implementation, ImplementationType
from .utils.intervals import JaxIntervals, ALL
from ..._libtoast import scan_map_float64 as scan_map_interval_float64_compiled, scan_map_float32 as scan_map_interval_float32_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def global_to_local_jax(global_pixels, npix_submap, global2local):
    """
    Convert global pixel indices to local submaps and pixels within the submap.

    Args:
        global_pixels (array):  The global pixel indices (size nsamples).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.

    Returns:
        (tuple of array):  The (local submap, pixel within submap) for each global pixel (each of size nsamples).
    """
    local_submaps = jnp.where(global_pixels < 0, -1, global_pixels % npix_submap)
    local_pixels = jnp.where(global_pixels < 0, -1, global2local[global_pixels // npix_submap])
    return (local_submaps, local_pixels)

def scan_map_inner_jax(mapdata, npix_submap, global2local, 
                       pixels, weights, det_data, 
                       should_zero, should_subtract):
    """
    Applies scan_map to a given interval.

    Args:
        mapdata (array, ?):  The local piece of the map (size ?*npix_submap*nmap).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.
        pixels (array, int): pixels (size nsample)
        weights (array, float64):  The pointing matrix weights (size: nsample*nmap).
        det_data (array, float64):  The timestream on which to accumulate the map values (size nsample).
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data

    Returns:
        det_data
    """
    # Get local submap and pixels
    submap, subpix = global_to_local_jax(pixels, npix_submap, global2local)

    # gets the local mapdata
    # zero-out samples with invalid indices
    # by default JAX will put any value where the indices were invalid instead of erroring out
    valid_samples = (subpix >= 0) & (submap >= 0)
    mapdata = jnp.where(valid_samples[:,jnp.newaxis], mapdata[submap,subpix,:], 0.0)

    # computes the update term
    update = jnp.sum(mapdata * weights, axis=1)

    # updates det_data and returns
    if should_zero: det_data = jnp.zeros_like(det_data)
    return (det_data - update) if should_subtract else (det_data + update)

# maps over intervals and detectors
scan_map_inner_jax = jax_xmap(scan_map_inner_jax, 
                              in_axes=[[...], # mapdata
                                       [...], # npix_submap
                                       [...], # global2local
                                       ['detectors','intervals',...], # pixels
                                       ['detectors','intervals',...], # weights
                                       ['detectors','intervals',...], # det_data
                                       [...], # should_zero
                                       [...]], # should_subtract
                              out_axes=['detectors','intervals',...])

def scan_map_interval_jax(mapdata,
                          nmap, npix_submap, global2local,
                          det_data, det_data_index,
                          pixels, pixels_index,
                          weights, weight_index,
                          interval_starts, interval_ends, intervals_max_length, 
                          should_zero, should_subtract):
    """
    Process all the intervals as a block.

    Args:
        mapdata (array, ?):  The local piece of the map (size ?*npix_submap*nmap).
        nmap (int): number of valid pixels
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
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data

    Returns:
        det_data (array, float): size ???*n_samp
    """
    # display sizes
    print(f"DEBUG: jit-compiling 'scan_map' with n_det:{det_data_index.size} nmap:{nmap} npix_submap:{npix_submap} n_view:{interval_starts.size} n_samp:{det_data.shape[-1]} intervals_max_length:{intervals_max_length} should_zero:{should_zero} should_subtract:{should_subtract}")

    # turns mapdata into a numpy array of shape ?*npix_submap*nmap
    mapdata = jnp.reshape(mapdata, newshape=(-1,npix_submap,nmap))

    # extract interval slices
    intervals = JaxIntervals(interval_starts, interval_ends+1, intervals_max_length) # end+1 as the interval is inclusive
    pixels_interval = JaxIntervals.get(pixels, (pixels_index,intervals)) # pixels[pixels_index, intervals]
    det_data_interval = JaxIntervals.get(det_data, (det_data_index,intervals)) # det_data[det_data_index, intervals]
    if weight_index is None:
        weights_interval = jnp.ones_like(pixels_interval)
    else:
        weights_interval = JaxIntervals.get(weights, (weight_index,intervals,ALL)) # weights[weight_index, intervals, :]

    # does the computation
    new_det_data_interval = scan_map_inner_jax(mapdata, npix_submap, global2local,
                                               pixels_interval, weights_interval, det_data_interval,
                                               should_zero, should_subtract)
    
    # updates results and returns
    # det_data[det_data_index, intervals] = new_det_data_interval
    det_data = JaxIntervals.set(det_data, (det_data_index,intervals), new_det_data_interval)
    return det_data

# jit compiling
scan_map_interval_jax = jax.jit(scan_map_interval_jax, 
                                static_argnames=['nmap', 'npix_submap', 'intervals_max_length', 'should_zero', 'should_subtract'],
                                donate_argnums=[4]) # donates det_data

def scan_map_jax(mapdata, nmap,
                 det_data, det_data_index,
                 pixels, pixels_index,
                 weights, weight_index,
                 intervals, map_dist, 
                 should_zero, should_subtract,
                 use_accel):
    """
    Sample a map into a timestream.

    This uses a distributed map and the pointing matrix
    to generate timestream values.

    Args:
        mapdata (Pixels):  The local piece of the map.
        nmap (int): number of valid pixels
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        pixels (array, int): pixels (size ???*n_samp)
        pixels_index (array, int): The indexes of the pixels (size n_det)
        weights (optional array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (optional array, int): The indexes of the weights (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        map_dist (PixelDistribution): encapsulate information to translate the pixel mapping
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    # extracts pixel distribution information
    npix_submap = map_dist._n_pix_submap
    global2local = map_dist._glob2loc
    # turns mapdata into a numpy array
    mapdata = mapdata.raw.array()

    # make sure the data is where we expect it
    assert_data_localization('scan_map', use_accel, [mapdata, global2local, det_data, det_data_index, pixels, pixels_index, weights, weight_index], [det_data])

    # prepares inputs
    intervals_max_length = np.max(1 + intervals.last - intervals.first) # end+1 as the interval is inclusive
    mapdata = MutableJaxArray.to_array(mapdata)
    global2local = MutableJaxArray.to_array(global2local)
    det_data_input = MutableJaxArray.to_array(det_data)
    det_data_index = MutableJaxArray.to_array(det_data_index)
    pixels = MutableJaxArray.to_array(pixels)
    pixels_index = MutableJaxArray.to_array(pixels_index)
    weights = MutableJaxArray.to_array(weights)
    weight_index = MutableJaxArray.to_array(weight_index)

    # track data movement
    dataMovementTracker.add("scan_map", use_accel, [mapdata, global2local, det_data_input, det_data_index, pixels, pixels_index, weights, weight_index, intervals.first, intervals.last], [det_data])

    # performs computation and updates det_data in place
    det_data[:] = scan_map_interval_jax(mapdata, nmap, npix_submap, global2local, 
                                        det_data_input, det_data_index, pixels, pixels_index, weights, weight_index,
                                        intervals.first, intervals.last, intervals_max_length, should_zero, should_subtract)

#-------------------------------------------------------------------------------------------------
# NUMPY

def global_to_local_numpy(global_pixels, npix_submap, global2local):
    """
    Convert global pixel indices to local submaps and pixels within the submap.

    Args:
        global_pixels (array):  The global pixel indices (size nsamples).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.

    Returns:
        (tuple of array):  The (local submap, pixel within submap) for each global pixel (each of size nsamples).

    NOTE: map_dist.global_pixel_to_submap(pixels) => global_to_local_numpy(pixels, map_dist._n_pix_submap, map_dist._glob2loc)
    """
    local_submaps = np.where(global_pixels < 0, -1, global_pixels % npix_submap)
    local_pixels = np.where(global_pixels < 0, -1, global2local[global_pixels // npix_submap])
    return (local_submaps, local_pixels)

def scan_map_interval_numpy(mapdata, npix_submap, global2local, 
                            pixels, weights, det_data, 
                            should_zero, should_subtract):
    """
    Applies scan_map to a given interval.

    Args:
        mapdata (array, ?):  The local piece of the map (size ?*npix_submap*nmap).
        npix_submap (int):  The number of pixels in each submap.
        global2local (array, int64):  The local submap for each global submap.
        pixels (array, int): pixels (size nsample)
        weights (array, float64):  The pointing matrix weights (size: nsample*nmap).
        det_data (array, float64):  The timestream on which to accumulate the map values (size nsample).
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data

    Returns:
        det_data
    """
    # Get local submap and pixels
    submap, subpix = global_to_local_numpy(pixels, npix_submap, global2local)

    # uses only samples with valid indices
    valid_samples = (subpix >= 0) & (submap >= 0)
    valid_weights = weights[valid_samples,:]
    valid_submap = submap[valid_samples]
    valid_subpix = subpix[valid_samples]
    valid_mapdata = mapdata[valid_submap,valid_subpix,:]

    # updates det_data
    # Note that the map scanned timestream will have
    # zeros anywhere that the pointing is bad, but those samples (and
    # any other detector flags) should be handled at other steps of the
    # processing.
    if should_zero:
        det_data[:] = 0.0
    if should_subtract:
        det_data[valid_samples] -= np.sum(valid_mapdata * valid_weights, axis=1)
    else:
        det_data[valid_samples] += np.sum(valid_mapdata * valid_weights, axis=1)
    return det_data

def scan_map_numpy(mapdata, nmap,
                   det_data, det_data_index,
                   pixels, pixels_index,
                   weights, weight_index,
                   intervals, map_dist, 
                   should_zero, should_subtract,
                   use_accel):
    """
    Sample a map into a timestream.

    This uses a distributed map and the pointing matrix
    to generate timestream values.

    Args:
        mapdata (Pixels):  The local piece of the map.
        nmap (int): number of valid pixels
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        pixels (array, int): pixels (size ???*n_samp)
        pixels_index (array, int): The indexes of the pixels (size n_det)
        weights (optional array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (optional array, int): The indexes of the weights (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        map_dist (PixelDistribution): encapsulate information to translate the pixel mapping
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    print(f"DEBUG: Running scan_map_numpy!")

    # extracts pixel distribution information
    # TODO remove assert if it is True, otherwise use second version for reshaping
    assert(map_dist._n_pix_submap == mapdata.distribution.n_pix_submap)
    npix_submap = map_dist._n_pix_submap
    global2local = map_dist._glob2loc
    # turns mapdata into a numpy array of shape ?*npix_submap*nmap
    mapdata = mapdata.raw.array()
    mapdata = np.reshape(mapdata, newshape=(-1,npix_submap,nmap))

    # iterates on detectors and intervals
    n_det = det_data_index.size
    for idet in range(n_det):
        p_index = pixels_index[idet]
        d_index = det_data_index[idet]
        w_index = None if (weight_index is None) else weight_index[idet]
        for interval in intervals:
            interval_start = interval['first']
            interval_end = interval['last']+1
            # gets interval data
            pixels_interval = pixels[p_index, interval_start:interval_end]
            weights_interval = np.ones_like(pixels_interval) if (weight_index is None) else weights[w_index, interval_start:interval_end, :]
            det_data_interval = det_data[d_index, interval_start:interval_end]
            # process the interval
            new_det_data_interval = scan_map_interval_numpy(mapdata, npix_submap, global2local,
                                                            pixels_interval, weights_interval, det_data_interval,
                                                            should_zero, should_subtract)
            det_data[d_index, interval_start:interval_end] = new_det_data_interval

#-------------------------------------------------------------------------------------------------
# C++

def scan_map_compiled(mapdata, nmap,
                      det_data, det_data_index,
                      pixels, pixels_index,
                      weights, weight_index,
                      intervals, map_dist, 
                      should_zero, should_subtract,
                      use_accel):
    """
    Sample a map into a timestream.

    This uses a distributed map and the pointing matrix
    to generate timestream values.

    Args:
        mapdata (Pixels):  The local piece of the map.
        nmap (int): number of valid pixels
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        pixels (array, int): pixels (size ???*n_samp)
        pixels_index (array, int): The indexes of the pixels (size n_det)
        weights (array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (array, int): The indexes of the weights (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        map_dist (PixelDistribution):
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place

    NOTE: Wraps implementations of scan_map to factor the datatype handling
    TODO: push the triple loop into the C++
    TODO: implement target offload version of the code and remove the RuntimeError
    """
    # raise an error as the C++ implementation does not implement target_offload yet
    if use_accel: raise RuntimeError("scan_map_compiled: there is no GPU-accelerated C++ backend yet.")

    # picks the correct implementation as a function of the data-type used
    if mapdata.dtype.char == 'd': scan_map_interval = scan_map_interval_float64_compiled
    elif mapdata.dtype.char == 'f': scan_map_interval = scan_map_interval_float32_compiled
    else: raise RuntimeError("Projection supports only float32 and float64 binned maps")

    # iterates on detectors and intervals
    n_det = det_data_index.size
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval['first']
            interval_end = interval['last']+1

            # gets interval pixels
            p_index = pixels_index[idet]
            pixels_interval = pixels[p_index, interval_start:interval_end]
            # gets interval weights
            if weight_index is None:
                weights_interval = np.ones_like(pixels_interval)
            else:
                w_index = weight_index[idet]
                weights_interval = weights[w_index, interval_start:interval_end, :]

            # Get local submap and pixels
            local_submap, local_pixels = map_dist.global_pixel_to_submap(pixels_interval)

            # runs the function (and flattens the weights)
            maptod = np.zeros_like(pixels_interval, dtype=det_data.dtype)
            scan_map_interval(mapdata.distribution.n_pix_submap, nmap, local_submap, local_pixels, mapdata.raw, weights_interval.reshape(-1), maptod)

            # Add or subtract to det_data, zero out if needed
            # Note that the map scanned timestream will have
            # zeros anywhere that the pointing is bad, but those samples (and
            # any other detector flags) should be handled at other steps of the
            # processing.
            d_index = det_data_index[idet]
            if should_zero:
                det_data[d_index, interval_start:interval_end] = 0.0
            if should_subtract:
                det_data[d_index, interval_start:interval_end] -= maptod
            else:
                det_data[d_index, interval_start:interval_end] += maptod

"""
template <typename T>
void global_to_local(size_t nsamp,
                     T const * global_pixels,
                     size_t npix_submap,
                     int64_t const * global2local,
                     T * local_submaps,
                     T * local_pixels) 
{
    double npix_submap_inv = 1.0 / static_cast <double> (npix_submap);

    // Note:  there is not much work in this loop, so it might benefit from
    // omp simd instead.  However, that would only be enabled if the input
    // memory buffers were aligned.  That could be ensured with care in the
    // calling code.  To be revisited if this code is ever the bottleneck.

    #pragma omp parallel for default(shared) schedule(static)
    for (size_t i = 0; i < nsamp; ++i) 
    {
        if (global_pixels[i] < 0) 
        {
            local_submaps[i] = -1;
            local_pixels[i] = -1;
        } 
        else 
        {
            local_pixels[i] = global_pixels[i] % npix_submap;
            local_submaps[i] = static_cast <T> (global2local[ static_cast <T> (static_cast <double> (global_pixels[i]) * npix_submap_inv)]);
        }
    }
}

template <typename T>
void scan_local_map(int64_t const * submap, int64_t subnpix, double const * weights,
                    int64_t nmap, int64_t * subpix, T const * map, double * tod,
                    int64_t nsamp) {
    // There is a single pixel vector valid for all maps.
    // The number of maps packed into "map" is the same as the number of weights
    // packed into "weights".
    //
    // The TOD is *NOT* set to zero, to allow accumulation.
    #pragma omp parallel for schedule(static) default(none) shared(submap, subnpix, weights, nmap, subpix, map, tod, nsamp)
    for (int64_t i = 0; i < nsamp; ++i) 
    {
        if ((subpix[i] < 0) || (submap[i] < 0)) 
        {
            continue;
        }
        int64_t offset = (submap[i] * subnpix + subpix[i]) * nmap;
        int64_t woffset = i * nmap;
        for (int64_t imap = 0; imap < nmap; ++imap) 
        {
            tod[i] += map[offset++] * weights[woffset++];
        }
    }

    return;
}
"""

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
scan_map = select_implementation(scan_map_compiled, 
                                 scan_map_numpy, 
                                 scan_map_jax,
                                 overide_implementationType=ImplementationType.NUMPY)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_scan_map")'

# to bench:
# use scanmap config and check ScanHealpixMap._exec field in timing.csv
# (or the full sky computation field in the shell output)
# one problem: this includes call to other operators

# 6 make use of use_accel in the operator