# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import select_implementation, ImplementationType
from .utils import MutableJaxArray
from ..._libtoast import scan_map_float64 as scan_map_interval_float64_compiled, scan_map_float32 as scan_map_interval_float32_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def scan_map_jitted(mapdata, npix_submap, nmap, submap, subpix, weights):
    """
    takes mapdata as a jax array and npix_submap as an integer
    """
    # display size information for debugging purposes
    print(f"DEBUG: jit compiling 'scan_map' npix_submap:{npix_submap} nsamp:{submap.size} nmap:{nmap} mapdata:{mapdata.shape} weights:{weights.shape}")

    # turns mapdata into an array of shape nsamp*nmap
    mapdata = jnp.reshape(mapdata, newshape=(-1,npix_submap,nmap))
    mapdata = mapdata[submap,subpix,:]

    # zero-out samples with invalid indices
    # by default JAX will put any value where the indices were invalid instead of erroring out
    valid_samples = (subpix >= 0) & (submap >= 0)
    mapdata = jnp.where(valid_samples[:,jnp.newaxis], mapdata, 0.0)

    # does the computation
    return jnp.sum(mapdata * weights, axis=1)

# Jit compiles the function
scan_map_jitted = jax.jit(scan_map_jitted, static_argnames=['npix_submap', 'nmap'])

def scan_map_interval_jax(mapdata, nmap, submap, subpix, weights, tod):
    """
    Sample a map into a timestream.

    This uses a local piece of a distributed map and the local pointing matrix
    to generate timestream values.

    Args:
        mapdata (Pixels):  The local piece of the map.
        nmap (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only submap)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap.
        weights (array, float64):  The pointing matrix weights (size: nsample*nmap).
        tod (array, float64):  The timestream on which to accumulate the map values.

    Returns:
        None: the result is put in tod.

    NOTE: as tod is always set to 0 just before calling the function, we put the value in to instead of adding them to tod
    """
    # gets number of pixels in each submap
    npix_submap = mapdata.distribution.n_pix_submap
    # converts inputs to arrays that can be injested by JAX
    mapdata = mapdata.raw
    mapdata_input = MutableJaxArray.to_array(mapdata)
    submap_input = MutableJaxArray.to_array(submap)
    subpix_input = MutableJaxArray.to_array(subpix)
    weights_input = MutableJaxArray.to_array(weights)
    # runs computations
    tod[:] = scan_map_jitted(mapdata_input, npix_submap, nmap, submap_input, subpix_input, weights_input)

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
        weights (array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        weight_index (array, int): The indexes of the weights (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        map_dist (PixelDistribution):
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    
    TODO vectorise loops
    """
    n_det = det_data_index.size

    # iterates on detectors and intervals
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

            # Process the interval
            maptod = np.zeros_like(pixels_interval, dtype=det_data.dtype)
            scan_map_interval_jax(mapdata, 
                                  nmap, 
                                  local_submap, 
                                  local_pixels, 
                                  weights_interval, 
                                  maptod)

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


#-------------------------------------------------------------------------------------------------
# NUMPY

def scan_map_interval_numpy(mapdata, submap, subpix, weights, det_data, should_zero, should_subtract):
    """
    Applies scan_map to a given interval.

    Args:
        mapdata (array, ?):  The local piece of the map (size nsamp*nmap).
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only submap)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap.
        weights (array, float64):  The pointing matrix weights (size: nsample*nmap).
        det_data (array, float64):  The timestream on which to accumulate the map values.
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data

    Returns:
        det_data
    """
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
        map_dist (PixelDistribution):
        should_zero (bool): should we zero det_data
        should_subtract (bool): should we subtract from det_data
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    print(f"DEBUG: Running scan_map_numpy!")

    # gets number of pixels in each submap
    npix_submap = mapdata.distribution.n_pix_submap
    # turns mapdata into a numpy array of shape nsamp*nmap
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
            # Get local submap and pixels
            local_submap, local_pixels = map_dist.global_pixel_to_submap(pixels_interval)
            # Process the interval
            new_det_data_interval = scan_map_interval_numpy(mapdata, local_submap, local_pixels, 
                                                            weights_interval, det_data_interval,
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
    TODO: implement target offload version of the code
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

# 3 make sure we run the verison we ask for
# 6 use_accel