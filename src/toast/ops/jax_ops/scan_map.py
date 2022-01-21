# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import select_implementation, ImplementationType
from ..._libtoast import scan_map_float64 as scan_map_float64_compiled, scan_map_float32 as scan_map_float32_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def scan_map_jitted(mapdata, npix_submap, nmap, submap, subpix, weights):
    """
    takes mapdata as a jax array and npix_submap as an integer
    """
    # display size information for debugging purposes
    print(f"DEBUG: jit compiling scan_map! npix_submap:{npix_submap} nsamp:{submap.size} nmap:{nmap} mapdata:{mapdata.shape} weights:{weights.shape}")

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

def scan_map_jax(mapdata, nmap, submap, subpix, weights, tod):
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

    NOTE: JAX port of the C++ implementation.
    NOTE: as tod is always set to 0 just before calling the function, we put the value in to instead of adding them to tod
    """
    # gets number of pixels in each submap
    npix_submap = mapdata.distribution.n_pix_submap
    # converts mapdata to a jax array
    mapdata = mapdata.raw.array()
    # runs computations
    tod[:] = scan_map_jitted(mapdata, npix_submap, nmap, submap, subpix, weights)

#-------------------------------------------------------------------------------------------------
# NUMPY

def scan_map_numpy(mapdata, nmap, submap, subpix, weights, tod):
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

    NOTE: Numpy port of the C++ implementation.
    NOTE: as tod is always set to 0 just before calling the function, we put the value in to instead of adding them to tod
    """
    # gets number of pixels in each submap
    npix_submap = mapdata.distribution.n_pix_submap

    # uses only samples with valid indices
    valid_samples = (subpix >= 0) & (submap >= 0)
    valid_weights = weights[valid_samples,:]
    valid_submap = submap[valid_samples]
    valid_subpix = subpix[valid_samples]

    # turns mapdata into a numpy array of shape nsamp*nmap
    mapdata = mapdata.raw.array()
    mapdata = np.reshape(mapdata, newshape=(-1,npix_submap,nmap))
    valid_mapdata = mapdata[valid_submap,valid_subpix,:]

    # updates tod
    tod[valid_samples] = np.sum(valid_mapdata * valid_weights, axis=1)

#-------------------------------------------------------------------------------------------------
# C++

def scan_map_compiled(mapdata, nmap, submap, subpix, weights, tod):
    """
    Sample a map into a timestream.

    This uses a local piece of a distributed map and the local pointing matrix
    to generate timestream values.

    Args:
        mapdata (Pixels):  The local piece of the map.
        nmap (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only submap)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap.
        weights (array, float64):  The pointing matrix weights for each time sample and map.
        tod (array, float64):  The timestream on which to accumulate the map values.

    Returns:
        None: the result is put in tod.

    NOTE: Wraps implementations of scan_map to factor the datatype handling
    """
    # picks the correct implementation as a function of the data-type used
    if mapdata.dtype.char == 'd': scan_map_function = scan_map_float64_compiled
    elif mapdata.dtype.char == 'f': scan_map_function = scan_map_float32_compiled
    else: raise RuntimeError("Projection supports only float32 and float64 binned maps")
    # gets number of pixels in each submap
    npix_submap = mapdata.distribution.n_pix_submap
    # runs the function (and flattens the weights)
    scan_map_function(npix_submap, nmap, submap, subpix, mapdata.raw, weights.reshape(-1), tod)

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
                                 default_implementationType=ImplementationType.COMPILED)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_scan_map")'
