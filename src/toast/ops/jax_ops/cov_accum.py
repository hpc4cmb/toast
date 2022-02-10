# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import get_compile_time, select_implementation, ImplementationType
from ..._libtoast import cov_accum_zmap as cov_accum_zmap_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def cov_accum_zmap_jitted(nsub, subsize, nnz, weights, scale, tod, zmap):
    # unflatten arrays
    print(f"DEBUG: jit-compiling 'cov_accum_zmap' for nsub:{nsub} subsize:{subsize} nnz:{nnz} nsamp:{weights.shape[0]} scale:{scale}")
    
    # values to be added to zmap
    scaled_signal = scale * tod
    shift = weights * scaled_signal[:,jnp.newaxis]

    # NOTE: JAX deals properly with duplicates indices according to our tests
    return zmap + shift

# jit
cov_accum_zmap_jitted = jax.jit(cov_accum_zmap_jitted, static_argnames=['nsub','subsize','nnz','scale'])

def cov_accum_zmap_jax(nsub, subsize, nnz, submap, subpix, weights, scale, tod, zmap):
    """
    Accumulate the noise weighted map.

    This uses a pointing matrix and timestream data to accumulate the local pieces
    of the noise weighted map.

    Args:
        nsub (int):  The number of locally stored submaps.
        subsize (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64): For each time domain sample, the submap index within the local map 
                               (i.e. including only locally stored submaps)
                               (size nsamp)
        subpix (array, int64): For each time domain sample, the pixel index within the submap (size nsamp).
        weights (array, float64): The pointing matrix weights for each time sample and map (shape nsamp*nnz).
        scale (float):  Optional scaling factor.
        tod (array, float64): The timestream to accumulate in the noise weighted map (size nsamp).
        zmap (array, float64): The local noise weighted map buffer (size nsub*subsize*nnz).

    Returns:
        None. (results are pur into zmap)
    """
    # converts into properly shaped numpy arrays
    zmap = np.reshape(zmap.array(), newshape=(nsub,subsize,nnz))
    weights = np.reshape(weights, newshape=(-1,nnz))

    # keeps only valid data
    # invalid indices are *very* uncommon
    # NOTE: in benchmarks this takes about one in 62 seconds
    valid_indices = (submap >= 0) & (subpix >= 0)
    if not np.all(valid_indices):
        submap = submap[valid_indices]
        subpix = subpix[valid_indices]
        tod = tod[valid_indices]
        weights = weights[valid_indices,:]
    
    # updates the slice of zmap with the result
    subzmap = zmap[submap,subpix,:]
    subzmap[:] = cov_accum_zmap_jitted(nsub, subsize, nnz, weights, scale, tod, subzmap)

#-------------------------------------------------------------------------------------------------
# NUMPY

def cov_accum_zmap_numpy(nsub, subsize, nnz, submap, subpix, weights, scale, tod, zmap):
    """
    Accumulate the noise weighted map.

    This uses a pointing matrix and timestream data to accumulate the local pieces
    of the noise weighted map.

    Args:
        nsub (int):  The number of locally stored submaps.

        subsize (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64): For each time domain sample, the submap index within the local map 
                               (i.e. including only locally stored submaps)
                               (size nsamp)
        subpix (array, int64): For each time domain sample, the pixel index within the submap (size nsamp).
        weights (array, float64): The pointing matrix weights for each time sample and map (shape nsamp*nnz).
        scale (float):  Optional scaling factor.
        tod (array, float64): The timestream to accumulate in the noise weighted map (size nsamp).
        zmap (array, float64): The local noise weighted map buffer (size nsub*subsize*nnz).

    Returns:
        None. (results are pur into zmap)
    """
    # converts arrays into properly shaped numpy arrays
    zmap = np.reshape(zmap.array(), newshape=(nsub,subsize,nnz))
    weights = np.reshape(weights, newshape=(-1,nnz))

    # TODO debug
    #print(f"DEBUG: running 'cov_accum_zmap_numpy' for nsub:{nsub} subsize:{subsize} nnz:{nnz} nsamp:{submap.size} scale:{scale}")

    # values to be added to zmap
    scaled_signal = scale * tod
    shift = weights * scaled_signal[:,np.newaxis]

    # keep only valid indices
    # invalid indices are *very* uncommon
    valid_indices = (submap >= 0) & (subpix >= 0)
    if not np.all(valid_indices):
        shift = shift[valid_indices,:]
        submap = submap[valid_indices]
        subpix = subpix[valid_indices]

    # NOTE: we use `np.add.at` as it deals properly with duplicates in the zmap indices
    #zmap[submap,subpix,:] += shift
    np.add.at(zmap, (submap,subpix), shift)

#-------------------------------------------------------------------------------------------------
# C++

"""
void toast::cov_accum_zmap(int64_t nsub, int64_t subsize, int64_t nnz,
                           int64_t nsamp,
                           int64_t const * indx_submap,
                           int64_t const * indx_pix, double const * weights,
                           double scale, double const * signal,
                           double * zdata) {
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (int64_t i = 0; i < nsamp; ++i) 
        {
            const int64_t isubmap = indx_submap[i] * subsize;
            const int64_t ipix = indx_pix[i];
            if ((isubmap < 0) || (ipix < 0)) continue;

            const int64_t hpx = isubmap + ipix;
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            const int64_t zpx = hpx * nnz;

            const double scaled_signal = scale * signal[i];
            double * zpointer = zdata + zpx;
            const double * wpointer = weights + i * nnz;
            for (int64_t j = 0; j < nnz; ++j, ++zpointer, ++wpointer) 
            {
                *zpointer += *wpointer * scaled_signal;
            }
        }
    }
}
"""

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
cov_accum_zmap = select_implementation(cov_accum_zmap_compiled, 
                                       cov_accum_zmap_numpy, 
                                       cov_accum_zmap_jax, 
                                       default_implementationType=ImplementationType.JAX)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
#cov_accum_zmap = get_compile_time(cov_accum_zmap)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_mapmaker_utils"); toast.tests.run("ops_mapmaker_binning")'

# to bench:
# use scanmap config and check BuildNoiseWeighted field in timing.csv
# one problem: this includes call to other operators