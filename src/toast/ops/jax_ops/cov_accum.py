# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from .utils import select_implementation, ImplementationType
from ..._libtoast import cov_accum_diag_hits as cov_accum_diag_hits_compiled, cov_accum_diag_invnpp as cov_accum_diag_invnpp_compiled
from ...utils import AlignedI64, AlignedF64

#-------------------------------------------------------------------------------------------------
# JAX

def cov_accum_diag_hits_inner_jax(nsubpix, submap, subpix, hits):
    """
    Args:
        nsubpix (int):  The number of pixels in each submap.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only locally stored submaps) (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap (size nsamp).
        hits (array, int64):  The local hitmap buffer to accumulate (size ???).

    Returns:
        hits
    """
    # displays problem size
    hits = jnp.reshape(hits, newshape=(-1, nsubpix))
    print(f"DEBUG: jit-compiling 'cov_accum_diag_hits' with nsubpix:{nsubpix} nsamp:{submap.size} hits:{hits.shape}")

    # updates hits
    added_value = jnp.where((submap >= 0) & (subpix >= 0), 1, 0)
    hits = hits.at[submap, subpix].add(added_value)
    return hits.ravel()

# jit compiling
cov_accum_diag_hits_inner_jax = jax.jit(cov_accum_diag_hits_inner_jax, static_argnames=['nsubpix'])

def cov_accum_diag_hits_jax(nsub, nsubpix, nnz, submap, subpix, hits):
    """
    Accumulate hit map.
    This uses a pointing matrix to accumulate the local pieces of the hit map.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only locally stored submaps) (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap (size nsamp).
        hits (array, int64):  The local hitmap buffer to accumulate (size ???).

    Returns:
        None (result is put in hits).
    """
    # converts hits to numpy array to make ingestion by JAX possible
    if type(hits) == AlignedI64: hits = hits.array()
    hits[:] = cov_accum_diag_hits_inner_jax(nsubpix, submap, subpix, hits)

def cov_accum_diag_invnpp_inner_jax(nsubpix, nnz, submap, subpix, weights, scale, invnpp):
    """
    Args:
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index
            within the local map (i.e. including only locally stored submaps)
            (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index
            within the submap (size nsamp).
        weights (array, float64):  The pointing matrix weights for each time
            sample and map (shape nw*nnz).
        scale (float):  Optional scaling factor.
        invnpp (array, float64):  The local buffer of diagonal inverse pixel
            covariances, stored as the lower triangle for each pixel (shape ?*nsubpix*block with block=(nnz * (nnz + 1))/2).

    Returns:
        invnpp.
    """
    # displays problem size
    block = (nnz * (nnz + 1)) // 2
    weights = jnp.reshape(weights, newshape=(-1,nnz))
    invnpp = jnp.reshape(invnpp, newshape=(-1,nsubpix,block))
    print(f"DEBUG: jit-compiling 'cov_accum_diag_invnpp' with nsubpix:{nsubpix} nnz:{nnz} nsamp:{submap.size} weights:{weights.shape} invnpp:{invnpp.shape}")

    # converts flat index (i_block) back to index into upper triangular matrix of side nnz
    # you can rederive the equations by knowing that i_block = col + row*nnz + row(row+1)/2
    # then assuming col=row (close enough since row <= col < nnz) and rounding down to get row
    i_block = jnp.arange(start=0, stop=block)
    row = (2*nnz + 1 - jnp.sqrt((2*nnz + 1)**2 - 8*i_block)).astype(int) // 2
    col = i_block + (row*(row+1))//2 - row*nnz

    # computes mask
    # newaxis are there to make dimenssion compatible with added_value
    submap_2D = submap[..., jnp.newaxis]
    subpix_2D = subpix[..., jnp.newaxis]
    valid_index = (submap_2D >= 0) & (subpix_2D >= 0)

    # updates invnpp
    added_value = weights[:,col] * weights[:,row] * scale
    masked_added_value = jnp.where(valid_index, added_value, 0.0)
    invnpp = invnpp.at[submap,subpix,:].add(masked_added_value)

    return invnpp.ravel()

# jit compiling
cov_accum_diag_invnpp_inner_jax = jax.jit(cov_accum_diag_invnpp_inner_jax, static_argnames=['nsubpix','nnz'])

def cov_accum_diag_invnpp_jax(nsub, nsubpix, nnz, submap, subpix, weights, scale, invnpp):
    """
    Accumulate block diagonal noise covariance.
    This uses a pointing matrix to accumulate the local pieces
    of the inverse diagonal pixel covariance.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index
            within the local map (i.e. including only locally stored submaps)
            (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index
            within the submap (size nsamp).
        weights (array, float64):  The pointing matrix weights for each time
            sample and map (shape nw*nnz).
        scale (float):  Optional scaling factor.
        invnpp (array, float64):  The local buffer of diagonal inverse pixel
            covariances, stored as the lower triangle for each pixel (shape ?*nsubpix*block with block=(nnz * (nnz + 1))/2).

    Returns:
        None (stores the result in invnpp).
    """
    # converts invnpp to numpy array to make ingestion by JAX possible
    if type(invnpp) == AlignedF64: invnpp = invnpp.array()
    invnpp[:] = cov_accum_diag_invnpp_inner_jax(nsubpix, nnz, submap, subpix, weights, scale, invnpp)

#-------------------------------------------------------------------------------------------------
# NUMPY

def cov_accum_diag_hits_numpy(nsub, nsubpix, nnz, submap, subpix, hits):
    """
    Accumulate hit map.
    This uses a pointing matrix to accumulate the local pieces of the hit map.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only locally stored submaps) (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap (size nsamp).
        hits (array, int64):  The local hitmap buffer to accumulate (size ???).

    Returns:
        None (result is put in hits).
    """
    nsamp = submap.size
    hits = np.asarray(hits).reshape((-1, nsubpix))
    print(f"DEBUG: running 'cov_accum_diag_hits_numpy' with nsub:{nsub} nsubpix:{nsubpix} nnz:{nnz} nsamp:{nsamp} hits:{hits.shape}")

    for i_samp in range(nsamp):
        isubmap = submap[i_samp]
        ipix = subpix[i_samp]
        if ((isubmap >= 0) and (ipix >= 0)):
            hits[isubmap, ipix] += 1

def cov_accum_diag_invnpp_numpy(nsub, nsubpix, nnz, submap, subpix, weights, scale, invnpp):
    """
    Accumulate block diagonal noise covariance.
    This uses a pointing matrix to accumulate the local pieces
    of the inverse diagonal pixel covariance.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index
            within the local map (i.e. including only locally stored submaps)
            (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index
            within the submap (size nsamp).
        weights (array, float64):  The pointing matrix weights for each time
            sample and map (shape nw*nnz).
        scale (float):  Optional scaling factor.
        invnpp (array, float64):  The local buffer of diagonal inverse pixel
            covariances, stored as the lower triangle for each pixel (shape ?*nsubpix*block with block=(nnz * (nnz + 1))/2).

    Returns:
        None (stores the result in invnpp).
    """
    nsamp = submap.size
    block = (nnz * (nnz + 1)) // 2
    weights = np.asarray(weights).reshape((-1,nnz))
    invnpp = np.asarray(invnpp).reshape((-1,nsubpix,block))
    print(f"DEBUG: running 'cov_accum_diag_invnpp_numpy' with nsub:{nsub} nsubpix:{nsubpix} nnz:{nnz} nsamp:{nsamp} weights:{weights.shape} scale:{scale} invnpp:{invnpp.shape}")

    for i_samp in range(nsamp):
        isubmap = submap[i_samp]
        ipix = subpix[i_samp]
        if ((isubmap >= 0) and (ipix >= 0)):
            for i_block in range(block):
                # converts i_block back to index into upper triangular matrix of side nnz
                # you can rederive the equations by knowing that i_block = col + row*nnz + row(row+1)/2
                # then assuming col=row (close enough since row <= col < nnz) and rounding down to get row
                row = int(2*nnz + 1 - np.sqrt((2*nnz + 1)**2 - 8*i_block)) // 2
                col = i_block + (row*(row+1))//2 - row*nnz
                invnpp[isubmap,ipix,i_block] += weights[i_samp,col] * weights[i_samp,row] * scale

#-------------------------------------------------------------------------------------------------
# C++

"""
void toast::cov_accum_diag_hits(int64_t nsub, int64_t subsize, int64_t nnz,
                                int64_t nsamp,
                                int64_t const * indx_submap,
                                int64_t const * indx_pix, int64_t * hits) {
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (size_t i = 0; i < nsamp; ++i) {
            if ((indx_submap[i] < 0) || (indx_pix[i] < 0)) continue;

            const int64_t hpx = (indx_submap[i] * subsize) + indx_pix[i];
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            hits[hpx] += 1;
        }
    }

    return;
}

void toast::cov_accum_diag_invnpp(int64_t nsub, int64_t subsize, int64_t nnz,
                                  int64_t nsamp,
                                  int64_t const * indx_submap,
                                  int64_t const * indx_pix,
                                  double const * weights,
                                  double scale,
                                  double * invnpp) {
    const int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int nthread = omp_get_num_threads();
        int trank = omp_get_thread_num();
        int64_t npix_thread = nsub * subsize / nthread + 1;
        int64_t first_pix = trank * npix_thread;
        int64_t last_pix = first_pix + npix_thread - 1;
        #endif // ifdef _OPENMP

        for (size_t i = 0; i < nsamp; ++i) {
            const int64_t isubmap = indx_submap[i] * subsize;
            const int64_t ipix = indx_pix[i];
            if ((isubmap < 0) || (ipix < 0)) continue;

            const int64_t hpx = isubmap + ipix;
            #ifdef _OPENMP
            if ((hpx < first_pix) || (hpx > last_pix)) continue;
            #endif // ifdef _OPENMP
            const int64_t ipx = hpx * block;

            const double * wpointer = weights + i * nnz;
            double * covpointer = invnpp + ipx;
            for (size_t j = 0; j < nnz; ++j, ++wpointer) {
                const double scaled_weight = *wpointer * scale;
                const double * wpointer2 = wpointer;
                for (size_t k = j; k < nnz; ++k, ++wpointer2, ++covpointer) {
                    *covpointer += *wpointer2 * scaled_weight;
                }
            }
        }
    }

    return;
}
"""

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
cov_accum_diag_hits = select_implementation(cov_accum_diag_hits_compiled, 
                                            cov_accum_diag_hits_numpy, 
                                            cov_accum_diag_hits_jax)
cov_accum_diag_invnpp = select_implementation(cov_accum_diag_invnpp_compiled, 
                                              cov_accum_diag_invnpp_numpy, 
                                              cov_accum_diag_invnpp_jax)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_mapmaker_utils"); toast.tests.run("covariance");'