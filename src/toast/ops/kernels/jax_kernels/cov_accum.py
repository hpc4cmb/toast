# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from ....jax.mutableArray import MutableJaxArray


def cov_accum_diag_hits_inner(nsubpix, submap, subpix, hits):
    """
    Args:
        nsubpix (int):  The number of pixels in each submap.
        submap (array, int64):  For each time domain sample, the submap index within the local map (i.e. including only locally stored submaps) (size nsamp)
        subpix (array, int64):  For each time domain sample, the pixel index within the submap (size nsamp).
        hits (array, int64):  The local hitmap buffer to accumulate (size ???).

    Returns:
        hits
    """
    # computes update
    added_value = jnp.where((submap >= 0) & (subpix >= 0), 1, 0)

    # updates hits
    hits = jnp.reshape(hits, newshape=(-1, nsubpix))
    hits = hits.at[submap, subpix].add(added_value)
    return hits.ravel()


# jit compiling
cov_accum_diag_hits_inner = jax.jit(
    cov_accum_diag_hits_inner, static_argnames=["nsubpix"], donate_argnums=[3]
)


def cov_accum_diag_hits(nsub, nsubpix, nnz, submap, subpix, hits):
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
    submap_input = MutableJaxArray.to_array(submap)
    subpix_input = MutableJaxArray.to_array(subpix)
    hits_input = MutableJaxArray.to_array(hits)
    hits[:] = cov_accum_diag_hits_inner(
        nsubpix, submap_input, subpix_input, hits_input
    )


def cov_accum_diag_invnpp_inner(
    nsubpix, nnz, submap, subpix, weights, scale, invnpp
):
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
    # reshape data
    block = (nnz * (nnz + 1)) // 2
    weights = jnp.reshape(weights, newshape=(-1, nnz))
    invnpp = jnp.reshape(invnpp, newshape=(-1, nsubpix, block))

    # converts flat index (i_block) back to index into upper triangular matrix of side nnz
    # you can rederive the equations by knowing that i_block = col + row*nnz + row(row+1)/2
    # then assuming col=row (close enough since row <= col < nnz) and rounding down to get row
    i_block = jnp.arange(start=0, stop=block)
    row = (2 * nnz + 1 - jnp.sqrt((2 * nnz + 1) ** 2 - 8 * i_block)).astype(int) // 2
    col = i_block + (row * (row + 1)) // 2 - row * nnz

    # computes mask
    # newaxis are there to make dimenssion compatible with added_value
    submap_2D = submap[..., jnp.newaxis]
    subpix_2D = subpix[..., jnp.newaxis]
    valid_index = (submap_2D >= 0) & (subpix_2D >= 0)

    # updates invnpp
    added_value = weights[:, col] * weights[:, row] * scale
    masked_added_value = jnp.where(valid_index, added_value, 0.0)
    invnpp = invnpp.at[submap, subpix, :].add(masked_added_value)

    return invnpp.ravel()


# jit compiling
cov_accum_diag_invnpp_inner = jax.jit(
    cov_accum_diag_invnpp_inner,
    static_argnames=["nsubpix", "nnz"],
    donate_argnums=[6],
)


def cov_accum_diag_invnpp(
    nsub, nsubpix, nnz, submap, subpix, weights, scale, invnpp
):
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
    submap_input = MutableJaxArray.to_array(submap)
    subpix_input = MutableJaxArray.to_array(subpix)
    weights_input = MutableJaxArray.to_array(weights)
    invnpp_input = MutableJaxArray.to_array(invnpp)
    invnpp[:] = cov_accum_diag_invnpp_inner(
        nsubpix, nnz, submap_input, subpix_input, weights_input, scale, invnpp_input
    )

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_mapmaker_utils"); toast.tests.run("covariance");'
