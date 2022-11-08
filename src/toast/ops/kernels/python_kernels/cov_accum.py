# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np



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
    nsamp = submap.size
    hits = np.asarray(hits).reshape((-1, nsubpix))
    print(
        f"DEBUG: running 'cov_accum_diag_hits_numpy' with nsub:{nsub} nsubpix:{nsubpix} nnz:{nnz} nsamp:{nsamp} hits:{hits.shape}"
    )

    for i_samp in range(nsamp):
        isubmap = submap[i_samp]
        ipix = subpix[i_samp]
        if (isubmap >= 0) and (ipix >= 0):
            hits[isubmap, ipix] += 1


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
    nsamp = submap.size
    block = (nnz * (nnz + 1)) // 2
    weights = np.asarray(weights).reshape((-1, nnz))
    invnpp = np.asarray(invnpp).reshape((-1, nsubpix, block))

    for i_samp in range(nsamp):
        isubmap = submap[i_samp]
        ipix = subpix[i_samp]
        if (isubmap >= 0) and (ipix >= 0):
            for i_block in range(block):
                # converts i_block back to index into upper triangular matrix of side nnz
                # you can rederive the equations by knowing that i_block = col + row*nnz + row(row+1)/2
                # then assuming col=row (close enough since row <= col < nnz) and rounding down to get row
                row = int(2 * nnz + 1 - np.sqrt((2 * nnz + 1) ** 2 - 8 * i_block)) // 2
                col = i_block + (row * (row + 1)) // 2 - row * nnz
                invnpp[isubmap, ipix, i_block] += (
                    weights[i_samp, col] * weights[i_samp, row] * scale
                )
