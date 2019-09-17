# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..timing import function_timer

from ..op import Operator

from .._libtoast import cov_mult_diag, cov_apply_diag, cov_eigendecompose_diag

from .pixels import DistPixels


@function_timer
def covariance_invert(npp, threshold, rcond=None):
    """Invert a diagonal noise covariance.

    This does an inversion of the covariance.  The threshold is
    applied to the condition number of each block of the matrix.  Pixels
    failing the cut are set to zero.

    Args:
        npp (DistPixels): The distributed covariance.
        threshold (float): The condition number threshold to apply.
        rcond (DistPixels): (Optional) The distributed inverse condition number map
            to fill.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp.nnz) - 1) / 2) + 0.5)
    nppdata = npp.flatdata
    if nppdata is None:
        nppdata = np.empty(shape=(0), dtype=np.float64)
    if rcond is not None:
        if rcond.size != npp.size:
            raise RuntimeError(
                "covariance matrix and condition number map must have same number "
                "of pixels"
            )
        if rcond.submap != npp.submap:
            raise RuntimeError(
                "covariance matrix and condition number map must have same submap size"
            )
        if rcond.nnz != 1:
            raise RuntimeError("condition number map should have NNZ = 1")

        rdata = rcond.flatdata
        if rdata is None:
            rdata = np.empty(shape=(0), dtype=np.float64)
        cov_eigendecompose_diag(
            npp.nsubmap, npp.submap, mapnnz, nppdata, rdata, threshold, True
        )

    else:
        temp = np.zeros(shape=(npp.nsubmap * npp.submap), dtype=np.float64)
        cov_eigendecompose_diag(
            npp.nsubmap, npp.submap, mapnnz, nppdata, temp, threshold, True
        )
    return


@function_timer
def covariance_multiply(npp1, npp2):
    """Multiply two diagonal noise covariances.

    This does an in-place multiplication of the covariance.
    The data values of the first covariance (npp1) are replaced with
    the result.

    Args:
        npp1 (3D array): The first distributed covariance.
        npp2 (3D array): The second distributed covariance.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp1.nnz) - 1) / 2) + 0.5)

    if npp1.size != npp2.size:
        raise RuntimeError("covariance matrices must have same number of pixels")
    if npp1.submap != npp2.submap:
        raise RuntimeError("covariance matrices must have same submap size")
    if npp1.nnz != npp2.nnz:
        raise RuntimeError("covariance matrices must have same NNZ values")

    npp1data = npp1.flatdata
    if npp1data is None:
        npp1data = np.empty(shape=(0), dtype=np.float64)
    npp2data = npp2.flatdata
    if npp2data is None:
        npp2data = np.empty(shape=(0), dtype=np.float64)
    cov_mult_diag(npp1.nsubmap, npp1.submap, mapnnz, npp1data, npp2data)
    return


@function_timer
def covariance_apply(npp, m):
    """Multiply a map by a diagonal noise covariance.

    This does an in-place multiplication of the covariance and a
    map.  The results are returned in place of the input map.

    Args:
        npp (DistPixels): The distributed covariance.
        m (DistPixels): The distributed map.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp.nnz) - 1) / 2) + 0.5)

    if m.size != npp.size:
        raise RuntimeError("covariance matrix and map must have same number of pixels")
    if m.submap != npp.submap:
        raise RuntimeError("covariance matrix and map must have same submap size")
    if m.nnz != mapnnz:
        raise RuntimeError("covariance matrix and map have incompatible NNZ values")

    nppdata = npp.flatdata
    if nppdata is None:
        nppdata = np.empty(shape=(0), dtype=np.float64)
    mdata = m.flatdata
    if mdata is None:
        mdata = np.empty(shape=(0), dtype=np.float64)
    cov_apply_diag(npp.nsubmap, npp.submap, mapnnz, nppdata, mdata)
    return


@function_timer
def covariance_rcond(npp):
    """Compute the inverse condition number map.

    This computes the inverse condition number map of the supplied
    covariance matrix.

    Args:
        npp (DistPixels): The distributed covariance.

    Returns:
        rcond (DistPixels): The distributed inverse condition number map.
    """
    mapnnz = int(((np.sqrt(8 * npp.nnz) - 1) / 2) + 0.5)

    rcond = DistPixels(
        comm=npp.comm,
        size=npp.size,
        nnz=1,
        dtype=np.float64,
        submap=npp.submap,
        local=npp.local,
        nest=npp.nested,
    )

    threshold = np.finfo(np.float64).eps

    nppdata = npp.flatdata
    if nppdata is None:
        nppdata = np.empty(shape=(0), dtype=np.float64)

    rdata = rcond.flatdata
    if rdata is None:
        rdata = np.empty(shape=(0), dtype=np.float64)

    cov_eigendecompose_diag(
        npp.nsubmap, npp.submap, mapnnz, nppdata, rdata, threshold, False
    )

    return rcond
