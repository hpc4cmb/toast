# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .timing import function_timer

from .operator import Operator

from ._libtoast import (
    AlignedF64,
    cov_mult_diag,
    cov_apply_diag,
    cov_eigendecompose_diag,
)

from .pixels import PixelData


@function_timer
def covariance_invert(npp, threshold, rcond=None):
    """Invert a diagonal noise covariance.

    This does an inversion of the covariance.  The threshold is
    applied to the condition number of each block of the matrix.  Pixels
    failing the cut are set to zero.

    Args:
        npp (PixelData):  The distributed covariance, with the lower triangle of the
            symmetric matrix at each pixel.
        threshold (float):  The condition number threshold to apply.
        rcond (PixelData):  (Optional) The distributed inverse condition number map
            to fill.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp.n_value) - 1) / 2) + 0.5)
    nppdata = npp.raw
    if nppdata is None:
        nppdata = np.empty(shape=0, dtype=np.float64)
    if rcond is not None:
        if rcond.distribution.n_pix != npp.distribution.n_pix:
            raise RuntimeError(
                "covariance matrix and condition number map must have same number "
                "of pixels"
            )
        if rcond.distribution.n_pix_submap != npp.distribution.n_pix_submap:
            raise RuntimeError(
                "covariance matrix and condition number map must have same submap size"
            )
        if rcond.n_value != 1:
            raise RuntimeError("condition number map should have n_value = 1")

        rdata = rcond.raw
        if rdata is None:
            rdata = np.empty(shape=0, dtype=np.float64)
        cov_eigendecompose_diag(
            npp.n_local_submap,
            npp.n_pix_submap,
            mapnnz,
            nppdata,
            rdata,
            threshold,
            True,
        )
    else:
        temp = AlignedF64(npp.n_local_submap * npp.n_pix_submap)
        cov_eigendecompose_diag(
            npp.n_local_submap, npp.n_pix_submap, mapnnz, nppdata, temp, threshold, True
        )
        temp.clear()
        del temp
    return


@function_timer
def covariance_multiply(npp1, npp2):
    """Multiply two diagonal noise covariances.

    This does an in-place multiplication of the covariance.
    The data values of the first covariance (npp1) are replaced with
    the result.

    Args:
        npp1 (PixelData): The first distributed covariance.
        npp2 (PixelData): The second distributed covariance.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp1.n_value) - 1) / 2) + 0.5)

    if npp1.n_pix != npp2.n_pix:
        raise RuntimeError("covariance matrices must have same number of pixels")
    if npp1.n_pix_submap != npp2.n_pix_submap:
        raise RuntimeError("covariance matrices must have same submap size")
    if npp1.n_value != npp2.n_value:
        raise RuntimeError("covariance matrices must have same n_values")

    npp1data = npp1.raw
    if npp1data is None:
        npp1data = np.empty(shape=0, dtype=np.float64)
    npp2data = npp2.raw
    if npp2data is None:
        npp2data = np.empty(shape=0, dtype=np.float64)
    cov_mult_diag(npp1.n_submap, npp1.n_pix_submap, mapnnz, npp1data, npp2data)
    return


@function_timer
def covariance_apply(npp, m):
    """Multiply a map by a diagonal noise covariance.

    This does an in-place multiplication of the covariance and a
    map.  The results are returned in place of the input map.

    Args:
        npp (PixelData): The distributed covariance.
        m (PixelData): The distributed map.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp.n_value) - 1) / 2) + 0.5)

    if m.n_pix != npp.n_pix:
        raise RuntimeError("covariance matrix and map must have same number of pixels")
    if m.n_pix_submap != npp.n_pix_submap:
        raise RuntimeError("covariance matrix and map must have same submap size")
    if m.n_value != mapnnz:
        raise RuntimeError("covariance matrix and map have incompatible NNZ values")

    nppdata = npp.raw
    if nppdata is None:
        nppdata = np.empty(shape=0, dtype=np.float64)
    mdata = m.raw
    if mdata is None:
        mdata = np.empty(shape=0, dtype=np.float64)
    cov_apply_diag(npp.n_submap, npp.n_pix_submap, mapnnz, nppdata, mdata)
    return


@function_timer
def covariance_rcond(npp):
    """Compute the inverse condition number map.

    This computes the inverse condition number map of the supplied
    covariance matrix.

    Args:
        npp (PixelData): The distributed covariance.

    Returns:
        rcond (PixelData): The distributed inverse condition number map.
    """
    mapnnz = int(((np.sqrt(8 * npp.n_value) - 1) / 2) + 0.5)

    rcond = PixelData(npp.distribution, np.float64, n_value=1)

    threshold = np.finfo(np.float64).eps

    nppdata = npp.raw
    if nppdata is None:
        nppdata = np.empty(shape=0, dtype=np.float64)

    rdata = rcond.raw
    if rdata is None:
        rdata = np.empty(shape=0, dtype=np.float64)

    cov_eigendecompose_diag(
        npp.n_submap, npp.n_pix_submap, mapnnz, nppdata, rdata, threshold, False
    )

    return rcond
