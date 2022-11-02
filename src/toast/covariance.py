# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ._libtoast import (
    AlignedF64,
    cov_apply_diag,
    cov_eigendecompose_diag,
    cov_mult_diag,
)
from .pixels import PixelData
from .timing import function_timer


# These are just wrappers to time the libtoast calls
@function_timer
def libtoast_cov_mult_diag(*args):
    cov_mult_diag(*args)


@function_timer
def libtoast_cov_apply_diag(*args):
    cov_apply_diag(*args)


@function_timer
def libtoast_cov_eigendecompose_diag(*args):
    cov_eigendecompose_diag(*args)


def create_local_invert(n_pix_submap, mapnnz, threshold, rcond, invert=False):
    """Generate a function for inverting locally owned submaps of a covariance.

    Args:
        n_pix_submap (int):  The number of pixels in a submap.
        mapnnz (int):  The number of map elements per pixel.
        threshold (float):  The condition number threshold to apply.
        rcond (PixelData):  If not None, the inverse condition number PixelData object.

    Returns:
        (function):  A function suitable for the sync_alltoallv() method.

    """

    @function_timer
    def local_invert(n_submap_value, receive_locations, receive, reduce_buf):
        # Locally invert owned submaps
        for sm, locs in receive_locations.items():
            # We have multiple copies of submap data- we will invert just the first
            # one and copy the result into the other buffer locations to be sent
            # back to the processes with this submap.
            reduce_buf[:] = receive[locs[0] : locs[0] + n_submap_value]
            rdata = None
            if rcond is None:
                rdata = np.empty(shape=0, dtype=np.float64)
            else:
                rcond.reduce_buf[:] = 0.0
                rdata = rcond.reduce_buf
            libtoast_cov_eigendecompose_diag(
                1,
                n_pix_submap,
                mapnnz,
                reduce_buf,
                rdata,
                threshold,
                invert,
            )
            for lc in locs:
                receive[lc : lc + n_submap_value] = reduce_buf
            if rcond is not None:
                for lc in rcond._recv_locations[sm]:
                    rcond.receive[lc : lc + n_pix_submap] = rcond.reduce_buf

    return local_invert


@function_timer
def covariance_invert(npp, threshold, rcond=None, use_alltoallv=False):
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
        use_alltoallv (bool):  If True, communicate submaps and have every process work
            on a portion of them.  This may be faster than processing all submaps
            locally.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp.n_value) - 1) / 2) + 0.5)
    nppdata = npp.raw
    if rcond is not None:
        if npp.distribution != rcond.distribution:
            raise RuntimeError(
                "covariance matrix and rcond must have same pixel distribution"
            )
        if rcond.n_value != 1:
            raise RuntimeError("condition number map should have n_value = 1")

    if use_alltoallv:
        if rcond is not None:
            # Ensure that the rcond receive buffer exists
            rcond.setup_alltoallv()
        linvert = create_local_invert(
            npp.distribution.n_pix_submap, mapnnz, threshold, rcond, invert=True
        )
        npp.sync_alltoallv(local_func=linvert)
        if rcond is not None:
            # Return rcond result
            rcond.reverse_alltoallv()
    else:
        rdata = None
        if rcond is None:
            rdata = np.empty(shape=0, dtype=np.float64)
        else:
            rdata = rcond.raw
        libtoast_cov_eigendecompose_diag(
            npp.distribution.n_local_submap,
            npp.distribution.n_pix_submap,
            mapnnz,
            nppdata,
            rdata,
            threshold,
            True,
        )

    # Invert the units
    new_units = (1.0 / npp.units).decompose()
    npp.update_units(new_units)

    return


def create_local_multiply(n_pix_submap, mapnnz, other):
    """Generate a function for multiplying locally owned submaps of covariances.

    Args:
        n_pix_submap (int):  The number of pixels in a submap.
        mapnnz (int):  The number of map elements per pixel.
        other (PixelData):  The other PixelData covariance object.

    Returns:
        (function):  A function suitable for the sync_alltoallv() method.

    """

    @function_timer
    def local_multiply(n_submap_value, receive_locations, receive, reduce_buf):
        for sm, locs in receive_locations.items():
            # We have multiple copies of submap data- we will multiply just the first
            # one and copy the result into the other buffer locations to be sent
            # back to the processes with this submap.
            reduce_buf[:] = receive[locs[0] : locs[0] + n_submap_value]
            other_buf = other.reduce_buf
            other_buf[:] = other.receive[locs[0] : locs[0] + n_submap_value]
            libtoast_cov_mult_diag(1, n_pix_submap, mapnnz, reduce_buf, other_buf)
            for lc in locs:
                receive[lc : lc + n_submap_value] = reduce_buf

    return local_multiply


@function_timer
def covariance_multiply(npp1, npp2, use_alltoallv=False):
    """Multiply two diagonal noise covariances.

    This does an in-place multiplication of the covariance.
    The data values of the first covariance (npp1) are replaced with
    the result.

    Args:
        npp1 (PixelData): The first distributed covariance.
        npp2 (PixelData): The second distributed covariance.
        use_alltoallv (bool):  If True, communicate submaps and have every process work
            on a portion of them.  This may be faster than processing all submaps
            locally.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp1.n_value) - 1) / 2) + 0.5)

    if npp1.distribution != npp2.distribution:
        raise RuntimeError("covariance matrices must have same pixel distribution")
    if npp1.n_value != npp2.n_value:
        raise RuntimeError("covariance matrices must have same n_values")

    if use_alltoallv:
        npp2.forward_alltoallv()
        lmultiply = create_local_multiply(npp1.distribution.n_pix_submap, mapnnz, npp2)
        npp1.sync_alltoallv(local_func=lmultiply)
    else:
        libtoast_cov_mult_diag(
            npp1.distribution.n_local_submap,
            npp1.distribution.n_pix_submap,
            mapnnz,
            npp1.raw,
            npp2.raw,
        )

    # Update the units
    new_units = (npp1.units * npp2.units).decompose()
    npp1.update_units(new_units)

    return


def create_local_apply(n_pix_submap, mapnnz, cov):
    """Generate a function for applying locally owned submaps of covariances.

    Args:
        n_pix_submap (int):  The number of pixels in a submap.
        mapnnz (int):  The number of map values per pixel.
        cov (PixelData):  The PixelData covariance object.

    Returns:
        (function):  A function suitable for the sync_alltoallv() method of the map.

    """

    @function_timer
    def local_apply(n_submap_value, receive_locations, receive, reduce_buf):
        for sm, locs in receive_locations.items():
            # We have multiple copies of submap data- we will multiply just the first
            # one and copy the result into the other buffer locations to be sent
            # back to the processes with this submap.

            cov_locs = cov._recv_locations[sm]
            cov_buf = cov.reduce_buf

            reduce_buf[:] = receive[locs[0] : locs[0] + n_submap_value]

            cov_buf[:] = cov.receive[
                cov_locs[0] : cov_locs[0] + (n_pix_submap * cov.n_value)
            ]

            libtoast_cov_apply_diag(1, n_pix_submap, mapnnz, cov_buf, reduce_buf)

            for lc in locs:
                receive[lc : lc + n_submap_value] = reduce_buf

    return local_apply


@function_timer
def covariance_apply(npp, m, use_alltoallv=False):
    """Multiply a map by a diagonal noise covariance.

    This does an in-place multiplication of the covariance and a
    map.  The results are returned in place of the input map.

    Args:
        npp (PixelData): The distributed covariance.
        m (PixelData): The distributed map.
        use_alltoallv (bool):  If True, communicate submaps and have every process work
            on a portion of them.  This may be faster than processing all submaps
            locally.

    Returns:
        None

    """
    mapnnz = int(((np.sqrt(8 * npp.n_value) - 1) / 2) + 0.5)

    if npp.distribution != m.distribution:
        raise RuntimeError(
            "covariance matrix and map must have same pixel distribution"
        )
    if m.n_value != mapnnz:
        raise RuntimeError("covariance matrix and map have incompatible NNZ values")

    if use_alltoallv:
        npp.forward_alltoallv()
        lapply = create_local_apply(npp.distribution.n_pix_submap, mapnnz, npp)
        m.sync_alltoallv(local_func=lapply)
    else:
        nppdata = npp.raw
        mdata = m.raw
        libtoast_cov_apply_diag(
            npp.distribution.n_local_submap,
            npp.distribution.n_pix_submap,
            mapnnz,
            nppdata,
            mdata,
        )

    # Update units
    new_units = (npp.units * m.units).decompose()
    m.update_units(new_units)
    return
