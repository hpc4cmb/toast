# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from ..._libtoast import build_noise_weighted as libtoast_build_noise_weighted
from ..._libtoast import cov_accum_diag_hits as libtoast_cov_accum_diag_hits
from ..._libtoast import cov_accum_diag_invnpp as libtoast_cov_accum_diag_invnpp
from ...accelerator import ImplementationType, kernel, use_accel_jax
from .kernels_numpy import (
    build_noise_weighted_numpy,
    cov_accum_diag_hits_numpy,
    cov_accum_diag_invnpp_numpy,
)

if use_accel_jax:
    from .kernels_jax import (
        build_noise_weighted_jax,
        cov_accum_diag_hits_jax,
        cov_accum_diag_invnpp_jax,
    )


@kernel(impl=ImplementationType.COMPILED, name="build_noise_weighted")
def build_noise_weighted_compiled(*args, use_accel=False):
    return libtoast_build_noise_weighted(*args, use_accel)


@kernel(impl=ImplementationType.COMPILED, name="cov_accum_diag_hits")
def cov_accum_diag_hits_compiled(*args, use_accel=False):
    return libtoast_cov_accum_diag_hits(*args, use_accel)


@kernel(impl=ImplementationType.COMPILED, name="cov_accum_diag_invnpp")
def cov_accum_diag_invnpp_compiled(*args, use_accel=False):
    return libtoast_cov_accum_diag_invnpp(*args, use_accel)


@kernel(impl=ImplementationType.DEFAULT)
def build_noise_weighted(
    global2local,
    zmap,
    pixel_index,
    pixels,
    weight_index,
    weights,
    data_index,
    det_data,
    flag_index,
    det_flags,
    det_scale,
    det_flag_mask,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel=False,
):
    """Kernel for accumulating the noise weighted map.

    Args:
        global2local (array):  The mapping from global submap to local submap index.
        zmap (array):  The local piece of the noise weighted map, indexed by submap,
            then pixel, then value.
        pixel_index (array):  The index into the detector pixel array for each
            detector.
        pixels (array):  The array of detector pixels for each sample.
        weight_index (array):  The index into the weights array for each detector.
        weights (array):  The array of I, Q, and U weights at each sample for each
            detector.
        data_index (array):  The index into the data array for each detector.
        det_data (array):  The detector data at each sample for each detector.
        flag_index (array):  The index into the flag array for each detector.
        det_flags (array):  The detector flag at each sample for each detector.
        det_scale (float):  Scale factor for each detector, applied before accumulating.
        det_flag_mask (int):  The flag mask to apply to the detector flags.
        intervals (array):  The array of sample intervals.
        shared_flags (array):  The array of common flags for each sample.
        shared_flag_mask (int):  The flag mask to apply.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None

    """
    return libtoast_build_noise_weighted(
        global2local,
        zmap,
        pixel_index,
        pixels,
        weight_index,
        weights,
        data_index,
        det_data,
        flag_index,
        det_flags,
        det_scale,
        det_flag_mask,
        intervals,
        shared_flags,
        shared_flag_mask,
        use_accel,
    )


@kernel(impl=ImplementationType.DEFAULT)
def cov_accum_diag_invnpp(
    nsub,
    nsubpix,
    nnz,
    submap,
    subpix,
    weights,
    scale,
    invnpp,
    use_accel=False,
):
    """Kernel for accumulating the inverse diagonal pixel noise covariance.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index
            within the local map (i.e. including only locally stored submaps)
        subpix (array, int64):  For each time domain sample, the pixel index
            within the submap.
        weights (array, float64):  The pointing matrix weights for each time
            sample and map.
        scale (float):  Optional scaling factor.
        invnpp (array, float64):  The local buffer of diagonal inverse pixel
            covariances, stored as the lower triangle for each pixel.

    Returns:
        None

    """
    return libtoast_cov_accum_diag_invnpp(
        nsub,
        nsubpix,
        nnz,
        submap,
        subpix,
        weights,
        scale,
        invnpp,
        use_accel,
    )


@kernel(impl=ImplementationType.DEFAULT)
def cov_accum_diag_hits(
    nsub,
    nsubpix,
    nnz,
    submap,
    subpix,
    hits,
    use_accel=False,
):
    """Kernel for accumulating the hits.

    Args:
        nsub (int):  The number of locally stored submaps.
        nsubpix (int):  The number of pixels in each submap.
        nnz (int):  The number of non-zeros in each row of the pointing matrix.
        submap (array, int64):  For each time domain sample, the submap index
            within the local map (i.e. including only locally stored submaps)
        subpix (array, int64):  For each time domain sample, the pixel index
            within the submap.
        hits (array, int64):  The local hitmap buffer to accumulate.

    Returns:
        None

    """
    return libtoast_cov_accum_diag_hits(
        nsub,
        nsubpix,
        nnz,
        submap,
        subpix,
        hits,
        use_accel,
    )
