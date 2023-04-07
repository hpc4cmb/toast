# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..._libtoast import filter_poly2D as libtoast_filter_poly2D
from ..._libtoast import filter_polynomial as libtoast_filter_polynomial
from ...accelerator import ImplementationType, kernel, use_accel_jax
from .kernels_numpy import filter_poly2D_numpy, filter_polynomial_numpy

if use_accel_jax:
    from .kernels_jax import filter_poly2D_jax, filter_polynomial_jax


@kernel(impl=ImplementationType.DEFAULT)
def filter_polynomial(
    order,
    flags,
    signals_list,
    starts,
    stops,
    use_accel=False,
):
    """Kernel to fit and subtract a polynomial from one or more signals.

    Args:
        order (int):  The order of the polynomial.
        flags (numpy array, uint8):  The common flags to use for all signals
        signals_list (list of numpy array of double):  A list of float64 arrays containing the signals.
        starts (numpy array, int64):  The start samples of each scan.
        stops (numpy array, int64):  The stop samples of each scan.

    Returns:
        None

    """
    return libtoast_filter_polynomial(
        order,
        flags,
        signals_list,
        starts,
        stops,
        use_accel,
    )


@kernel(impl=ImplementationType.DEFAULT)
def filter_poly2D(
    det_groups,
    templates,
    signals,
    masks,
    coeff,
    use_accel=False,
):
    """Kernel for solving 2D polynomial coefficients at each sample.

    Args:
        det_groups (numpy array, int32):  The group index for each of the N_detectors detector index.
        templates (numpy array, float64):  The N_detectors x N_modes templates.
        signals (numpy array, float64):  The N_sample x N_detector data.
        masks (numpy array, uint8):  The N_sample x N_detector mask.
        coeff (numpy array, float64):  The N_sample x N_group x N_mode output coefficients.

    Returns:
        None

    """
    return libtoast_filter_poly2D(
        det_groups,
        templates,
        signals,
        masks,
        coeff,
        use_accel,
    )


@kernel(impl=ImplementationType.COMPILED, name="filter_polynomial")
def filter_polynomial_compiled(*args, use_accel=False):
    return libtoast_filter_polynomial(*args, use_accel)


@kernel(impl=ImplementationType.COMPILED, name="filter_poly2D")
def filter_poly2D_compiled(*args, use_accel=False):
    return libtoast_filter_poly2D(*args, use_accel)
