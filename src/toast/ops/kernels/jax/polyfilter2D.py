# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.maps import xmap as jax_xmap

from ....jax.mutableArray import MutableJaxArray


def filter_poly2D_sample_group(
    igroup, det_groups, templates, signals_sample, masks_sample
):
    """
    filter_poly2D for a given group and sample

    Args:
        igroup (uint): index of the group
        det_groups (numpy array, int32):  The group index for each of the N_detectors detector index.
        templates (numpy array, float64):  The N_detectors x N_modes templates.
        signals_samples (numpy array, float64):  The N_detector data.
        masks_sample (numpy array, uint8):  The N_detector mask.

    Returns:
        coeff (numpy array, float64):  The N_mode output coefficients.
    """
    # Masks detectors not in this group
    masks_group = jnp.where(det_groups != igroup, 0.0, masks_sample)

    # rhs = (mask * templates).T  @  (mask * signals.T) = (mask * templates).T  @  signals.T
    # A = (mask * templates).T  @  (mask * templates)
    masked_template = masks_group[:, jnp.newaxis] * templates  # N_detectors x N_modes
    rhs = jnp.dot(masked_template.T, signals_sample.T)  # N_modes
    A = jnp.dot(masked_template.T, masked_template)  # N_modes x N_modes

    # Fits the coefficients
    (coeff_sample_group, _residue, _rank, _singular_values) = jnp.linalg.lstsq(
        A, rhs, rcond=1e-3
    )
    # Sometimes the mask will be all zeroes in which case A=0 and rhs=0 causing the coeffs to be nan
    # We thus replace nans with 0s (Numpy does it by default)
    coeff_sample_group = jnp.nan_to_num(coeff_sample_group, nan=0.0)

    return coeff_sample_group


def filter_poly2D_coeffs(ngroup, det_groups, templates, signals, masks):
    """
    Args:
        ngroup (uint): number of groups
        det_groups (numpy array, int32):  The group index for each of the N_detectors detector index.
        templates (numpy array, float64):  The N_detectors x N_modes templates.
        signals (numpy array, float64):  The N_sample x N_detector data.
        masks (numpy array, uint8):  The N_sample x N_detector mask.

    Returns:
        coeff (numpy array, float64):  The N_sample x N_group x N_mode output coefficients.
    """
    # batch on group and sample dimenssions
    filter_poly2D_sample_group_batched = jax_xmap(
        filter_poly2D_sample_group,
        in_axes=[
            ["group"],  # igroup
            [...],  # det_groups
            [...],  # templates
            ["sample", ...],  # signals
            ["sample", ...],
        ],  # masks
        out_axes=["sample", "group", ...],
    )

    # runs for all the groups and samples simultaneously
    igroup = jnp.arange(start=0, stop=ngroup)
    return filter_poly2D_sample_group_batched(
        igroup, det_groups, templates, signals, masks
    )


# JIT compiles the code
filter_poly2D_coeffs = jax.jit(filter_poly2D_coeffs, static_argnames="ngroup")


def filter_poly2D(det_groups, templates, signals, masks, coeff):
    """
    Solves for 2D polynomial coefficients at each sample.

    Args:
        det_groups (numpy array, int32):  The group index for each of the N_detectors detector index.
        templates (numpy array, float64):  The N_detectors x N_modes templates.
        signals (numpy array, float64):  The N_sample x N_detector data.
        masks (numpy array, uint8):  The N_sample x N_detector mask.
        coeff (numpy array, float64):  The N_sample x N_group x N_mode output coefficients.

    Returns:
        None: The coefficients are updated in place.
    """
    # does computation with JAX function and assign result to coeffs
    ngroup = coeff.shape[1]
    det_groups_input = MutableJaxArray.to_array(det_groups)
    templates_input = MutableJaxArray.to_array(templates)
    signals_input = MutableJaxArray.to_array(signals)
    masks_input = MutableJaxArray.to_array(masks)
    coeff[:] = filter_poly2D_coeffs(
        ngroup, det_groups_input, templates_input, signals_input, masks_input
    )


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_polyfilter")'
