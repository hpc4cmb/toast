# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from .utils import get_compile_time, select_implementation, ImplementationType
from ..._libtoast import filter_poly2D as filter_poly2D_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def filter_poly2D_sample_group(igroup, det_groups, templates, signals_sample, masks_sample):
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
    masked_template = masks_group[:,jnp.newaxis] * templates # N_detectors x N_modes
    rhs = jnp.dot(masked_template.T, signals_sample.T) # N_modes
    A = jnp.dot(masked_template.T, masked_template) # N_modes x N_modes

    # Fits the coefficients
    (coeff_sample_group, _residue, _rank, _singular_values) = jnp.linalg.lstsq(A, rhs, rcond=1e-3)
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
    # problem size
    (nsample, ndet) = masks.shape
    nmode = templates.shape[1]
    print(f"DEBUG: jit-compiling 'filter_poly2D' nsample:{nsample} ngroup:{ngroup} nmode:{nmode} ndet:{ndet}")

    # batch on group and sample dimenssions
    filter_poly2D_sample_group_batched = jax_xmap(filter_poly2D_sample_group, 
                                                  in_axes=[['group'], # igroup
                                                           [...], # det_groups
                                                           [...], # templates
                                                           ['sample',...], # signals
                                                           ['sample',...]], # masks
                                                  out_axes=['sample','group',...])

    # runs for all the groups and samples simultaneously
    igroup = jnp.arange(start=0, stop=ngroup)
    return filter_poly2D_sample_group_batched(igroup, det_groups, templates, signals, masks)

# JIT compiles the code
filter_poly2D_coeffs = jax.jit(filter_poly2D_coeffs, static_argnames='ngroup')

def filter_poly2D_jax(det_groups, templates, signals, masks, coeff):
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
    coeff[:] = filter_poly2D_coeffs(ngroup, det_groups, templates, signals, masks)

#-------------------------------------------------------------------------------------------------
# NUMPY

def filter_poly2D_numpy(det_groups, templates, signals, masks, coeff):
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
    # problem size
    (nsample, ngroup, _nmode) = coeff.shape
    _ndet = det_groups.size

    # For each sample
    for isamp in range(nsample):
        masks_sample = masks[isamp,:] # N_detector
        signals_sample = signals[isamp,:] # N_detector

        # For each group of detectors
        for igroup in range(ngroup): 
            # Masks detectors not in this group
            masks_group = np.where(det_groups != igroup, 0.0, masks_sample)

            # rhs = (mask * templates).T  @  (mask * signals.T) = (mask * templates).T  @  signals.T
            # A = (mask * templates).T  @  (mask * templates)
            masked_template = masks_group[:,np.newaxis] * templates # N_detectors x N_modes
            rhs = np.dot(masked_template.T, signals_sample.T) # N_modes
            A = np.dot(masked_template.T, masked_template) # N_modes x N_modes

            # Fits the coefficients
            (coeff_sample_group, _residue, _rank, _singular_values) = np.linalg.lstsq(A, rhs, rcond=1e-3)
            coeff[isamp,igroup,:] = coeff_sample_group

#-------------------------------------------------------------------------------------------------
# C++

"""
void toast::filter_poly2D_solve(
    int64_t nsample, int32_t ndet, int32_t ngroup, int32_t nmode,
    int32_t const * det_group, double const * templates, uint8_t const * masks,
    double const * signals, double * coeff) {
    // For each sample, solve for the regression coefficients.
    // The templates are flat packed across (detectors, modes).
    // The mask is flat packed across (samples, detectors).
    // The signals are flat packed across (samples, detectors).
    // The coefficients are flat packed across (samples, groups, modes).

    #pragma omp parallel default(shared)
    {
        // These are all thread-private
        int inmode = (int)nmode;
        int rank;
        int info;
        int one = 1;
        double rcond_limit = 1e-3;
        int LWORK = toast::LinearAlgebra::gelss_buffersize(inmode, inmode, one, inmode,
                                                           inmode, rcond_limit);
        toast::AlignedVector <double> rhs(nmode);
        toast::AlignedVector <double> A(nmode * nmode);
        toast::AlignedVector <double> singular_values(nmode);
        toast::AlignedVector <double> WORK(LWORK);

        #pragma omp for schedule(static)
        for (int64_t isamp = 0; isamp < nsample; ++isamp) {
            // For this sample...
            for (int32_t igroup = 0; igroup < ngroup; ++igroup) {
                // For this group of detectors...
                // Zero out solve buffers
                std::fill(rhs.begin(), rhs.end(), 0.0);
                std::fill(A.begin(), A.end(), 0.0);

                // Accumulate the RHS and design matrix one detector at a time.  Imagine
                // we have 2 detectors and 3 modes:
                //
                //       mask = [m_1 m_2] (These are either 0 or 1, so m_1 * m_1 == m_1)
                //
                //  templates = [[a_1 b_1 c_1],
                //               [a_2 b_2 c_2]]
                //
                //     signal = [s_1 s_2]
                //
                //        RHS =  (mask * templates)^T  X  (mask * signals^T)
                //            =  [[a_1 * s_1 * m_1 + a_2 * s_2 * m_2],
                //                [b_1 * s_1 * m_1 + b_2 * s_2 * m_2],
                //                [c_1 * s_1 * m_1 + c_2 * s_2 * m_2]]
                //
                //          A = (mask * templates)^T  X  (mask * templates)
                //            = [ [a_1 * a_1 * m_1 + a_2 * a_2 * m_2,
                //                 a_1 * b_1 * m_1 + a_2 * b_2 * m_2,
                //                 a_1 * c_1 * m_1 + a_2 * c_2 * m_2],
                //                [b_1 * a_1 * m_1 + b_2 * a_2 * m_2,
                //                 b_1 * b_1 * m_1 + b_2 * b_2 * m_2,
                //                 b_1 * c_1 * m_1 + b_2 * c_2 * m_2],
                //                [c_1 * a_1 * m_1 + c_2 * a_2 * m_2,
                //                 c_1 * b_1 * m_1 + c_2 * b_2 * m_2,
                //                 c_1 * c_1 * m_1 + c_2 * c_2 * m_2] ]
                //

                for (int32_t idet = 0; idet < ndet; ++idet) {
                    // For each detector...
                    if (det_group[idet] != igroup) {
                        // This detectors is not in this group
                        continue;
                    }

                    // Mask value for this detector
                    double det_mask = (masks[isamp * ndet + idet] == 0) ? 0.0 : 1.0;

                    // Signal value for this detector
                    double det_sig = signals[isamp * ndet + idet];

                    for (int32_t imode = 0; imode < nmode; ++imode) {
                        int32_t tmpl_off = idet * nmode;
                        rhs[imode] += templates[tmpl_off + imode] * det_sig * det_mask;

                        for (int32_t jmode = imode; jmode < nmode; ++jmode) {
                            double val = templates[tmpl_off + imode] *
                                         templates[tmpl_off + jmode] * det_mask;
                            A[imode * nmode + jmode] += val;
                            if (jmode > imode) {
                                A[jmode * nmode + imode] += val;
                            }
                        }
                    }
                }

                // DGELSS will overwrite RHS with the fitting
                // coefficients.  A is overwritten with
                // singular vectors.
                toast::LinearAlgebra::gelss(
                    inmode, inmode, one, A.data(), inmode,
                    rhs.data(), inmode, singular_values.data(), rcond_limit,
                    &rank, WORK.data(), LWORK, &info);
                int64_t offset = isamp * (ngroup * nmode) + igroup * nmode;
                if (info == 0) {
                    // Solve was successful
                    for (int64_t m = 0; m < nmode; ++m) {
                        coeff[offset + m] = rhs[m];
                    }
                } else {
                    // Failed
                    for (int64_t m = 0; m < nmode; ++m) {
                        coeff[offset + m] = 0.0;
                    }
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
filter_poly2D = select_implementation(filter_poly2D_compiled, 
                                      filter_poly2D_numpy, 
                                      filter_poly2D_jax)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
filter_poly2D = get_compile_time(filter_poly2D)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_polyfilter")'
