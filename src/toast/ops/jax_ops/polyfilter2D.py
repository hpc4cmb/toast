# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import get_compile_time, select_implementation, ImplementationType
from ..._libtoast import filter_poly2D as filter_poly2D_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def filter_poly2D_jax(order, flags, signals_list, starts, stops):
    """
    Solves for 2D polynomial coefficients at each sample.

    Args:
        det_groups (numpy array, int32):  The group index for each detector index.
        templates (numpy array, float64):  The N_detectors x N_modes templates.
        signals (numpy array, float64):  The N_sample x N_detector data.
        masks (numpy array, uint8):  The N_sample x N_detector mask.
        coeff (numpy array, float64):  The N_sample x N_group x N_mode output coefficients.

    Returns:
        None: The signals are updated in place.

    NOTE: port of `filter_poly2D` from compiled code to JAX
    """
    # TODO

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

    NOTE: port of `filter_poly2D` from compiled code to Numpy
    """
    # problem size
    nsample = signals.shape[0]
    ndet = signals.shape[1]
    ngroup = coeff.shape[1]
    nmode = templates.shape[1]
    #(nsample, ngroup, nmode) = coeff.shape
    #ndet = det_groups.size

    # For each sample
    for isamp in range(nsample):
        # For each group of detectors
        for igroup in range(ngroup): 
            # Gets solve buffers
            rhs = np.zeros(nmode)
            A = np.zeros((nmode, nmode))

            # TODO try and rewrite using the following formulas
            # TODO this will have to take `det_groups` into account 
            # rhs =  (mask * templates).T  @  (mask * signals.T)
            # A = (mask * templates).T  @  (mask * templates)

            # For each detector
            for idet in range(ndet): 
                # This detectors is not in this group
                if (det_groups[idet] != igroup) : continue

                # Mask value for this detector
                det_mask = 0.0 if (masks[isamp,idet] == 0) else 1.0

                # Signal value for this detector
                det_sig = signals[isamp,idet]

                for imode in range(nmode): 
                    rhs[imode] += templates[idet,imode] * det_sig * det_mask

                    for jmode in range(imode, nmode): 
                        val = templates[idet,imode] * templates[idet,jmode] * det_mask
                        A[imode,jmode] += val
                        if (jmode > imode): 
                            A[jmode,imode] += val
    
            # gets the fitting coefficients.
            (x, _residue, _rank, _singular_values) = np.linalg.lstsq(A, rhs, rcond=1e-3)
            coeff[isamp,igroup,:] = x

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
                                      filter_poly2D_jax, 
                                      default_implementationType=ImplementationType.NUMPY)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
#filter_poly2D = get_compile_time(filter_poly2D)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_polyfilter")'
