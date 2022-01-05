# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import get_compile_time
from ..._libtoast import filter_poly2D as filter_poly2D_compiled

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

def filter_poly2D(order, flags, signals, starts, stops, use_compiled=True):
    """
    Used in test to select the `filter_poly2D` implementation
    TODO: this is for test purposes
    """
    if use_compiled: filter_poly2D_compiled(order, flags, signals, starts, stops)
    else: filter_poly2D_jax(order, flags, signals, starts, stops)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
filter_poly2D = get_compile_time(filter_poly2D)

#-------------------------------------------------------------------------------------------------
# JAX

def filter_poly2D_interval(flags_interval, signals_interval, order):
    """
    Process a single interval
    Return signals_interval modified by applying the polynomial filter
    """
    # problem size
    norder = order + 1
    scanlen = flags_interval.size
    print(f"DEBUG: jit-compiling scanlen:{scanlen} nsignal:{signals_interval.shape[1]} order:{order}")

    # Build the full template matrix used to clean the signal.
    # We subtract the template value even from flagged samples to support point source masking etc.
    full_templates = jnp.empty(shape=(scanlen, norder)) # scanlen*norder
    # deals with order 0
    if norder > 0: 
        # full_templates[:,0] = 1
        full_templates = full_templates.at[:,0].set(1)
    # deals with order 1
    if norder > 1:
        # defines x
        indices = jnp.arange(start=0, stop=scanlen)
        xstart = (1. / scanlen) - 1.
        dx = 2. / scanlen
        x = xstart + dx*indices
        # full_templates[:,1] = x
        full_templates = full_templates.at[:,1].set(x)
    # deals with other orders
    # this loop will be unrolled but this should be okay as `order` is likely small
    for iorder in range(2,norder):
        previous_previous_order = full_templates[:,iorder-2]
        previous_order = full_templates[:,iorder-1]
        current_order = ((2 * iorder - 1) * x * previous_order - (iorder - 1) * previous_previous_order) / iorder
        # full_templates[:,iorder] = current_order
        full_templates = full_templates.at[:,iorder].set(current_order)

    # Assemble the flagged template matrix used in the linear regression
    # builds masks to operate where flags are set to 0
    mask = flags_interval == 0
    # zero out the rows that are flagged or outside the interval
    masked_templates = full_templates * mask[:, jnp.newaxis] # nb_zero_flags*norder

    # Square the template matrix for A^T.A
    invcov = jnp.dot(masked_templates.T, masked_templates) # norder*norder

    # Project the signals against the templates
    # we do not mask flagged signals as they are multiplied with zeros anyway
    proj = jnp.dot(masked_templates.T, signals_interval) # norder*nsignal

    # Fit the templates against the data
    (x, _residue, _rank, _singular_values) = jnp.linalg.lstsq(invcov, proj, rcond=1e-3) # norder*nsignal

    # computes the value to be subtracted from the signals
    return signals_interval - jnp.dot(full_templates, x)

# JIT compiles the JAX function
filter_poly2D_interval = jax.jit(filter_poly2D_interval, static_argnames=['order'])
# dummy call to warm-up the jit
dummy_order = 1
dummy_scanlen = 2
dummy_nsignal = 1
dummy_flags = np.zeros(shape=(dummy_scanlen,))
dummy_interval = np.zeros(shape=(dummy_scanlen,dummy_nsignal))
filter_poly2D_interval(dummy_flags, dummy_interval, dummy_order)

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
    # validate order
    if (order < 0): return

    # converts signal into a numpy array to avoid having to loop over them
    signals = np.array(signals_list).T # n*nsignal

    # loop over intervals, this is fine as long as there are only few intervals
    print(f"DEBUG: nb intervals:{starts.size}")
    for (start,stop) in zip(starts,stops):
        # validates interval
        start = np.maximum(0, start)
        stop = np.minimum(flags.size - 1, stop)
        if (stop < start): continue
        # extracts the intervals from flags and signals
        flags_interval = flags[start:(stop+1)] # scanlen
        signals_interval = signals[start:(stop+1),:] # scanlen*nsignal        
        # updates signal interval
        signals_interval[:] = filter_poly2D_interval(flags_interval, signals_interval, order)

    # puts new signals back into the list
    for isignal, signal in enumerate(signals_list):
        signal[:] = signals[:,isignal]

#-------------------------------------------------------------------------------------------------
# NUMPY

def filter_poly2D_numpy(order, flags, signals_list, starts, stops):
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

    NOTE: port of `filter_poly2D` from compiled code to Numpy
    """
    # validate order
    if (order < 0): return

    # problem size
    n = flags.size
    norder = order + 1

    # converts signal into a numpy array to avoid having to loop over them
    signals = np.array(signals_list).T # n*nsignal

    # NOTE: that loop is parallel in the C++ code
    for (start, stop) in zip(starts, stops):
        # validates interval
        start = np.maximum(0, start)
        stop = np.minimum(n-1, stop)
        if (stop < start): continue
        scanlen = stop - start + 1

        # extracts the signals that will be impacted by this interval
        signals_interval = signals[start:(stop+1),:] # scanlen*nsignal

        # set aside the indexes of the zero flags to be used as a mask
        flags_interval = flags[start:(stop+1)] # scanlen
        zero_flags = np.where(flags_interval == 0)
        nb_zero_flags = zero_flags[0].size
        if (nb_zero_flags == 0): continue

        # Build the full template matrix used to clean the signal.
        # We subtract the template value even from flagged samples to
        # support point source masking etc.
        full_templates = np.zeros(shape=(scanlen, norder)) # scanlen*norder
        xstart = (1. / scanlen) - 1.
        xstop = (1. / scanlen) + 1.
        dx = 2. / scanlen
        x = np.arange(start=xstart, stop=xstop, step=dx)
        # deals with order 0
        if norder > 0: full_templates[:,0] = 1
        # deals with order 1
        if norder > 1: full_templates[:,1] = x
        # deals with other orders
        # NOTE: this formulation is inherently sequential but this should be okay as `order` is likely small
        for iorder in range(2,norder):
            previous_previous_order = full_templates[:,iorder-2]
            previous_order = full_templates[:,iorder-1]
            full_templates[:,iorder] = ((2 * iorder - 1) * x * previous_order - (iorder - 1) * previous_previous_order) / iorder
        
        # Assemble the flagged template matrix used in the linear regression
        masked_templates = full_templates[zero_flags] # nb_zero_flags*norder

        # Square the template matrix for A^T.A
        invcov = np.dot(masked_templates.T, masked_templates) # norder*norder

        # Project the signals against the templates
        masked_signals = signals_interval[zero_flags] # nb_zero_flags*nsignal
        proj = np.dot(masked_templates.T, masked_signals) # norder*nsignal

        # Fit the templates against the data
        # by minimizing the norm2 of the difference and the solution vector
        (x, _residue, _rank, _singular_values) = np.linalg.lstsq(invcov, proj, rcond=1e-3) # norder*nsignal
        signals_interval -= np.dot(full_templates, x)
    
    # puts resulting signals back into list form
    for isignal, signal in enumerate(signals_list):
        signal[:] = signals[:,isignal]

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