# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import get_compile_time, select_implementation, ImplementationType
from .utils.intervals import JaxIntervals, ALL
from ..._libtoast import filter_polynomial as filter_polynomial_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def filter_polynomial_interval(flags_interval, signals_interval, order):
    """
    Process a single interval
    Return signals_interval modified by applying the polynomial filter
    """
    # problem size
    norder = order + 1
    scanlen = flags_interval.size
    print(f"DEBUG: jit-compiling 'filter_polynomial' scanlen:{scanlen} nsignal:{signals_interval.shape[1]} order:{order}")

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
    # by zeroing out the rows that are flagged
    valid_rows = (flags_interval == 0)
    masked_templates = jnp.where(valid_rows[:, jnp.newaxis], full_templates, 0.0) # nb_zero_flags*norder

    # Square the template matrix for A^T.A
    invcov = jnp.dot(masked_templates.T, masked_templates) # norder*norder

    # Project the signals against the templates
    # we do not mask flagged signals as they are multiplied with zeros anyway
    proj = jnp.dot(masked_templates.T, signals_interval) # norder*nsignal

    # Fit the templates against the data
    (x, _residue, _rank, _singular_values) = jnp.linalg.lstsq(invcov, proj, rcond=1e-3) # norder*nsignal

    # computes the value to be subtracted from the signals
    return signals_interval - jnp.dot(full_templates, x)

# JIT compiles the code
filter_polynomial_interval = jax.jit(filter_polynomial_interval, static_argnames=['order'])

def filter_polynomial_jax(order, flags, signals_list, starts, stops):
    """
    Fit and subtract a polynomial from one or more signals.

    Args:
        order (int):  The order of the polynomial.
        flags (numpy array, uint8):  The common flags to use for all signals
        signals_list (list of numpy array of double):  A list of float64 arrays containing the signals.
        starts (numpy array, int64):  The start samples of each scan.
        stops (numpy array, int64):  The stop samples of each scan.

    Returns:
        None: The signals are updated in place.
    """
    # validate order
    if (order < 0): return

    # converts signals from a list into a numpy array to avoid having to loop over them
    signals = np.array(signals_list).T # n*nsignal

    # loop over intervals, this is fine as long as there are only few intervals
    # TODO port to JaxIntervals, could be done with vmap and setting padding of flags_interval to 1
    for (start,stop) in zip(starts,stops):
        # validates interval
        start = np.maximum(0, start)
        stop = np.minimum(flags.size - 1, stop) + 1
        if (stop <= start): continue
        # extracts the intervals from flags and signals
        flags_interval = flags[start:stop] # scanlen
        signals_interval = signals[start:stop,:] # scanlen*nsignal        
        # updates signal interval
        signals[start:stop,:] = filter_polynomial_interval(flags_interval, signals_interval, order)

    # puts new signals back into the list
    for isignal, signal in enumerate(signals_list):
        signal[:] = signals[:,isignal]

#-------------------------------------------------------------------------------------------------
# NUMPY

def filter_polynomial_numpy(order, flags, signals_list, starts, stops):
    """
    Fit and subtract a polynomial from one or more signals.

    Args:
        order (int):  The order of the polynomial.
        flags (numpy array, uint8):  The common flags to use for all signals
        signals_list (list of numpy array of double):  A list of float64 arrays containing the signals.
        starts (numpy array, int64):  The start samples of each scan.
        stops (numpy array, int64):  The stop samples of each scan.

    Returns:
        None: The signals are updated in place.

    NOTE: port of `filter_polynomial` from compiled code to Numpy
    """
    # validate order
    if (order < 0): return

    # problem size
    n = flags.size
    norder = order + 1

    # converts signal into a numpy array to avoid having to loop over them
    signals = np.array(signals_list).T # n*nsignal

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
        # NOTE: we could recycle full_template if the previous scanlen is identical
        full_templates = np.empty(shape=(scanlen, norder)) # scanlen*norder
        # deals with order 0
        if norder > 0: 
            full_templates[:,0] = 1
        # deals with order 1
        if norder > 1: 
            # defines x
            xstart = (1. / scanlen) - 1.
            xstop = (1. / scanlen) + 1.
            dx = 2. / scanlen
            x = np.arange(start=xstart, stop=xstop, step=dx)
            # sets ordre 1
            full_templates[:,1] = x
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
void toast::filter_polynomial(int64_t order, size_t n, uint8_t * flags,
                              std::vector <double *> const & signals, size_t nscan,
                              int64_t const * starts, int64_t const * stops) {
    if (order < 0) return;

    int nsignal = signals.size();
    int norder = order + 1;

    char upper = 'U';
    char lower = 'L';
    char notrans = 'N';
    char trans = 'T';
    double fzero = 0.0;
    double fone = 1.0;

    for (size_t iscan = 0; iscan < nscan; ++iscan) 
    {
        int64_t start = starts[iscan];
        int64_t stop = stops[iscan];
        if (start < 0) start = 0;
        if (stop > n - 1) stop = n - 1;
        if (stop < start) continue;
        int scanlen = stop - start + 1;

        int ngood = 0;
        for (size_t i = 0; i < scanlen; ++i) 
        {
            if (flags[start + i] == 0) ngood++;
        }
        if (ngood == 0) continue;

        // Build the full template matrix used to clean the signal.
        // We subtract the template value even from flagged samples to
        // support point source masking etc.
        toast::AlignedVector <double> full_templates(scanlen * norder);

        double dx = 2. / scanlen;
        double xstart = 0.5 * dx - 1;
        double * current, * last, * lastlast;

        for (size_t iorder = 0; iorder < norder; ++iorder) 
        {
            current = &full_templates[iorder * scanlen];
            if (iorder == 0) 
            {
                for (size_t i = 0; i < scanlen; ++i) 
                {
                    current[i] = 1;
                }
            } 
            else if (iorder == 1) 
            {
                for (size_t i = 0; i < scanlen; ++i) 
                {
                    const double x = xstart + i * dx;
                    current[i] = x;
                }
            } 
            else 
            {
                last = &full_templates[(iorder - 1) * scanlen];
                lastlast = &full_templates[(iorder - 2) * scanlen];
                double orderinv = 1. / iorder;

                for (size_t i = 0; i < scanlen; ++i) 
                {
                    const double x = xstart + i * dx;
                    current[i] = ((2 * iorder - 1) * x * last[i] - (iorder - 1) * lastlast[i]) * orderinv;
                }
            }
        }

        // Assemble the flagged template matrix used in the linear
        // regression

        toast::AlignedVector <double> masked_templates(ngood * norder);

        for (size_t iorder = 0; iorder < norder; ++iorder) 
        {
            size_t offset = iorder * ngood;
            current = &full_templates[iorder * scanlen];
            for (size_t i = 0; i < scanlen; ++i) 
            {
                if (flags[start + i] == 0) 
                {
                    masked_templates[offset++] = current[i];
                }
            }
        }

        // Square the template matrix for A^T.A
        toast::AlignedVector <double> invcov(norder * norder);
        toast::LinearAlgebra::syrk(upper, trans, norder, ngood, fone,
                                   masked_templates.data(), ngood, fzero, invcov.data(),
                                   norder);

        // Project the signals against the templates

        toast::AlignedVector <double> masked_signals(ngood * nsignal);

        for (size_t isignal = 0; isignal < nsignal; ++isignal) 
        {
            size_t offset = isignal * ngood;
            double * signal = signals[isignal] + start;
            for (int64_t i = 0; i < scanlen; ++i) 
            {
                if (flags[start + i] == 0) 
                {
                    masked_signals[offset++] = signal[i];
                }
            }
        }

        toast::AlignedVector <double> proj(norder * nsignal);

        toast::LinearAlgebra::gemm(trans, notrans, norder, nsignal, ngood,
                                   fone, masked_templates.data(), ngood,
                                   masked_signals.data(), ngood,
                                   fzero, proj.data(), norder);

        // Symmetrize the covariance matrix, dgells is written for
        // generic matrices

        for (size_t row = 0; row < norder; ++row) 
        {
            for (size_t col = row + 1; col < norder; ++col) 
            {
                invcov[col + row * norder] = invcov[row + col * norder];
            }
        }

        // Fit the templates against the data.
        // DGELSS minimizes the norm of the difference and the solution vector
        // and overwrites proj with the fitting coefficients.
        int rank, info;
        double rcond_limit = 1e-3;
        int LWORK = toast::LinearAlgebra::gelss_buffersize(norder, norder, nsignal,
                                                           norder, norder, rcond_limit);
        toast::AlignedVector <double> WORK(LWORK);
        toast::AlignedVector <double> singular_values(norder);
        toast::LinearAlgebra::gelss(
            norder, norder, nsignal, invcov.data(), norder,
            proj.data(), norder, singular_values.data(), rcond_limit,
            &rank, WORK.data(), LWORK, &info);

        for (int iorder = 0; iorder < norder; ++iorder) 
        {
            double * temp = &full_templates[iorder * scanlen];
            for (int isignal = 0; isignal < nsignal; ++isignal) 
            {
                double * signal = &signals[isignal][start];
                double amp = proj[iorder + isignal * norder];
                if (toast::is_aligned(signal) && toast::is_aligned(temp)) 
                {
                    for (size_t i = 0; i < scanlen; ++i) 
                    {
                        signal[i] -= amp * temp[i];
                    }
                } 
                else 
                {
                    for (size_t i = 0; i < scanlen; ++i) 
                    {
                        signal[i] -= amp * temp[i];
                    }
                }
            }
        }
    }
}
"""

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
filter_polynomial = select_implementation(filter_polynomial_compiled, 
                                          filter_polynomial_numpy, 
                                          filter_polynomial_jax)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
filter_polynomial = get_compile_time(filter_polynomial)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_polyfilter")'
