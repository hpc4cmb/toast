# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import jax
import jax.numpy as jnp
import numpy as np


def filter_polynomial_interval(flags_interval, signals_interval, order):
    """
    Process a single interval
    Return signals_interval modified by applying the polynomial filter
    """
    # problem size
    norder = order + 1
    scanlen = flags_interval.size

    # Build the full template matrix used to clean the signal.
    # We subtract the template value even from flagged samples to support point source masking etc.
    full_templates = jnp.empty(shape=(scanlen, norder))  # scanlen*norder
    # deals with order 0
    if norder > 0:
        # full_templates[:,0] = 1
        full_templates = full_templates.at[:, 0].set(1)
    # deals with order 1
    if norder > 1:
        # defines x
        indices = jnp.arange(start=0, stop=scanlen)
        xstart = (1.0 / scanlen) - 1.0
        dx = 2.0 / scanlen
        x = xstart + dx * indices
        # full_templates[:,1] = x
        full_templates = full_templates.at[:, 1].set(x)
    # deals with other orders
    # this loop will be unrolled but this should be okay as `order` is likely small
    for iorder in range(2, norder):
        previous_previous_order = full_templates[:, iorder - 2]
        previous_order = full_templates[:, iorder - 1]
        current_order = (
            (2 * iorder - 1) * x * previous_order
            - (iorder - 1) * previous_previous_order
        ) / iorder
        # full_templates[:,iorder] = current_order
        full_templates = full_templates.at[:, iorder].set(current_order)

    # Assemble the flagged template matrix used in the linear regression
    # by zeroing out the rows that are flagged
    valid_rows = flags_interval == 0
    masked_templates = jnp.where(
        valid_rows[:, jnp.newaxis], full_templates, 0.0
    )  # nb_zero_flags*norder

    # Square the template matrix for A^T.A
    invcov = jnp.dot(masked_templates.T, masked_templates)  # norder*norder

    # Project the signals against the templates
    # we do not mask flagged signals as they are multiplied with zeros anyway
    proj = jnp.dot(masked_templates.T, signals_interval)  # norder*nsignal

    # Fit the templates against the data
    (x, _residue, _rank, _singular_values) = jnp.linalg.lstsq(
        invcov, proj, rcond=1e-3
    )  # norder*nsignal

    # computes the value to be subtracted from the signals
    return signals_interval - jnp.dot(full_templates, x)


# JIT compiles the code
filter_polynomial_interval = jax.jit(
    filter_polynomial_interval, static_argnames=["order"]
)


def filter_polynomial(order, flags, signals_list, starts, stops):
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
    if order < 0:
        return

    # converts signals from a list into a numpy array to avoid having to loop over them
    signals = np.array(signals_list).T  # n*nsignal

    # loop over intervals, this is fine as long as there are only few intervals
    # TODO port to JaxIntervals, could be done with vmap and setting padding of flags_interval to 1
    for (start, stop) in zip(starts, stops):
        # validates interval
        start = np.maximum(0, start)
        stop = np.minimum(flags.size - 1, stop) + 1
        if stop <= start:
            continue
        # extracts the intervals from flags and signals
        flags_interval = flags[start:stop]  # scanlen
        signals_interval = signals[start:stop, :]  # scanlen*nsignal
        # updates signal interval
        signals[start:stop, :] = filter_polynomial_interval(
            flags_interval, signals_interval, order
        )

    # puts new signals back into the list
    for isignal, signal in enumerate(signals_list):
        signal[:] = signals[:, isignal]


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_polyfilter")'
