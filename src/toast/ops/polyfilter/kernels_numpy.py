# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ...accelerator import ImplementationType, kernel


@kernel(impl=ImplementationType.NUMPY, name="filter_polynomial")
def filter_polynomial_numpy(order, flags, signals_list, starts, stops, use_accel=False):
    # validate order
    if order < 0:
        return

    # problem size
    n = flags.size
    norder = order + 1

    # converts signal into a numpy array to avoid having to loop over them
    signals = np.array(signals_list).T  # n*nsignal

    for start, stop in zip(starts, stops):
        # validates interval
        start = np.maximum(0, start)
        stop = np.minimum(n - 1, stop)
        if stop < start:
            continue
        scanlen = stop - start + 1

        # extracts the signals that will be impacted by this interval
        signals_interval = signals[start : (stop + 1), :]  # scanlen*nsignal

        # set aside the indexes of the zero flags to be used as a mask
        flags_interval = flags[start : (stop + 1)]  # scanlen
        zero_flags = np.where(flags_interval == 0)
        nb_zero_flags = zero_flags[0].size
        if nb_zero_flags == 0:
            continue

        # Build the full template matrix used to clean the signal.
        # We subtract the template value even from flagged samples to
        # support point source masking etc.
        # NOTE: we could recycle full_template if the previous scanlen is identical
        full_templates = np.empty(shape=(scanlen, norder))  # scanlen*norder
        # deals with order 0
        if norder > 0:
            full_templates[:, 0] = 1
        # deals with order 1
        if norder > 1:
            # defines x
            xstart = (1.0 / scanlen) - 1.0
            xstop = (1.0 / scanlen) + 1.0
            dx = 2.0 / scanlen
            x = np.arange(start=xstart, stop=xstop, step=dx)
            # sets ordre 1
            full_templates[:, 1] = x
        # deals with other orders
        # NOTE: this formulation is inherently sequential but this should be okay as `order` is likely small
        for iorder in range(2, norder):
            previous_previous_order = full_templates[:, iorder - 2]
            previous_order = full_templates[:, iorder - 1]
            full_templates[:, iorder] = (
                (2 * iorder - 1) * x * previous_order
                - (iorder - 1) * previous_previous_order
            ) / iorder

        # Assemble the flagged template matrix used in the linear regression
        masked_templates = full_templates[zero_flags]  # nb_zero_flags*norder

        # Square the template matrix for A^T.A
        invcov = np.dot(masked_templates.T, masked_templates)  # norder*norder

        # Project the signals against the templates
        masked_signals = signals_interval[zero_flags]  # nb_zero_flags*nsignal
        proj = np.dot(masked_templates.T, masked_signals)  # norder*nsignal

        # Fit the templates against the data
        # by minimizing the norm2 of the difference and the solution vector
        (x, _residue, _rank, _singular_values) = np.linalg.lstsq(
            invcov, proj, rcond=1e-3
        )  # norder*nsignal
        signals_interval -= np.dot(full_templates, x)

    # puts resulting signals back into list form
    for isignal, signal in enumerate(signals_list):
        signal[:] = signals[:, isignal]


@kernel(impl=ImplementationType.NUMPY, name="filter_poly2D")
def filter_poly2D_numpy(det_groups, templates, signals, masks, coeff, use_accel=False):
    ngroup = coeff.shape[1]
    nsample = signals.shape[0]
    for isample in range(nsample):
        for igroup in range(ngroup):
            good = det_groups == igroup
            mask = masks[isample, good]
            t = templates[good].T.copy() * mask
            proj = np.dot(t, signals[isample, good] * mask)
            ccinv = np.dot(t, t.T)
            coeff[isample, igroup] = np.linalg.lstsq(ccinv, proj, rcond=1.0e-6)[0]
