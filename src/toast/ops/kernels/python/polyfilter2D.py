# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


# fails comparison test in polyfilter
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
    # problem size
    (nsample, ngroup, _nmode) = coeff.shape

    # For each sample
    for isamp in range(nsample):
        # For each group of detectors
        for igroup in range(ngroup):
            # Masks detectors not in this group
            good = det_groups == igroup
            mask = masks[isamp, good]
            t = templates[good].T.copy() * mask
            # Projection
            proj = np.dot(t, signals[isamp, good] * mask)
            ccinv = np.dot(t, t.T)
            # Fits the coefficients
            coeff[isamp, igroup] = np.linalg.lstsq(ccinv, proj, rcond=1e-3)[0]


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_polyfilter")'
