# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


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
        masks_sample = masks[isamp, :]  # N_detector
        signals_sample = signals[isamp, :]  # N_detector

        # For each group of detectors
        for igroup in range(ngroup):
            # Masks detectors not in this group
            masks_group = np.where(det_groups != igroup, 0.0, masks_sample)

            # rhs = (mask * templates).T  @  (mask * signals.T) = (mask * templates).T  @  signals.T
            # A = (mask * templates).T  @  (mask * templates)
            masked_template = (
                masks_group[:, np.newaxis] * templates
            )  # N_detectors x N_modes
            rhs = np.dot(masked_template.T, signals_sample.T)  # N_modes
            A = np.dot(masked_template.T, masked_template)  # N_modes x N_modes

            # Fits the coefficients
            (coeff_sample_group, _residue, _rank, _singular_values) = np.linalg.lstsq(
                A, rhs, rcond=1e-3
            )
            coeff[isamp, igroup, :] = coeff_sample_group


def _py_filterpoly2D():
    gt.start("Poly2D:  Solve templates (with python)")
    for isample in range(nsample):
        for group, igroup in group_ids.items():
            good = group_det == igroup
            mask = masks[isample, good]
            t = templates[good].T.copy() * mask
            proj = np.dot(t, signals[isample, good] * mask)
            ccinv = np.dot(t, t.T)
            coeff[isample, igroup] = np.linalg.lstsq(ccinv, proj, rcond=None)[0]
    gt.stop("Poly2D:  Solve templates (with python)")
