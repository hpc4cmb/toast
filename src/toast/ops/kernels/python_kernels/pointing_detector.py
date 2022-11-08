# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from .math import qarray

def pointing_detector_inner(flag, boresight, focalplane, mask):
    """
    Process a single detector and a single sample inside an interval.

    Args:
        flag (uint8)
        boresight (array, double): size 4
        focalplane (array, double): size 4
        mask (uint8)

    Returns:
        quats (array, double): size 4
    """
    if (flag & mask) == 0:
        quats = qarray.mult_one_one(boresight, focalplane)
    else:
        quats = focalplane


def pointing_detector(
    focalplane,
    boresight,
    quat_index,
    quats,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel,
):
    """
    Args:
        focalplane (array, double): size n_det*4
        boresight (array, double): size n_samp*4
        quat_index (array, int): size n_det
        quats (array, double): size ???*n_samp*4
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        use_accel (bool): should weuse the accelerator

    Returns:
        None (the result is put in quats).
    """
    # iterates on all detectors and all intervals
    n_det = quat_index.size
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1
            for isamp in range(interval_start, interval_end):
                q_index = quat_index[idet]
                quats[q_index, isamp, :] = pointing_detector_inner(
                    shared_flags[isamp],
                    boresight[isamp, :],
                    focalplane[idet, :],
                    shared_flag_mask,
                )

def _py_pointing_detector(
    self,
    fp_quats,
    bore_data,
    quat_indx,
    quat_data,
    intr_data,
    flag_data,
):
    """Internal python implementation for comparison tests."""
    for idet in range(len(quat_indx)):
        qidx = quat_indx[idet]
        for vw in intr_data:
            samples = slice(vw.first, vw.last + 1, 1)
            bore = np.array(bore_data[samples])
            if self.shared_flags is not None:
                good = (flag_data[samples] & self.shared_flag_mask) == 0
                bore[np.invert(good)] = np.array([0, 0, 0, 1], dtype=np.float64)
            quat_data[qidx][samples] = qarray.mult_one_one(bore, fp_quats[idet])