# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .math import qarray


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
        q_index = quat_index[idet]
        focalplane_det = focalplane[idet, :]
        for interval in intervals:
            # extract the interval slices
            interval_start = interval.first
            interval_end = interval.last + 1
            flags_samples = shared_flags[interval_start:interval_end]
            boresight_samples = boresight[interval_start:interval_end, :]
            # does the quaternion multiplication
            new_quats = qarray.mult(boresight_samples, focalplane_det)
            # masks bad samples
            good_samples = (flags_samples & shared_flag_mask) == 0
            quats[q_index, interval_start:interval_end, :] = np.where(
                good_samples[:, np.newaxis], new_quats, focalplane_det[np.newaxis, :]
            )


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix"); toast.tests.run("ops_demodulate"); toast.tests.run("ops_pointing_wcs")'
