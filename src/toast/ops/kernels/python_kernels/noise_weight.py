# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

def noise_weight(
    det_data, det_data_index, intervals, detector_weights, use_accel
):
    """
    multiplies det_data by the weighs in detector_weights

    Args:
        det_data (array, float): size ???*n_samp
        det_data_index (array, int): The indexes of the det_data (size n_det)
        intervals (array, Interval): The intervals to modify (size n_view)
        detector_weights (list, double): The weight to be used for each detcetor (size n_det)
        use_accel (bool): should we use an accelerator

    Returns:
        None: det_data is updated in place
    """
    # iterates on detectors and intervals
    n_det = det_data_index.size
    for idet in range(n_det):
        d_index = det_data_index[idet]
        detector_weight = detector_weights[idet]
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1
            # applies the multiplication
            det_data[d_index, interval_start:interval_end] *= detector_weight
