# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ...accelerator import ImplementationType, kernel


@kernel(impl=ImplementationType.NUMPY, name="noise_weight")
def noise_weight_numpy(
    det_data,
    det_data_index,
    intervals,
    detector_weights,
    use_accel,
):
    # Iterates over detectors and intervals
    n_det = det_data_index.size
    for idet in range(n_det):
        d_index = det_data_index[idet]
        detector_weight = detector_weights[idet]
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1
            # Multiply by the weight
            det_data[d_index, interval_start:interval_end] *= detector_weight
