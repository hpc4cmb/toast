# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ... import qarray as qa
from ...accelerator import ImplementationType, kernel


@kernel(impl=ImplementationType.NUMPY, name="pointing_detector")
def pointing_detector_numpy(
    focalplane,
    boresight,
    quat_index,
    quats,
    intervals,
    shared_flags,
    shared_flag_mask,
    use_accel=False,
):
    for idet in range(len(quat_index)):
        qidx = quat_index[idet]
        for vw in intervals:
            samples = slice(vw.first, vw.last + 1, 1)
            bore = np.array(boresight[samples])
            if shared_flags is not None:
                good = (shared_flags[samples] & shared_flag_mask) == 0
                bore[np.invert(good)] = np.array([0, 0, 0, 1], dtype=np.float64)
            quats[qidx][samples] = qa.mult(bore, focalplane[idet])
