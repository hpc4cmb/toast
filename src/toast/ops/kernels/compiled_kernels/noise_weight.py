# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..python_kernels import noise_weight as noise_weight_python


def noise_weight(det_data, det_data_index, intervals, detector_weights, use_accel):
    # TODO make a C++ implementation with support for OpenMP target offload
    if use_accel:
        raise RuntimeError(
            "noise_weight_compiled: there is currently no OpenMP offload version of noise_weight"
        )
    noise_weight_python(
        det_data, det_data_index, intervals, detector_weights, use_accel
    )
