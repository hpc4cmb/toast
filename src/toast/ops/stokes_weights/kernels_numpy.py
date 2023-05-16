# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ... import qarray as qa
from ...accelerator import ImplementationType, kernel


@kernel(impl=ImplementationType.NUMPY, name="stokes_weights_IQU")
def stokes_weights_IQU_numpy(
    quat_index,
    quats,
    weight_index,
    weights,
    hwp,
    intervals,
    epsilon,
    gamma,
    cal,
    IAU,
    use_accel,
):
    if hwp is not None and len(hwp) == 0:
        hwp = None

    if IAU:
        U_sign = -1.0
    else:
        U_sign = 1.0

    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)
    for idet in range(len(quat_index)):
        qidx = quat_index[idet]
        widx = weight_index[idet]
        eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet])
        for vw in intervals:
            samples = slice(vw.first, vw.last + 1, 1)
            vd = qa.rotate(quats[qidx][samples], zaxis)
            vo = qa.rotate(quats[qidx][samples], xaxis)

            # The vector orthogonal to the line of sight that is parallel
            # to the local meridian.
            dir_ang = np.arctan2(vd[:, 1], vd[:, 0])
            dir_r = np.sqrt(1.0 - vd[:, 2] * vd[:, 2])
            vm_z = -dir_r
            vm_x = vd[:, 2] * np.cos(dir_ang)
            vm_y = vd[:, 2] * np.sin(dir_ang)

            # Compute the rotation angle from the meridian vector to the
            # orientation vector.  The direction vector is normal to the plane
            # containing these two vectors, so the rotation angle is:
            #
            # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
            #
            alpha_y = (
                vd[:, 0] * (vm_y * vo[:, 2] - vm_z * vo[:, 1])
                - vd[:, 1] * (vm_x * vo[:, 2] - vm_z * vo[:, 0])
                + vd[:, 2] * (vm_x * vo[:, 1] - vm_y * vo[:, 0])
            )
            alpha_x = vm_x * vo[:, 0] + vm_y * vo[:, 1] + vm_z * vo[:, 2]

            alpha = np.arctan2(alpha_y, alpha_x)

            if hwp is None:
                ang = 2.0 * alpha
            else:
                ang = 2.0 * (alpha + 2.0 * (hwp - gamma[idet]))

            weights[widx][samples, 0] = cal
            weights[widx][samples, 1] = cal * eta * np.cos(ang)
            weights[widx][samples, 2] = -cal * eta * np.sin(ang) * U_sign


@kernel(impl=ImplementationType.NUMPY, name="stokes_weights_I")
def stokes_weights_I_numpy(weight_index, weights, intervals, cal, use_accel):
    for idet in range(len(weight_index)):
        widx = weight_index[idet]
        for vw in intervals:
            samples = slice(vw.first, vw.last + 1, 1)
            weights[widx][samples] = cal
