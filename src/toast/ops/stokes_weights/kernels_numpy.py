# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ... import qarray as qa

from ...accelerator import kernel, ImplementationType


@kernel(impl=ImplementationType.NUMPY, name="stokes_weights_IQU")
def stokes_weights_IQU_numpy(
    quat_index, quats, weight_index, weights, hwp, intervals, epsilon, cal, use_accel
):
    if hwp is not None and len(hwp) == 0:
        hwp = None
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)
    for idet in range(len(quat_index)):
        qidx = quat_index[idet]
        widx = weight_index[idet]
        eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet])
        for vw in intervals:
            samples = slice(vw.first, vw.last + 1, 1)
            dir = qa.rotate(quats[qidx][samples], zaxis)
            orient = qa.rotate(quats[qidx][samples], xaxis)

            # The vector orthogonal to the line of sight that is parallel
            # to the local meridian.
            dir_ang = np.arctan2(dir[:, 1], dir[:, 0])
            dir_r = np.sqrt(1.0 - dir[:, 2] * dir[:, 2])
            m_z = dir_r
            m_x = -dir[:, 2] * np.cos(dir_ang)
            m_y = -dir[:, 2] * np.sin(dir_ang)

            # Compute the rotation angle from the meridian vector to the
            # orientation vector.  The direction vector is normal to the plane
            # containing these two vectors, so the rotation angle is:
            #
            # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
            # angle = atan2(
            #     d_x (m_y o_z - m_z o_y)
            #       - d_y (m_x o_z - m_z o_x)
            #       + d_z (m_x o_y - m_y o_x),
            #     m_x o_x + m_y o_y + m_z o_z
            # )
            #
            ay = (
                dir[:, 0] * (m_y * orient[:, 2] - m_z * orient[:, 1])
                - dir[:, 1] * (m_x * orient[:, 2] - m_z * orient[:, 0])
                + dir[:, 2] * (m_x * orient[:, 1] - m_y * orient[:, 0])
            )
            ax = m_x * orient[:, 0] + m_y * orient[:, 1] + m_z * orient[:, 2]
            ang = np.arctan2(ay, ax)
            if hwp is not None:
                ang += 2.0 * hwp[samples]
            ang *= 2.0
            weights[widx][samples, 0] = cal
            weights[widx][samples, 1] = cal * eta * np.cos(ang)
            weights[widx][samples, 2] = cal * eta * np.sin(ang)


@kernel(impl=ImplementationType.NUMPY, name="stokes_weights_I")
def stokes_weights_I_numpy(weight_index, weights, intervals, cal, use_accel):
    for idet in range(len(weight_index)):
        widx = weight_index[idet]
        for vw in intervals:
            samples = slice(vw.first, vw.last + 1, 1)
            weights[widx][samples] = cal

