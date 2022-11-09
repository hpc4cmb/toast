# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .math import qarray


def stokes_weights_IQU_inner(eps, cal, pin, hwpang, weights):
    """
    Compute the Stokes weights for one detector and the IQU mode.

    Args:
        eps (float):  The cross polar response.
        cal (float):  A constant to apply to the pointing weights.
        pin (array, float64):  Ddetector quaternions (size 4).
        hwpang (float):  The HWP angle.
        weights (array, float64):  Detector weights for the specified mode (size 3).

    Returns:
        None (the result is put in weights).
    """
    # constants
    xaxis = np.array([1.0, 0.0, 0.0])
    zaxis = np.array([0.0, 0.0, 1.0])
    eta = (1.0 - eps) / (1.0 + eps)

    # applies quaternion rotation
    dir = qarray.rotate_one_one(pin, zaxis)
    orient = qarray.rotate_one_one(pin, xaxis)

    # computes by and bx
    by = orient[0] * dir[1] - orient[1] * dir[0]
    bx = (
        orient[0] * (-dir[2] * dir[0])
        + orient[1] * (-dir[2] * dir[1])
        + orient[2] * (dir[0] * dir[0] + dir[1] * dir[1])
    )

    # computes detang
    detang = np.arctan2(by, bx)
    detang += 2.0 * hwpang
    detang *= 2.0

    # puts values into weights
    weights[0] = cal
    weights[1] = np.cos(detang) * eta * cal
    weights[2] = np.sin(detang) * eta * cal


def stokes_weights_IQU(
    quat_index, quats, weight_index, weights, hwp, intervals, epsilon, cal, use_accel
):
    """
    Compute the Stokes weights for the "IQU" mode.

    Args:
        quat_index (array, int): size n_det
        quats (array, double): size ???*n_samp*4
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, float64): The flat packed detectors weights for the specified mode (size ???*n_samp*3)
        hwp (optional array, float64):  The HWP angles (size n_samp, could be None).
        intervals (array, Interval): The intervals to modify (size n_view)
        epsilon (array, float):  The cross polar response (size n_det).
        cal (float):  A constant to apply to the pointing weights.
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in weights).
    """
    # insures hwp is a non empty array
    if (hwp is None) or (hwp.size == 0):
        n_samp = quats.shape[1]
        hwp = np.zeros(n_samp)

    # iterates on detectors and intervals
    n_det = quat_index.size
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval.first
            interval_end = interval.last + 1
            for isamp in range(interval_start, interval_end):
                w_index = weight_index[idet]
                q_index = quat_index[idet]
                stokes_weights_IQU_inner(
                    epsilon[idet],
                    cal,
                    quats[q_index, isamp, :],
                    hwp[isamp],
                    weights[w_index, isamp, :],
                )


def stokes_weights_I(weight_index, weights, intervals, cal, use_accel):
    """
    Compute the Stokes weights for the "I" mode.

    Args:
        weight_index (array, int): The indexes of the weights (size n_det)
        weights (array, float64): The flat packed detectors weights for the specified mode (size n_det*n_samp)
        intervals (array, Interval): The intervals to modify (size n_view)
        cal (float):  A constant to apply to the pointing weights.
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in weights).
    """
    for interval in intervals:
        interval_start = interval.first
        interval_end = interval.last + 1
        weights[weight_index, interval_start:interval_end] = cal


# hwp_data = None
# if self.hwp_angle is not None:
#   hwp_data = ob.shared[self.hwp_angle].data
def _py_stokes_weights(
    self,
    quat_indx,
    quat_data,
    weight_indx,
    weight_data,
    intr_data,
    cal,
    det_epsilon,
    hwp_data,
):
    """Internal python implementation for comparison tests."""
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)
    if self.mode == "IQU":
        for idet in range(len(quat_indx)):
            qidx = quat_indx[idet]
            widx = weight_indx[idet]
            eta = (1.0 - det_epsilon[idet]) / (1.0 + det_epsilon[idet])
            for vw in intr_data:
                samples = slice(vw.first, vw.last + 1, 1)
                dir = qarray.rotate_one_one(quat_data[qidx][samples], zaxis)
                orient = qarray.rotate_one_one(quat_data[qidx][samples], xaxis)

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
                if hwp_data is not None:
                    ang += 2.0 * hwp_data[samples]
                ang *= 2.0
                weight_data[widx][samples, 0] = cal
                weight_data[widx][samples, 1] = cal * eta * np.cos(ang)
                weight_data[widx][samples, 2] = cal * eta * np.sin(ang)
    else:
        for idet in range(len(quat_indx)):
            widx = weight_indx[idet]
            for vw in intr_data:
                samples = slice(vw.first, vw.last + 1, 1)
                weight_data[widx][samples] = cal
