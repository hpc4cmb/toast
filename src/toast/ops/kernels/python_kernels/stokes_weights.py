# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .math import qarray

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
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)

    # insures hwp is a non empty array
    hwp = None if (hwp.size == 0) else hwp

    # iterates on detectors and intervals
    n_det = quat_index.size
    for idet in range(n_det):
        w_index = weight_index[idet]
        q_index = quat_index[idet]
        eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet])
        for interval in intervals:
            samples = slice(interval.first, interval.last + 1, 1)
            dir = qarray.rotate(quats[q_index][samples], zaxis)
            orient = qarray.rotate(quats[q_index][samples], xaxis)

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
            weights[w_index][samples, 0] = cal
            weights[w_index][samples, 1] = cal * eta * np.cos(ang)
            weights[w_index][samples, 2] = cal * eta * np.sin(ang)

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


# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix"); toast.tests.run("ops_sim_tod_dipole"); toast.tests.run("ops_sim_tod_atm")'
