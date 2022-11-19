# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

def rotate(q_in, v_in):
    """
    Rotate a batch of vectors by a batch of quaternion.

    Args:
        q_in(array, double): a batch of quaternions of shape (nb_quats,4)
        v_in(array, double): a batch of vectors of size (nb_vec,3)

    Returns:
        v_out(array, double): batch of vectors of size (nb_quats,3)
    """
    # insures the inputs have the proper shapes
    q_arr = np.reshape(q_in, newshape=(-1,4))
    v_arr = np.reshape(v_in, newshape=(-1,3))
    batch_size = max(q_arr.shape[0], v_arr.shape[0])

    # normalize quaternion
    q_unit = q_arr / np.linalg.norm(q_arr, axis=1)[:,np.newaxis]

    # builds the elments that make the matrix representation of the quaternion
    xw = q_unit[:,3] * q_unit[:,0]
    yw = q_unit[:,3] * q_unit[:,1]
    zw = q_unit[:,3] * q_unit[:,2]
    x2 = -q_unit[:,0] * q_unit[:,0]
    xy = q_unit[:,0] * q_unit[:,1]
    xz = q_unit[:,0] * q_unit[:,2]
    y2 = -q_unit[:,1] * q_unit[:,1]
    yz = q_unit[:,1] * q_unit[:,2]
    z2 = -q_unit[:,2] * q_unit[:,2]

    # matrix product
    v_out = np.empty(shape=(batch_size,3))
    v_out[:,0] = (
        2 * ((y2 + z2) * v_arr[:,0] + (xy - zw) * v_arr[:,1] + (yw + xz) * v_arr[:,2]) + v_arr[:,0]
    )
    v_out[:,1] = (
        2 * ((zw + xy) * v_arr[:,0] + (x2 + z2) * v_arr[:,1] + (yz - xw) * v_arr[:,2]) + v_arr[:,1]
    )
    v_out[:,2] = (
        2 * ((xz - yw) * v_arr[:,0] + (xw + yz) * v_arr[:,1] + (x2 + y2) * v_arr[:,2]) + v_arr[:,2]
    )

    # returns properly shaped output
    if (q_in.ndim < 2) and (v_in.ndim < 2):
        v_out = np.reshape(v_out, newshape=(3,))
    return v_out

def mult(p_in, q_in):
    """
    Multiply quaternion arrays.

    The number of quaternions in both input arrays should either be
    equal or there should be one quaternion in one of the arrays (which
    is then multiplied by all quaternions in the other array).

    Args:
        p_in (array_like):  flattened 1D array of float64 values.
        q_in (array_like):  flattened 1D array of float64 values.

    Returns:
        out (array_like):  flattened 1D array of float64 values.
    """
    p_arr = np.reshape(p_in, newshape=(-1, 4))
    q_arr = np.reshape(q_in, newshape=(-1, 4))
    nb_quaternions = max(p_arr.shape[0], q_arr.shape[0])
    out = np.empty(shape=(nb_quaternions,4))
    out[:,0] = p_arr[:,0] * q_arr[:,3] + p_arr[:,1] * q_arr[:,2] - p_arr[:,2] * q_arr[:,1] + p_arr[:,3] * q_arr[:,0]
    out[:,1] = -p_arr[:,0] * q_arr[:,2] + p_arr[:,1] * q_arr[:,3] + p_arr[:,2] * q_arr[:,0] + p_arr[:,3] * q_arr[:,1]
    out[:,2] = p_arr[:,0] * q_arr[:,1] - p_arr[:,1] * q_arr[:,0] + p_arr[:,2] * q_arr[:,3] + p_arr[:,3] * q_arr[:,2]
    out[:,3] = -p_arr[:,0] * q_arr[:,0] - p_arr[:,1] * q_arr[:,1] - p_arr[:,2] * q_arr[:,2] + p_arr[:,3] * q_arr[:,3]
    if (p_in.ndim < 2) and (q_in.ndim < 2):
        out = np.reshape(out, newshape=(4,))
    return out
