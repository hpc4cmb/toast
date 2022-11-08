# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


def rotate_one_one(q, v_in):
    """
    Rotate a vector by a quaternion.

    Args:
        q(array, double): quaternion of shape (4)
        v_in(array, double): vector of size 3

    Returns:
        v_out(array, double): vector of size 3
    """
    # normalize quaternion
    q_unit = q / np.linalg.norm(q)

    # builds the elments that make the matrix representation of the quaternion
    xw = q_unit[3] * q_unit[0]
    yw = q_unit[3] * q_unit[1]
    zw = q_unit[3] * q_unit[2]
    x2 = -q_unit[0] * q_unit[0]
    xy = q_unit[0] * q_unit[1]
    xz = q_unit[0] * q_unit[2]
    y2 = -q_unit[1] * q_unit[1]
    yz = q_unit[1] * q_unit[2]
    z2 = -q_unit[2] * q_unit[2]

    # matrix product
    v_out = np.empty(3)
    v_out[0] = (
        2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] + (yw + xz) * v_in[2]) + v_in[0]
    )
    v_out[1] = (
        2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] + (yz - xw) * v_in[2]) + v_in[1]
    )
    v_out[2] = (
        2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] + (x2 + y2) * v_in[2]) + v_in[2]
    )

    return v_out


# -----


def mult_one_one(p, q):
    """
    compose two quaternions
    """
    r = np.empty(4)
    r[0] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    r[1] = -p[0] * q[2] + p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    r[2] = p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2]
    r[3] = -p[0] * q[0] - p[1] * q[1] - p[2] * q[2] + p[3] * q[3]
    return r


def mult_one_many(p, q_arr):
    q_arr = np.reshape(q_arr, newshape=(-1, 4))
    out = np.empty_like(q_arr)
    nb_quaternions = q_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one(p, q_arr[i, :])
    return out


def mult_many_one(p_arr, q):
    p_arr = np.reshape(p_arr, newshape=(-1, 4))
    out = np.empty_like(p_arr)
    nb_quaternions = p_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one(p_arr[i, :], q)
    return out


def mult_many_many(p_arr, q_arr):
    p_arr = np.reshape(p_arr, newshape=(-1, 4))
    q_arr = np.reshape(q_arr, newshape=(-1, 4))
    out = np.empty_like(q_arr)
    nb_quaternions = q_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one(p_arr[i, :], q_arr[i, :])
    return out


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
    # picks the correct implementation depending on which input is an array (if any)
    # print(f"DEBUG: running 'mult_numpy' for p:{p_in.size} and q:{q_in.size}")
    p_is_array = p_in.size > 4
    q_is_array = q_in.size > 4
    if p_is_array and q_is_array:
        return mult_many_many(p_in, q_in)
    if p_is_array:
        return mult_many_one(p_in, q_in)
    if q_is_array:
        return mult_one_many(p_in, q_in)
    return mult_one_one(p_in, q_in)
