# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import jax
import jax.numpy as jnp
import numpy as np

from ..mutableArray import MutableJaxArray

# ----------------------------------------------------------------------------------------
# ROTATE


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
    q_unit = q / jnp.linalg.norm(q)

    # builds the elements that make the matrix representation of the quaternion
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
    v_out_0 = (
        2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] + (yw + xz) * v_in[2]) + v_in[0]
    )
    v_out_1 = (
        2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] + (yz - xw) * v_in[2]) + v_in[1]
    )
    v_out_2 = (
        2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] + (x2 + y2) * v_in[2]) + v_in[2]
    )

    return jnp.array([v_out_0, v_out_1, v_out_2])


def rotate_zaxis(q):
    """
    Rotate the zaxis ([0,0,1]) by a given quaternion.

    Args:
        q(array, double): quaternion of shape (4)

    Returns:
        v_out(array, double): vector of size 3
    """
    # normalize quaternion
    q_unit = q / jnp.linalg.norm(q)

    # performs the matrix multiplication
    x, y, z, w = q_unit
    return 2 * jnp.array([y * w + x * z, y * z - x * w, 0.5 - x * x - y * y])


def rotate_xaxis(q):
    """
    Rotate the xaxis ([1,0,0]) by a given quaternion.

    Args:
        q(array, double): quaternion of shape (4)

    Returns:
        v_out(array, double): vector of size 3
    """
    # normalize quaternion
    q_unit = q / jnp.linalg.norm(q)

    # performs the matrix multiplication
    x, y, z, w = q_unit
    return 2 * jnp.array([0.5 - y * y - z * z, z * w + x * y, x * z - y * w])


# ----------------------------------------------------------------------------------------
# MULT

# def mult_one_one(p, q):
#    """
#    compose two quaternions
#
#    NOTE: this version uses an explicit shuffling of the coordinates
#    """
#    r0 =  p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
#    r1 = -p[0] * q[2] + p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
#    r2 =  p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2]
#    r3 = -p[0] * q[0] - p[1] * q[1] - p[2] * q[2] + p[3] * q[3]
#    return jnp.array([r0,r1,r2,r3])


def mult_one_one(p, q):
    """
    compose two quaternions

    NOTE: this version use a tensor product to keep computation to matrix products
    """
    # reshuffles q into a matrix
    mat = jnp.array(
        [
            [[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],  # row1
            [[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]],  # row2
            [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],  # row3
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # row4
        ]
    )
    qMat = jnp.dot(q, mat)
    # multiplies p with the reshuffled q
    return jnp.dot(p, qMat)


# loops on either or both of the inputs
mult_one_many = jax.vmap(mult_one_one, in_axes=(None, 0), out_axes=0)
mult_many_one = jax.vmap(mult_one_one, in_axes=(0, None), out_axes=0)
mult_many_many = jax.vmap(mult_one_one, in_axes=(0, 0), out_axes=0)


def mult_pure(p_in, q_in):
    # picks the correct impelmentation depending on which input is an array (if any)
    p_is_array = p_in.ndim > 1
    q_is_array = q_in.ndim > 1
    if p_is_array and q_is_array:
        return mult_many_many(p_in, q_in)
    if p_is_array:
        return mult_many_one(p_in, q_in)
    if q_is_array:
        return mult_one_many(p_in, q_in)
    return mult_one_one(p_in, q_in)


# jit compiles the jax function
mult_pure = jax.jit(mult_pure)


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
    # casts the data to arrays and performs the computation
    p_in_input = MutableJaxArray.to_array(p_in)
    q_in_input = MutableJaxArray.to_array(q_in)
    out = mult_pure(p_in_input, q_in_input)
    # converts to a numpy type if the input was a numpy type
    return jax.device_get(out) if isinstance(p_in, np.ndarray) else out


# To test:
# python -c 'import toast.tests; toast.tests.run("qarray");'
