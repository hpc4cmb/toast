# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
import jax
import jax.numpy as jnp

from ....jax.mutableArray import MutableJaxArray
from ....jax.implementation_selection import select_implementation

from ....qarray import mult as mult_compiled

# -------------------------------------------------------------------------------------------------
# JAX


def rotate_one_one_jax(q, v_in):
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


# -----


# def mult_one_one_jax(p, q):
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


def mult_one_one_jax(p, q):
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
mult_one_many_jax = jax.vmap(mult_one_one_jax, in_axes=(None, 0), out_axes=0)
mult_many_one_jax = jax.vmap(mult_one_one_jax, in_axes=(0, None), out_axes=0)
mult_many_many_jax = jax.vmap(mult_one_one_jax, in_axes=(0, 0), out_axes=0)


def mult_pure_jax(p_in, q_in):
    # picks the correct impelmentation depending on which input is an array (if any)
    print(f"DEBUG: jit-compiling 'qarray.mult' for p:{p_in.shape} and q:{q_in.shape}")
    p_is_array = p_in.ndim > 1
    q_is_array = q_in.ndim > 1
    if p_is_array and q_is_array:
        return mult_many_many_jax(p_in, q_in)
    if p_is_array:
        return mult_many_one_jax(p_in, q_in)
    if q_is_array:
        return mult_one_many_jax(p_in, q_in)
    return mult_one_one_jax(p_in, q_in)


# jit compiles the jax function
mult_pure_jax = jax.jit(mult_pure_jax)


def mult_jax(p_in, q_in):
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
    out = mult_pure_jax(p_in_input, q_in_input)
    # converts to a numpy type if the input was a numpy type
    return jax.device_get(out) if isinstance(p_in, np.ndarray) else out


# -------------------------------------------------------------------------------------------------
# NUMPY


def rotate_one_one_numpy(q, v_in):
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


def mult_one_one_numpy(p, q):
    """
    compose two quaternions
    """
    r = np.empty(4)
    r[0] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    r[1] = -p[0] * q[2] + p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    r[2] = p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2]
    r[3] = -p[0] * q[0] - p[1] * q[1] - p[2] * q[2] + p[3] * q[3]
    return r


def mult_one_many_numpy(p, q_arr):
    q_arr = np.reshape(q_arr, newshape=(-1, 4))
    out = np.empty_like(q_arr)
    nb_quaternions = q_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one_numpy(p, q_arr[i, :])
    return out


def mult_many_one_numpy(p_arr, q):
    p_arr = np.reshape(p_arr, newshape=(-1, 4))
    out = np.empty_like(p_arr)
    nb_quaternions = p_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one_numpy(p_arr[i, :], q)
    return out


def mult_many_many_numpy(p_arr, q_arr):
    p_arr = np.reshape(p_arr, newshape=(-1, 4))
    q_arr = np.reshape(q_arr, newshape=(-1, 4))
    out = np.empty_like(q_arr)
    nb_quaternions = q_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one_numpy(p_arr[i, :], q_arr[i, :])
    return out


def mult_numpy(p_in, q_in):
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
        return mult_many_many_numpy(p_in, q_in)
    if p_is_array:
        return mult_many_one_numpy(p_in, q_in)
    if q_is_array:
        return mult_one_many_numpy(p_in, q_in)
    return mult_one_one_numpy(p_in, q_in)


# -------------------------------------------------------------------------------------------------
# C++

"""
// Rotate an array of vectors by an array of quaternions.
void toast::qa_rotate_many_one(size_t nq, double const * q,
                               double const * v_in, double * v_out) 
{
    toast::AlignedVector <double> q_unit(4 * nq);
    toast::qa_normalize(nq, 4, 4, q, q_unit.data());

    for (size_t i = 0; i < nq; ++i) 
    {
        size_t vfout = 3 * i;
        size_t qf = 4 * i;
        double xw =  q_unit[qf + 3] * q_unit[qf + 0];
        double yw =  q_unit[qf + 3] * q_unit[qf + 1];
        double zw =  q_unit[qf + 3] * q_unit[qf + 2];
        double x2 = -q_unit[qf + 0] * q_unit[qf + 0];
        double xy =  q_unit[qf + 0] * q_unit[qf + 1];
        double xz =  q_unit[qf + 0] * q_unit[qf + 2];
        double y2 = -q_unit[qf + 1] * q_unit[qf + 1];
        double yz =  q_unit[qf + 1] * q_unit[qf + 2];
        double z2 = -q_unit[qf + 2] * q_unit[qf + 2];

        v_out[vfout + 0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] + (yw + xz) * v_in[2]) + v_in[0];
        v_out[vfout + 1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] + (yz - xw) * v_in[2]) + v_in[1];
        v_out[vfout + 2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] + (x2 + y2) * v_in[2]) + v_in[2];
    }
}
"""

"""
void toast::qa_mult_one_one(double const * p, double const * q,
                            double * r) 
{
    r[0] =  p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
    r[1] = -p[0] * q[2] + p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
    r[2] =  p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2];
    r[3] = -p[0] * q[0] - p[1] * q[1] - p[2] * q[2] + p[3] * q[3];

    return;
}
"""

# -------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
mult = select_implementation(mult_compiled, mult_numpy, mult_jax)

# To test:
# python -c 'import toast.tests; toast.tests.run("qarray");'

# TODO to bench:
# use scanmap_config.toml
# run_mapmaker|MapMaker._exec|BinMap._exec|Pipeline._exec|PixelsHealpix._exec|PointingDetectorSimple._exec
# line 48 in timing.csv
