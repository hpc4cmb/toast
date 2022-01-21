# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import jax.numpy as jnp

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
    v_out_0 = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] + (yw + xz) * v_in[2]) + v_in[0]
    v_out_1 = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] + (yz - xw) * v_in[2]) + v_in[1]
    v_out_2 = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] + (x2 + y2) * v_in[2]) + v_in[2]

    return jnp.array([v_out_0, v_out_1, v_out_2])

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
    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] + (yw + xz) * v_in[2]) + v_in[0]
    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] + (yz - xw) * v_in[2]) + v_in[1]
    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] + (x2 + y2) * v_in[2]) + v_in[2]

    return v_out

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
