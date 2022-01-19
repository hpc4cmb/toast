# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

#-------------------------------------------------------------------------------------------------
# JAX

#-------------------------------------------------------------------------------------------------
# NUMPY

def list_dot_numpy(a, b):
    """
    Dot product of lists of arrays.

    Args:
        a(double const *): array of shape (n,4)
        b(double const *): array of shape (n,4)

    Returns:
        dotprot(double *): array of size n
    """
    return np.sum(a * b, axis=1)

def amplitude_numpy(v):
    """
    Norm of quaternion array.

    Args:
        v(double const *): array of shape (n,4)

    Returns:
        norm(double *): array of shape n
    """
    norm2 = list_dot_numpy(v, v)
    norm = np.sqrt(norm2)
    return norm

def normalize_numpy(q_in):
    """
    Normalize quaternion array.

    Args:
        q_in(double const *): array of shape (n,4)
    
    Returns:
        q_out(double *): array of shape (n,4)
    """
    norm = amplitude_numpy(q_in)
    # TODO might need to add axis here
    return q_in / norm

def rotate_many_one_numpy(q, v_in, v_out):
    """
    Rotate an array of vectors by an array of quaternions.

    Args:
        q(double const *): array of quaternions of shape (nq,4)
        v_in(double const *): vector of size 3
    
    Returns:
        v_out(double *): array of vectors of shape (nq,3)
    """
    q_unit = normalize_numpy(q)

    nq = q.shape[0]
    v_out = np.zeros(shape=(nq,3))
    # TODO can the inner loop be turned into linear algebra that would then be vectorized?
    for i in range(nq):
        q_uniti = q_unit[i,:] # quaternion of size 4
        v_outi = v_out[i,:] # v of size 3

        xw =  q_uniti[3] * q_uniti[0]
        yw =  q_uniti[3] * q_uniti[1]
        zw =  q_uniti[3] * q_uniti[2]
        x2 = -q_uniti[0] * q_uniti[0]
        xy =  q_uniti[0] * q_uniti[1]
        xz =  q_uniti[0] * q_uniti[2]
        y2 = -q_uniti[1] * q_uniti[1]
        yz =  q_uniti[1] * q_uniti[2]
        z2 = -q_uniti[2] * q_uniti[2]

        v_outi[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] + (yw + xz) * v_in[2]) + v_in[0]
        v_outi[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] + (yz - xw) * v_in[2]) + v_in[1]
        v_outi[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] + (x2 + y2) * v_in[2]) + v_in[2]

    return v_out

#-------------------------------------------------------------------------------------------------
# C++

"""
// Dot product of lists of arrays.
void toast::qa_list_dot(size_t n, size_t m, size_t d, double const * a,
                        double const * b, double * dotprod) 
{
    for (size_t i = 0; i < n; ++i) 
    {
        dotprod[i] = 0.0;
        size_t off = m * i;
        for (size_t j = 0; j < d; ++j) 
        {
            dotprod[i] += a[off + j] * b[off + j];
        }
    }
}

// Norm of quaternion array
void toast::qa_amplitude(size_t n, size_t m, size_t d, double const * v, double * norm) 
{
    toast::AlignedVector <double> temp(n);
    toast::qa_list_dot(n, m, d, v, v, temp.data());
    toast::vsqrt(n, temp.data(), norm); // sqrt(n, input, output)
}

// Normalize quaternion array.
void toast::qa_normalize(size_t n, size_t m, size_t d,
                         double const * q_in, double * q_out) 
{
    toast::AlignedVector <double> norm(n);
    toast::qa_amplitude(n, m, d, q_in, norm.data());

    for (size_t i = 0; i < n; ++i) 
    {
        size_t off = m * i;
        for (size_t j = 0; j < d; ++j) 
        {
            q_out[off + j] = q_in[off + j] / norm[i];
        }
    }
}

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
