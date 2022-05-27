# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import jax
import jax.numpy as jnp

from .utils import ImplementationType, select_implementation
from ...qarray import mult as mult_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def mult_one_one_jax(p, q):
    """
    compose two quaternions
    """
    r0 =  p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    r1 = -p[0] * q[2] + p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    r2 =  p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2]
    r3 = -p[0] * q[0] - p[1] * q[1] - p[2] * q[2] + p[3] * q[3]
    return jnp.array([r0,r1,r2,r3])

# loops on either or both of the inputs
mult_one_many_jax = jax.vmap(mult_one_one_jax, in_axes=(None,0), out_axes=0)
mult_many_one_jax = jax.vmap(mult_one_one_jax, in_axes=(0,None), out_axes=0)
mult_many_many_jax = jax.vmap(mult_one_one_jax, in_axes=(0,0), out_axes=0)

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
    out = mult_pure_jax(p_in, q_in)
    # converts back to numpy as some functions want to do later modifications in place
    return np.array(out)

#-------------------------------------------------------------------------------------------------
# NUMPY

def mult_one_one_numpy(p, q):
    """
    compose two quaternions
    """
    r = np.empty(4)
    r[0] =  p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    r[1] = -p[0] * q[2] + p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    r[2] =  p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2]
    r[3] = -p[0] * q[0] - p[1] * q[1] - p[2] * q[2] + p[3] * q[3]
    return r

def mult_one_many_numpy(p, q_arr):
    q_arr = np.reshape(q_arr, newshape=(-1,4))
    out = np.empty_like(q_arr)
    nb_quaternions = q_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one_numpy(p, q_arr[i,:])
    return out

def mult_many_one_numpy(p_arr, q):
    p_arr = np.reshape(p_arr, newshape=(-1,4))
    out = np.empty_like(p_arr)
    nb_quaternions = p_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one_numpy(p_arr[i,:], q)
    return out

def mult_many_many_numpy(p_arr, q_arr):
    p_arr = np.reshape(p_arr, newshape=(-1,4))
    q_arr = np.reshape(q_arr, newshape=(-1,4))
    out = np.empty_like(q_arr)
    nb_quaternions = q_arr.shape[0]
    for i in range(nb_quaternions):
        out[i:] = mult_one_one_numpy(p_arr[i,:], q_arr[i,:])
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
    #print(f"DEBUG: running 'mult_numpy' for p:{p_in.size} and q:{q_in.size}")
    p_is_array = (p_in.size > 4)
    q_is_array = (q_in.size > 4)
    if p_is_array and q_is_array:
        return mult_many_many_numpy(p_in, q_in)
    if p_is_array:
        return mult_many_one_numpy(p_in, q_in)
    if q_is_array:
        return mult_one_many_numpy(p_in, q_in)
    return mult_one_one_numpy(p_in, q_in)
    
#-------------------------------------------------------------------------------------------------
# C++

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

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
mult = select_implementation(mult_compiled, 
                             mult_numpy, 
                             mult_jax, 
                             default_implementationType=ImplementationType.JAX)

# To test:
# python -c 'import toast.tests; toast.tests.run("qarray");'

# TODO to bench:
# use scanmap_config.toml
# run_mapmaker|MapMaker._exec|BinMap._exec|Pipeline._exec|PixelsHealpix._exec|PointingDetectorSimple._exec
# line 48 in timing.csv