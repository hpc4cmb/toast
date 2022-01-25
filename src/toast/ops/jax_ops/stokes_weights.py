# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import get_compile_time, select_implementation, ImplementationType
from .utils import math_qarray as qarray
from ..._libtoast import stokes_weights as stokes_weights_compiled

#-------------------------------------------------------------------------------------------------
# JAX

def stokes_weights_single_jax(eps, cal, pdata, hwpang, flag):
    """
    Compute the Stokes weights for one detector.

    Args:
        eps (float):  The cross polar response.
        cal (float):  A constant to apply to the pointing weights.
        pdata (array, float64):  The array of detector quaternions (size 4).
        hwpang (float64):  The HWP angle (could be None).
        flag (uint8):  The pointing flag (could be None).

    Returns:
        weights (array, float64):  The detector weights for the specified mode (size 3)
    """
    # initialize pin
    if (flag is None):
        pin = pdata
    else:
        nullquat = jnp.array([0.0, 0.0, 0.0, 1.0])
        pin = jnp.where(flag == 0, pdata, nullquat)
    
    # applies quaternion rotations
    zaxis = jnp.array([0.0, 0.0, 1.0])
    dir = qarray.rotate_one_one_numpy(pin, zaxis)
    xaxis = jnp.array([1.0, 0.0, 0.0])
    orient = qarray.rotate_one_one_numpy(pin, xaxis)

    # computes by and bx
    by = orient[0] * dir[1] - orient[1] * dir[0]
    bx = orient[0] * (-dir[2] * dir[0]) + \
         orient[1] * (-dir[2] * dir[1]) + \
         orient[2] * (dir[0] * dir[0] + dir[1] * dir[1])
    
    # computes detang
    detang = jnp.arctan2(by, bx)
    if (hwpang is not None):
        detang = detang + 2.0 * hwpang
    detang = 2.0 * detang

    # puts values into weights
    eta = (1.0 - eps) / (1.0 + eps)
    weights = jnp.array([cal, jnp.cos(detang) * eta * cal, jnp.sin(detang) * eta * cal])
    return weights

def stokes_weights_IQU_jax(eps, cal, pdata, hwpang, flags):
    """
    Compute the Stokes weights for one detector.

    Args:
        eps (float):  The cross polar response.
        cal (float):  A constant to apply to the pointing weights.
        pdata (array, float64):  The flat-packed array of detector quaternions (size 4*n).
        hwpang (array, float64):  The HWP angles (size n, could also be None).
        flags (array, uint8):  The pointing flags (size n, could also be None).

    Returns:
        weights (array, float64):  The flat packed detector weights for the specified mode (shape nx3).
    """
    # puts pdata back into shape
    pdata = np.reshape(pdata, newshape=(-1, 4))
    # problem size
    print(f"DEBUG: jit-compiling 'stokes_weights' eps:{eps} cal:{cal} n:{pdata.shape[0]}")
    # batch stokes_weights on the n dimenssion
    stokes_weights = jax.vmap(stokes_weights_single_jax, in_axes=(None, None, 0, 0, 0), out_axes=0)
    return stokes_weights(eps, cal, pdata, hwpang, flags)

# TODO jit
# stokes_weights_IQU_jax = jax.jit(stokes_weights_IQU_jax, static_argnames=['eps', 'cal'])

def stokes_weights_jax(eps, cal, mode, pdata, hwpang, flags, weights):
    """
    Compute the Stokes weights for one detector.

    Args:
        eps (float):  The cross polar response.
        cal (float):  A constant to apply to the pointing weights.
        mode (str):  Either "I" or "IQU".
        pdata (array, float64):  The flat-packed array of detector quaternions (size 4*n).
        hwpang (array, float64):  The HWP angles (size n, could also be None).
        flags (array, uint8):  The pointing flags (size n, could also be None).
        weights (array, float64):  The flat packed detector weights for the specified mode (size 3*n).

    Returns:
        None (the result is put in weights).
    """
    if (mode == "I"):
        # sets weights to cal
        weights[:] = cal
    elif (mode == "IQU"):
        # sets weights with (flattened) result
        weights[:] = stokes_weights_IQU_jax(eps, cal, pdata, hwpang, flags).ravel()
    else:
        # thow error due to unknown mode
        msg = f"Unknown stokes weights mode '{mode}'"
        raise RuntimeError(msg)

#-------------------------------------------------------------------------------------------------
# NUMPY

def stokes_weights_IQU_numpy(eps, cal, pdata, hwpang, flags, weights):
    """
    Compute the Stokes weights for one detector and the IQU mode.

    Args:
        eps (float):  The cross polar response.
        cal (float):  A constant to apply to the pointing weights.
        pdata (array, float64):  The flat-packed array of detector quaternions (size 4*n).
        hwpang (array, float64):  The HWP angles (size n, could also be None).
        flags (array, uint8):  The pointing flags (size n, could also be None).
        weights (array, float64):  The flat packed detector weights for the specified mode (size 3*n).

    Returns:
        None (the result is put in weights).
    """
    # puts data back into shape
    n = flags.size
    pdata = np.reshape(pdata, newshape=(n, 4))
    weights = np.reshape(weights, newshape=(n, 3))

    # constants
    xaxis = np.array([1.0, 0.0, 0.0])
    zaxis = np.array([0.0, 0.0, 1.0])
    nullquat = np.array([0.0, 0.0, 0.0, 1.0])
    eta = (1.0 - eps) / (1.0 + eps)

    for i in range(n):
        # initialize pin
        if (flags is None):
            pin = np.copy(pdata[i, :])
        else:
            pin = np.where(flags == 0, pdata[i, :], nullquat)
        
        # applies quaternion rotation
        dir = qarray.rotate_one_one_numpy(pin, zaxis)
        orient = qarray.rotate_one_one_numpy(pin, xaxis)

        # computes by and bx
        by = orient[0] * dir[1] - orient[1] * dir[0]
        bx = orient[0] * (-dir[2] * dir[0]) + \
             orient[1] * (-dir[2] * dir[1]) + \
             orient[2] * (dir[0] * dir[0] + dir[1] * dir[1])
        
        # computes detang
        detang = np.arctan2(by, bx)
        if (hwpang is not None):
            detang += 2.0 * hwpang[i]
        detang *= 2.0

        # puts values into weights
        weights[i,0] = cal
        weights[i,1] = np.cos(detang) * eta * cal
        weights[i,2] = np.sin(detang) * eta * cal

def stokes_weights_numpy(eps, cal, mode, pdata, hwpang, flags, weights):
    """
    Compute the Stokes weights for one detector.

    Args:
        eps (float):  The cross polar response.
        cal (float):  A constant to apply to the pointing weights.
        mode (str):  Either "I" or "IQU".
        pdata (array, float64):  The flat-packed array of detector quaternions (size 4*n).
        hwpang (array, float64):  The HWP angles (size n, could also be None).
        flags (array, uint8):  The pointing flags (size n, could also be None).
        weights (array, float64):  The flat packed detector weights for the specified mode (size 3*n).

    Returns:
        None (the result is put in weights).
    """
    if (mode == "I"):
        # sets weights to cal
        weights[:] = cal
    elif (mode == "IQU"):
        # sets weights according to IQU computation
        stokes_weights_IQU_numpy(eps, cal, pdata, hwpang, flags, weights)
    else:
        # thow error due to unknown mode
        msg = f"Unknown stokes weights mode '{mode}'"
        raise RuntimeError(msg)

#-------------------------------------------------------------------------------------------------
# C++

"""
void toast::stokes_weights(double eps, double cal, std::string const & mode,
                           size_t n, double const * pdata,
                           double const * hwpang,  uint8_t const * flags,
                           double * weights) {
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double nullquat[4] = {0.0, 0.0, 0.0, 1.0};

    double eta = (1.0 - eps) / (1.0 + eps);

    toast::AlignedVector <double> dir(3 * n);
    toast::AlignedVector <double> pin(4 * n);

    if (flags == NULL) {
        std::copy(pdata, pdata + (4 * n), pin.begin());
    } else {
        size_t off;
        for (size_t i = 0; i < n; ++i) {
            off = 4 * i;
            if (flags[i] == 0) {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } else {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }
    }

    toast::qa_rotate_many_one(n, pin.data(), zaxis, dir.data());

    if (mode == "I") {
        for (size_t i = 0; i < n; ++i) {
            weights[i] = cal;
        }
    } else if (mode == "IQU") {
        toast::AlignedVector <double> orient(3 * n);
        toast::AlignedVector <double> buf1(n);
        toast::AlignedVector <double> buf2(n);

        toast::qa_rotate_many_one(n, pin.data(), xaxis, orient.data());

        double * bx = buf1.data();
        double * by = buf2.data();

        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            by[i] = orient[off + 0] * dir[off + 1] - orient[off + 1] *
                    dir[off + 0];
            bx[i] = orient[off + 0] * (-dir[off + 2] * dir[off + 0]) +
                    orient[off + 1] * (-dir[off + 2] * dir[off + 1]) +
                    orient[off + 2] * (dir[off + 0] * dir[off + 0] +
                                       dir[off + 1] * dir[off + 1]);
        }

        toast::AlignedVector <double> detang(n);

        toast::vatan2(n, by, bx, detang.data());

        if (hwpang == NULL) {
            for (size_t i = 0; i < n; ++i) {
                detang[i] *= 2.0;
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                detang[i] += 2.0 * hwpang[i];
                detang[i] *= 2.0;
            }
        }

        double * sinout = buf1.data();
        double * cosout = buf2.data();

        toast::vsincos(n, detang.data(), sinout, cosout);

        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            weights[off + 0] = cal;
            weights[off + 1] = cosout[i] * eta * cal;
            weights[off + 2] = sinout[i] * eta * cal;
        }
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "unknown stokes weights mode \"" << mode << "\"";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }
}
"""

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
stokes_weights = select_implementation(stokes_weights_compiled, 
                                       stokes_weights_numpy, 
                                       stokes_weights_jax, 
                                       default_implementationType=ImplementationType.NUMPY)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
stokes_weights = get_compile_time(stokes_weights)

# To test:
# TODO find test, tod_pointing?
# python -c 'import toast.tests; toast.tests.run("ops_")'
