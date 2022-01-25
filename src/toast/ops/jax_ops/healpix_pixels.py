
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import ImplementationType, math_qarray as qarray, math_healpix as healpix
from ..._libtoast import healpix_pixels as healpix_pixels_compiled

# -------------------------------------------------------------------------------------------------
# JAX

def healpix_pixels_single_jax(hpix, nest, pdata, flag):
    """
    Compute the healpix pixel indices for one detector.

    Args:
        hpix (HealpixPixels):  The healpix projection object.
        nest (bool):  If True, then use NESTED ordering, else RING.
        pdata (array, float64):  The flat-packed array of detector quaternions (size 4).
        flag (uint8):  The pointing flag (could also be None).

    Returns:
        pixel (int64):  The detector pixel indice
    """
    # initialize pin
    if (flag is None):
        pin = pdata
    else:
        nullquat = jnp.array([0.0, 0.0, 0.0, 1.0])
        pin = jnp.where(flag == 0, pdata, nullquat)

    # initialize dir
    zaxis = jnp.array([0.0, 0.0, 1.0])
    dir = qarray.rotate_one_one_jax(pin, zaxis)

    if (nest):
        pixel = healpix.vec2nest_jax(hpix, dir)
    else:
        pixel = healpix.vec2ring_jax(hpix, dir)

    if (flag is not None):
        pixel = jnp.where(flag == 0, pixel, -1)

    return pixel

def healpix_pixels_several_jax(hpix, nest, pdata, flags):
    # puts pdata back into shape
    pdata = np.reshape(pdata, newshape=(-1, 4))
    # problem size
    print(f"DEBUG: jit-compiling 'healpix_pixels' nest:{nest} n:{pdata.shape[0]} nside:{hpix.nside}")
    # batch healpix_pixels on the n dimenssion
    healpix_pixels = jax.vmap(healpix_pixels_single_jax, in_axes=(None, None, 0, 0), out_axes=0)
    return healpix_pixels(hpix, nest, pdata, flags)

# jit
healpix_pixels_several_jax = jax.jit(healpix_pixels_several_jax, static_argnames=['nest', 'hpix'])

def healpix_pixels_jax(hpix, nest, pdata, flags, pixels):
    """
    Compute the healpix pixel indices for one detector.

    Args:
        hpix (HealpixPixels_JAX):  The healpix projection object.
        nest (bool):  If True, then use NESTED ordering, else RING.
        pdata (array, float64):  The flat-packed array of detector quaternions (size 4*n).
        flags (array, uint8):  The pointing flags (could also be None).
        pixels (array, int64):  The detector pixel indices to store the result (size n).

    Returns:
        None (results are stored in pixels).
    """
    pixels[:] = healpix_pixels_several_jax(hpix, nest, pdata, flags)

# -------------------------------------------------------------------------------------------------
# NUMPY

def healpix_pixels_numpy(hpix, nest, pdata, flags, pixels):
    """
    Compute the healpix pixel indices for one detector.

    Args:0.000
        hpix (HealpixPixels):  The healpix projection object.
        nest (bool):  If True, then use NESTED ordering, else RING.
        pdata (array, float64):  The flat-packed array of detector quaternions (size 4*n).
        flags (array, uint8):  The pointing flags (size n, could also be None).
        pixels (array, int64):  The detector pixel indices to store the result (size n).

    Returns:
        None (results are stored in pixels).
    """
    # puts pdata back into shape
    n = pixels.size
    pdata = np.reshape(pdata, newshape=(n, 4))

    # constants
    zaxis = np.array([0.0, 0.0, 1.0])
    nullquat = np.array([0.0, 0.0, 0.0, 1.0])

    for i in range(n):
        # initialize pin
        if (flags is None):
            pin = np.copy(pdata[i, :])
        else:
            pin = np.where(flags == 0, pdata[i, :], nullquat)

        # initialize dir
        dir = qarray.rotate_one_one_numpy(pin, zaxis)

        # computes pixel
        if (nest):
            pixel = healpix.vec2nest_numpy(hpix, dir)
        else:
            pixel = healpix.vec2ring_numpy(hpix, dir)

        if (flags is not None):
            pixel = pixel if (flags[i] == 0) else -1

        # saves result
        pixels[i] = pixel

# -------------------------------------------------------------------------------------------------
# C++

"""
void toast::healpix_pixels(toast::HealpixPixels const & hpix, bool nest,
                           size_t n, double const * pdata,
                           uint8_t const * flags, int64_t * pixels) 
{
    double zaxis[3] = {0.0, 0.0, 1.0};
    double nullquat[4] = {0.0, 0.0, 0.0, 1.0};

    toast::AlignedVector <double> dir(3 * n);
    toast::AlignedVector <double> pin(4 * n);

    if (flags == NULL) 
    {
        std::copy(pdata, pdata + (4 * n), pin.begin());
    } 
    else 
    {
        size_t off;
        for (size_t i = 0; i < n; ++i) 
        {
            off = 4 * i;
            if (flags[i] == 0) 
            {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } 
            else 
            {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }
    }

    toast::qa_rotate_many_one(n, pin.data(), zaxis, dir.data());

    if (nest) 
    {
        hpix.vec2nest(n, dir.data(), pixels);
    } 
    else 
    {
        hpix.vec2ring(n, dir.data(), pixels);
    }

    if (flags != NULL) 
    {
        for (size_t i = 0; i < n; ++i) 
        {
            pixels[i] = (flags[i] == 0) ? pixels[i] : -1;
        }
    }
}
"""

# -------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

def select_implementation_and_pick_hpix(f_compiled, f_numpy, f_jax, default_implementationType):
    """
    Returns a transformed function that takes an additional argument to select the implementation
    Picks the proper `hpix` argument as a function of the implementation used
    """
    def f_switch(hpix, hpix_jax, nest, pdata, flags, pixels, implementationType=default_implementationType):
        if implementationType == ImplementationType.COMPILED:
            return f_compiled(hpix, nest, pdata, flags, pixels)
        if implementationType == ImplementationType.NUMPY:
            return f_numpy(hpix, nest, pdata, flags, pixels)
        if implementationType == ImplementationType.JAX:
            return f_jax(hpix_jax, nest, pdata, flags, pixels)
    return f_switch

# lets us play with the various implementations
healpix_pixels = select_implementation_and_pick_hpix(healpix_pixels_compiled,
                                       healpix_pixels_numpy,
                                       healpix_pixels_jax,
                                       default_implementationType=ImplementationType.JAX)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix")'
