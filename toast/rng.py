# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from ._rng import *


def random(samples, counter=(0,0), sampler="gaussian", key=(0,0)):
    """
    High level interface to internal Random123 library.

    This provides an interface for calling the internal functions to generate
    random values from common distributions.

    Args:
        samples (int): The number of samples to return.
        counter (tuple): Two uint64 values which (along with the key) define
            the starting state of the generator.
        sampler (string): The distribution to sample from.  Allowed values are
            "gaussian", "uniform_01", "uniform_m11", and "uniform_uint64".
        key (tuple): Two uint64 values which (along with the counter) define
            the starting state of the generator.
    Returns:
        array: The random values of appropriate type for the sampler.
    """
    ret = None
    if sampler == "gaussian":
        ret = np.zeros(samples, dtype=np.float64)
        cbrng_normal(samples, 0, counter[0], counter[1], key[0], key[1], ret)
    elif sampler == "uniform_01":
        ret = np.zeros(samples, dtype=np.float64)
        cbrng_uniform_01_f64(samples, 0, counter[0], counter[1], key[0], key[1], ret)
    elif sampler == "uniform_m11":
        ret = np.zeros(samples, dtype=np.float64)
        cbrng_uniform_m11_f64(samples, 0, counter[0], counter[1], key[0], key[1], ret)
    elif sampler == "uniform_uint64":
        ret = np.zeros(samples, dtype=np.uint64)
        cbrng_uniform_uint64(samples, 0, counter[0], counter[1], key[0], key[1], ret)
    else:
        raise ValueError("Undefined sampler. Choose among: gaussian, uniform_01, uniform_m11, uniform_uint64")
    return ret

