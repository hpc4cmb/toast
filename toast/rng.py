# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from ._rng import *


# define a more pythonic class here to represent the random
# number generator, which uses stuff from _rng.

class CBRNG:
    """
    Counter-based random number generator class


    """

    def random(array, counter, sampler="gaussian" , key=[0xcafebead,0xbaadfeed]):
        if sampler == "gaussian":
            cbrng_normal(np.shape(array)[0], 0, counter[0], counter[1], key[0], key[1], array)
        elif sampler == "uniform_01":
            cbrng_uniform_01_f64(np.shape(array)[0], 0, counter[0], counter[1], key[0], key[1], array)
        elif sampler == "uniform_m11":
            cbrng_uniform_m11_f64(np.shape(array)[0], 0, counter[0], counter[1], key[0], key[1], array)
        elif sampler == "uniform_uint64":
            cbrng_uniform_uint64(np.shape(array)[0], 0, counter[0], counter[1], key[0], key[1], array)
        else:
            print("Undefined sampler. Choose among: gaussian, uniform_01, uniform_m11, uniform_uint64")
        return
