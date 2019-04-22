# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ._libtoast import (Environment, Timer, GlobalTimers, Logger)

from ._libtoast import (AlignedI8, AlignedU8, AlignedI16, AlignedU16,
                        AlignedI32, AlignedU32, AlignedI64, AlignedU64,
                        AlignedF32, AlignedF64)

from ._libtoast import (vsin, vcos, vsincos, vatan2, vsqrt, vrsqrt, vexp, vlog)

from ._libtoast import (vfast_sin, vfast_cos, vfast_sincos, vfast_atan2,
                        vfast_sqrt, vfast_rsqrt, vfast_exp, vfast_log,
                        vfast_erfinv)


def object_ndim(x):
    """Get the number of dimension of an object.

    Scalars return 0.  Numpy arrays return their actual ndim value.
    Objects that support the buffer protocol return the ndim value of the
    corresponding memoryview.  Nested lists are traversed to compute the
    effective number of dimensions.

    Args:
        x (object): some stuff.

    Returns:
        (int): the number of dimensions of the stuff.

    """
    try:
        nd = x.ndim
        return nd
    except AttributeError:
        # Not a numpy array...
        try:
            view = memoryview(x)
            nd = view.ndim
            return nd
        except TypeError:
            # Does not support buffer protocol...
            try:
                l = len(x)
                # It's a list!
                nd = 1
                cur = x[0]
                try:
                    l = len(cur)
                    nd += 1
                    cur = cur[0]
                    try:
                        l = len(cur)
                        nd += 1
                        # Using lists of more than 3 dimensions (rather than
                        # a numpy array) is kind of crazy...
                    except TypeError:
                        pass
                except TypeError:
                    pass
                return nd
            except TypeError:
                # Must be a scalar.
                return 0


def ensure_buffer_f64(data):
    """Return a flattened object that supports the buffer protocol.

    Numpy arrays and objects supporting the buffer protocol are flattened and
    copied to contiguous memory if needed.  Scalars and lists are converted
    to numpy arrays.

    Args:
        data (object):  A scalar or iterable.

    Returns:
        (array):  The input unmodified or converted to an array.

    """
    return np.ascontiguousarray(data, dtype=np.float64).flatten()
    # if isinstance(data, np.ndarray):
    #     print("ensure: found numpy array, shape=", data.shape, flush=True)
    #     return np.ascontiguousarray(data.flatten(), dtype=np.float64)
    # try:
    #     view = memoryview(data)
    #     if view.ndim == 0:
    #         # A single element...
    #         print("ensure: converting scalar buffer", flush=True)
    #         return np.array([view], dtype=np.float64)
    #     elif (not view.c_contiguous) or (view.ndim != 1):
    #         print("ensure: found non-contiguous memory view shape=", view.shape, flush=True)
    #         return np.ascontiguousarray(view, dtype=np.float64)
    #     else:
    #         print("ensure: returning original data shape=", view.shape, flush=True)
    #         return data
    # except TypeError:
    #     # Does not support buffer protocol
    #     print("ensure: converting non-buffer object ", data, flush=True)
    #     return np.ascontiguousarray(data, dtype=np.float64)
