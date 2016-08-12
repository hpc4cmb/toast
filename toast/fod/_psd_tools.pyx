# Copyright (c) 2015-2016 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed 
# by a BSD-style license that can be found in the LICENSE file.

cimport cython
from cython.parallel import parallel, prange

import numpy as np
cimport numpy as np

np.import_array()

f64 = np.float64
i64 = np.int64
i32 = np.int32

ctypedef np.float64_t f64_t
ctypedef np.uint64_t u64_t
ctypedef np.int64_t i64_t
ctypedef np.int32_t i32_t
ctypedef np.uint8_t u8_t

@cython.boundscheck(False)
@cython.wraparound(False)    
def autosums(np.ndarray[f64_t, ndim=1] x, np.ndarray[u8_t, ndim=1, cast=True] good, int lagmax):

    cdef long n = x.size

    x = np.ascontiguousarray(x)

    cdef long i

    for i in range(n):
        if good[i] != 0:
            good[i] = 1
        else:
            x[i] = 0.

    cdef np.ndarray[f64_t, ndim=1, mode='c'] sums = np.zeros(lagmax, dtype=f64)
    cdef np.ndarray[i64_t, ndim=1, mode='c'] hits = np.zeros(lagmax, dtype=i64)

    cdef long lag
    cdef long j
    cdef double lagsum
    cdef long hitsum

    for lag in prange(lagmax, schedule='dynamic', nogil=True):
        j = lag
        lagsum = 0.
        hitsum = 0
        for i in range(n-lag):
            lagsum = lagsum + x[i] * x[j]
            hitsum = hitsum + good[i] * good[j]
            j = j + 1
        sums[lag] = lagsum
        hits[lag] = hitsum

    return sums, hits
