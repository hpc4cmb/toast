# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ._libtoast import FFTDirection, FFTPlanReal1D, FFTPlanReal1DStore, FFTPlanType
from .utils import object_ndim


def r1d_forward(indata):
    """High level forward FFT interface to internal library.

    This function uses the internal store of FFT plans to do a forward
    1D real FFT.  Data is copied into the internal aligned memory
    buffer inside the plan and the result is copied out and returned.

    If a 2D array is passed, the first dimension is assumed to be the number of
    FFTs to batch at once.

    Args:
        indata (array): The input data array.

    Returns:
        (array): The output Fourier-domain data in FFTW half-complex format.

    """
    ndim = object_ndim(indata)
    count = None
    length = None
    if ndim == 1:
        # Just a single FFT
        count = 1
        length = len(indata)
    else:
        # We must have a batch of FFTs
        count = indata.shape[0]
        length = indata.shape[1]

    store = FFTPlanReal1DStore.get()
    plan = store.forward(length, count)

    if count == 1:
        plan.tdata(0)[:] = indata
    else:
        for indx in range(count):
            plan.tdata(indx)[:] = indata[indx]
    plan.exec()
    if count == 1:
        return np.array(plan.fdata(0))
    else:
        ret = np.zeros_like(indata)
        for indx in range(count):
            ret[indx, :] = plan.fdata(indx)
        return ret


def r1d_backward(indata):
    """High level backward FFT interface to internal library.

    This function uses the internal store of FFT plans to do a backward
    1D real FFT.  Data is copied into the internal aligned memory
    buffer inside the plan and the result is copied out and returned.

    If a 2D array is passed, the first dimension is assumed to be the number of
    FFTs to batch at once.

    Args:
        indata (array): The input Fourier-domain data array in FFTW
            half-complex format.

    Returns:
        (array): The output time domain data.

    """
    ndim = object_ndim(indata)
    count = None
    length = None
    if ndim == 1:
        # Just a single FFT
        count = 1
        length = len(indata)
    else:
        # We must have a batch of FFTs
        count = indata.shape[0]
        length = indata.shape[1]

    store = FFTPlanReal1DStore.get()
    plan = store.backward(length, count)

    if count == 1:
        plan.fdata(0)[:] = indata
    else:
        for indx in range(count):
            plan.fdata(indx)[:] = indata[indx]
    plan.exec()
    if count == 1:
        return np.array(plan.tdata(0))
    else:
        ret = np.zeros_like(indata)
        for indx in range(count):
            ret[indx, :] = plan.tdata(indx)
        return ret
