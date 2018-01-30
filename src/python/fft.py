# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
import timemory

from .ctoast import ( fft_r1d_store_get, fft_r1d_store_forward,
    fft_r1d_store_backward, fft_r1d_exec, fft_r1d_tdata_set, fft_r1d_tdata_get,
    fft_r1d_fdata_set, fft_r1d_fdata_get )


def r1d_forward(indata):
    """
    High level forward FFT interface to internal library.

    This function uses the internal store of FFT plans to do a forward
    1D real FFT.  Data is copied into the internal aligned memory 
    buffer inside the plan and the result is copied out and returned.

    Args:
        indata (array): The input data array.

    Returns:
        array: The output Fourier-domain data in FFTW half-complex format.
    """
    autotimer = timemory.auto_timer()
    cnt = 1
    len = indata.shape[0]
    store = fft_r1d_store_get()
    plan = fft_r1d_store_forward(store, len, cnt)
    fft_r1d_tdata_set(plan, [ indata ])
    fft_r1d_exec(plan)
    return fft_r1d_fdata_get(plan)[0]


def r1d_backward(indata):
    """
    High level backward FFT interface to internal library.

    This function uses the internal store of FFT plans to do a backward
    1D real FFT.  Data is copied into the internal aligned memory 
    buffer inside the plan and the result is copied out and returned.

    Args:
        indata (array): The input Fourier-domain data array in FFTW 
            half-complex format.

    Returns:
        array: The output data.
    """
    autotimer = timemory.auto_timer()
    cnt = 1
    len = indata.shape[0]
    store = fft_r1d_store_get()
    plan = fft_r1d_store_backward(store, len, cnt)
    fft_r1d_fdata_set(plan, [ indata ])
    fft_r1d_exec(plan)
    return fft_r1d_tdata_get(plan)[0]


