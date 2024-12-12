# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import windows

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


def convolve(
    data,
    rate,
    kernel_freq=None,
    kernels=None,
    kernel_func=None,
    deconvolve=False,
    algorithm="numpy",
    debug=None,
):
    """Convolve timestream data with Fourier domain kernels.

    If data is a 2D array, the first dimension is the number of timestreams.
    The FFT length is chosen to be the smallest radix-2 value sufficient to store
    twice the data length.  The input data is centered in the time-domain buffer
    and is reflected about the endpoints to avoid discontinuities.  Any additional
    samples on either end are zero padded.  The reflected data is further apodized
    smoothly to zero with a Gaussian window.

    If `kernels` is a 2D array, there should be one per timestream, and all
    kernels should be specified on the same frequency grid given by `kernel_freq.
    If `kernels` is 1D, the same kernel will be applied to all timestreams.  The
    kernels will be interpolated in frequency space to match the FFT resolution.
    Alternatively, if `kernel_func` is specified, it will be called with the
    detector index and the Fourier domain frequencies.  The function should return
    the complex kernel values at these frequencies.

    The input data is modified in-place.

    Args:
        data (iterable):  Sequence of timestreams.
        rate (float):  Sample rate in Hz.
        kernel_freq (array):  The common frequency values for all input kernels.
        kernels (array):  Array of Fourier domain kernels.
        deconvolve (bool):  If True, divide by the kernel instead of multiplying.
        use_numpy (bool):  If True, use numpy.fft instead of internal tools.

    Returns:
        None

    """
    if kernel_func is not None:
        if kernel_freq is not None or kernels is not None:
            msg = "Cannot specify both the kernel function and explicit kernel values"
            raise RuntimeError(msg)
    else:
        if kernel_freq is None or kernels is None:
            msg = "Must specify either the kernel function or explicit kernel values"
            raise RuntimeError(msg)
        if len(kernel_freq.shape) != 1:
            msg = "kernel_freq should be the common frequencies for all kernels"
            raise RuntimeError(msg)
        if len(kernels.shape) == 1:
            # Common kernel
            if kernels.shape[0] != kernel_freq.shape[0]:
                msg = "common kernel should have same length as frequencies"
                raise RuntimeError(msg)
        elif len(kernels.shape) == 2:
            if kernels.shape[1] != kernel_freq.shape[0]:
                msg = "kernels should have same length as frequencies"
                raise RuntimeError(msg)
        else:
            msg = "kernels should be a 1D or 2D array"
            raise RuntimeError(msg)

    n_tod = len(data)
    n_samp = len(data[0])
    for itod in range(1, n_tod):
        if len(data[itod]) != n_samp:
            msg = "All detector arrays should be the same length"
            raise RuntimeError(msg)

    # Find the FFT size and frequencies
    order = int(np.ceil(np.log(n_samp) / np.log(2)))
    n_fft = 2 ** (order + 1)
    n_psd = n_fft // 2 + 1
    freq = np.fft.rfftfreq(n_fft, d=1.0 / rate)
    n_buffer = (n_fft - n_samp) // 2
    n_reflect = min(n_buffer, n_samp)

    # Apodization of reflected extension
    apodize = windows.general_gaussian(
        n_reflect * 2,
        3.0,
        (n_reflect // 2),
        sym=True,
    )[:n_reflect]

    def _interpolate_kernel(kern):
        """Helper function to interpolate a single kernel.

        Only used when interpolating an explicit kernel.  Not used with a kernel.
        function.
        """
        # If we are deconvolving, ensure that zero-values are set to something
        # reasonable and small.
        if kernel_freq is None:
            raise RuntimeError("Should never get here- input kernel_freq is None")
        kern_mag = np.absolute(kern)
        kern_ang = np.angle(kern)
        if deconvolve:
            kern_extent = np.max(kern_mag)
            kern_limit = 1.0e-5 * kern_extent
            safe_mag = np.array(kern_mag)
            safe_mag[kern_mag < kern_limit] = kern_limit
        else:
            safe_mag = kern_mag
        mag_interp = PchipInterpolator(kernel_freq, safe_mag, extrapolate=True)
        ang_interp = PchipInterpolator(kernel_freq, kern_ang, extrapolate=True)
        mag_out = mag_interp(freq)
        ang_out = ang_interp(freq)
        out = mag_out * np.exp(1j * ang_out)
        return out

    def _set_input(td, hndl):
        """Helper function to populate the time domain buffer."""
        td[:] = 0
        td[n_buffer - n_reflect : n_buffer] = hndl[n_reflect - 1 :: -1]
        td[n_buffer : n_buffer + n_samp] = hndl[:]
        td[n_buffer + n_samp : n_buffer + n_samp + n_reflect] = hndl[
            -1 : -(n_reflect + 1) : -1
        ]
        td[n_buffer - n_reflect : n_buffer] *= apodize
        td[n_buffer + n_samp + n_reflect - 1 : n_buffer + n_samp - 1 : -1] *= apodize

    def _debug_plot_fourier(fdata, plotroot):
        """Helper function to plot fourier domain data."""
        import matplotlib.pyplot as plt

        for frange in [(0, len(fdata)), (0, 32), (-32, len(fdata))]:
            plotfile = f"{plotroot}_{frange[0]}-{frange[1]}.pdf"
            pslc = slice(frange[0], frange[1], 1)
            fig = plt.figure(figsize=(12, 16), dpi=72)
            ax = fig.add_subplot(2, 1, 1, aspect="auto")
            ax.plot(freq[pslc], np.real(fdata[pslc]))
            ax.set_title("Fourier Domain Real Part")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Amplitude")
            ax = fig.add_subplot(2, 1, 2, aspect="auto")
            ax.plot(freq[pslc], np.imag(fdata[pslc]))
            ax.set_title("Fourier Domain Imaginary Part")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Amplitude")
            plt.savefig(plotfile)
            plt.close()

    def _debug_plot_tod(tdata, plotroot):
        """Helper function to plot time domain data."""
        import matplotlib.pyplot as plt

        for frange in [(0, len(tdata)), (0, 500), (-500, len(tdata))]:
            plotfile = f"{plotroot}_{frange[0]}-{frange[1]}.pdf"
            pslc = slice(frange[0], frange[1], 1)
            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(np.arange(n_fft)[pslc], tdata[pslc])
            ax.set_title("Time Domain Data")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            plt.savefig(plotfile)
            plt.close()

    common_kernel = None
    if kernel_func is None and len(kernels.shape) == 1:
        # We have a common kernel, interpolate it now
        common_kernel = _interpolate_kernel(kernels)
        if debug is not None:
            _debug_plot_fourier(common_kernel, f"{debug}_kernel_common")

    if algorithm == "numpy":
        # Buffers we will re-use
        tdata = np.empty(n_fft)
        fdata = np.empty(n_psd, dtype=np.complex128)

        # Loop over FFTs.
        for itod in range(n_tod):
            handle = data[itod]
            _set_input(tdata, handle)
            if debug is not None:
                # Plot the initial time domain buffer
                _debug_plot_tod(tdata, f"{debug}_tin_{itod}")

            # Forward transform
            fdata[:] = np.fft.rfft(tdata, norm="backward")

            if debug is not None:
                # Plot the initial fourier domain buffer
                _debug_plot_fourier(fdata, f"{debug}_fin_{itod}")

            if common_kernel is None:
                if kernel_func is None:
                    krn = _interpolate_kernel(kernels[itod])
                else:
                    krn = kernel_func(itod, freq)
                if debug is not None:
                    _debug_plot_fourier(krn, f"{debug}_kernel_{itod}")
            else:
                krn = common_kernel

            # Convolve with kernel
            if deconvolve:
                fdata[:] /= krn
            else:
                fdata[:] *= krn
            # For a real transform, the nyquist frequency element is real
            fdata.imag[-1] = 0
            # Remove DC level
            fdata[0] = 0
            if debug is not None:
                # Plot the final fourier domain buffer
                _debug_plot_fourier(fdata, f"{debug}_fout_{itod}")

            # Inverse transform
            tdata[:] = np.fft.irfft(fdata, norm="backward")

            # Copy back to input array
            if debug is not None:
                # Plot the final time domain buffer
                _debug_plot_tod(tdata, f"{debug}_tout_{itod}")
            handle[:] = tdata[n_buffer : n_buffer + n_samp]
    elif algorithm == "internal":
        # We are using the internal FFT tools, so we batch-process all transforms
        store = FFTPlanReal1DStore.get()
        fplan = store.forward(n_fft, n_tod)
        rplan = store.backward(n_fft, n_tod)

        # Copy input into buffers
        for itod in range(n_tod):
            td = fplan.tdata(itod)
            handle = data[itod]
            _set_input(td, handle)
            if debug is not None:
                # Plot the initial time domain buffer
                _debug_plot_tod(td, f"{debug}_tin_{itod}")

        # Forward transform
        fplan.exec()

        # Convolve with kernels
        for itod in range(n_tod):
            rplan.fdata(itod)[:] = fplan.fdata(itod)
            fd = rplan.fdata(itod)

            if debug is not None:
                # Plot the initial fourier domain buffer
                fcomplex = np.zeros(n_psd, dtype=np.complex128)
                fcomplex.real[:] = fd[:n_psd]
                fcomplex.imag[1:-1] = fd[-1 : n_psd - 1 : -1]
                _debug_plot_fourier(fcomplex, f"{debug}_fin_{itod}")

            if common_kernel is None:
                if kernel_func is None:
                    krn = _interpolate_kernel(kernels[itod])
                else:
                    krn = kernel_func(itod, freq)
                if debug is not None:
                    _debug_plot_fourier(krn, f"{debug}_kernel_{itod}")
            else:
                krn = common_kernel
            # The real and imaginary parts of the input and kernel,
            # excluding the zero and nyquist frequencies.
            kre = np.real(krn[1:-1])
            kim = np.imag(krn[1:-1])
            fre = np.array(fd[1 : n_psd - 1])
            fim = np.array(fd[-1 : n_psd - 1 : -1])
            nyq = fd[n_psd - 1]
            # We handle the zero and nyquist frequencies separately
            if deconvolve:
                denom = kre**2 + kim**2
                # Real values
                fd[1 : n_psd - 1] = (fre * kre + fim * kim) / denom
                # Nyquist
                fd[n_psd - 1] = (nyq * krn.real[-1]) / (krn.real[-1] ** 2)
                # Imaginary values
                fd[-1 : n_psd - 1 : -1] = (kre * fim - fre * kim) / denom
            else:
                # Real values
                fd[1 : n_psd - 1] = fre * kre - fim * kim
                # Nyquist
                fd[n_psd - 1] = nyq * krn.real[-1]
                # Imaginary values
                fd[-1 : n_psd - 1 : -1] = kre * fim + fre * kim
            # Remove DC level
            fd[0] = 0
            if debug is not None:
                # Plot the final fourier domain buffer
                fcomplex = np.zeros(n_psd, dtype=np.complex128)
                fcomplex.real[:] = fd[:n_psd]
                fcomplex.imag[1:-1] = fd[-1 : n_psd - 1 : -1]
                _debug_plot_fourier(fcomplex, f"{debug}_fout_{itod}")

        # Reverse transform
        rplan.exec()

        # Copy out data
        for itod in range(n_tod):
            td = rplan.tdata(itod)
            if debug is not None:
                # Plot the final time domain buffer
                _debug_plot_tod(td, f"{debug}_tout_{itod}")
            handle = data[itod]
            handle[:] = td[n_buffer : n_buffer + n_samp]

        # Save memory by clearing the fft plans
        store.clear()
    else:
        msg = f"Unknown algorithm choice '{algorithm}'.  Should"
        msg += " be one of 'numpy' or 'internal'."
        raise RuntimeError(msg)
