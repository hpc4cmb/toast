# Copyright (c) 2015-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import windows

from ._libtoast import (
    FFTDirection,  # noqa
    FFTPlanReal1D,  # noqa
    FFTPlanReal1DStore,  # noqa
    FFTPlanType,  # noqa
)
from .utils import object_ndim, extend_flags, Logger
from .timing import function_timer, Timer

try:
    import finufft

    have_finufft = True
except ImportError:
    have_finufft = False


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


class AlgorithmBase(object):
    """Base class for convolution algorithms."""

    def __init__(
        self,
        n_tod,
        n_samp,
        rate,
        flags,
        flag_mask,
        kernel_freq,
        kernels,
        kernel_func,
        deconvolve,
    ):
        self.n_tod = n_tod
        self.n_samp = n_samp
        self.rate = rate
        self.flags = flags
        self.flag_mask = flag_mask
        self.kernel_freq = kernel_freq
        self.kernels = kernels
        self.kernel_func = kernel_func
        self.deconvolve = deconvolve

    def _convolve(self, data, debug):
        raise NotImplementedError("convolve method fell through to base class")

    @function_timer
    def convolve(self, data, debug=None):
        """Convolve (or deconvolve) a data array."""
        # Check dimensions
        if len(data) != self.n_tod:
            msg = f"Algorithm convolve data has {len(data)} TODs, not {self.n_tod}"
            raise ValueError(msg)
        for itod, tod in enumerate(data):
            if len(tod) != self.n_samp:
                msg = f"Algorithm convolve data[{itod}] has {len(tod)} samples,"
                msg += f" not {self.n_samp}"
                raise ValueError(msg)
        self._convolve(data, debug)

    def apodization(self, n_reflect):
        """Compute the apodization window"""
        apodize = windows.general_gaussian(
            n_reflect * 2,
            3.0,
            (n_reflect // 2),
            sym=True,
        )[:n_reflect]
        return apodize

    def set_rfft_input(self, tod, tdata, n_buffer, n_reflect, apodize):
        """Helper function to populate the RFFT time domain buffer."""
        if len(tod) != self.n_samp:
            msg = f"Input TOD has {len(tod)} samples instead of {self.n_samp}"
            raise RuntimeError(msg)
        tdata[:] = 0
        tdata[n_buffer - n_reflect : n_buffer] = tod[n_reflect - 1 :: -1]
        tdata[n_buffer : n_buffer + self.n_samp] = tod[:]
        tdata[n_buffer + self.n_samp : n_buffer + self.n_samp + n_reflect] = tod[
            -1 : -(n_reflect + 1) : -1
        ]
        tdata[n_buffer - n_reflect : n_buffer] *= apodize
        tdata[
            n_buffer + self.n_samp + n_reflect - 1 : n_buffer + self.n_samp - 1 : -1
        ] *= apodize

    @function_timer
    def interpolate_rfft_kernel(self, kernel_freq, kern, freq, deconvolve):
        """Helper function to interpolate a single kernel when using RFFT.

        Only used when interpolating an explicit kernel.  Not used with a kernel.
        function.
        """
        # If we are deconvolving, ensure that zero-values are set to something
        # reasonable and small.
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

    def _debug_plot_fourier(self, freq, fdata, plotroot):
        """Helper function to plot fourier domain data."""
        import matplotlib.pyplot as plt

        for frange in [(0, len(fdata)), (0, 500)]:
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

    def _debug_plot_tod(self, n_samp, tdata, plotroot):
        """Helper function to plot time domain data."""
        import matplotlib.pyplot as plt

        for frange in [(0, len(tdata)), (0, 500), (-500, len(tdata))]:
            plotfile = f"{plotroot}_{frange[0]}-{frange[1]}.pdf"
            pslc = slice(frange[0], frange[1], 1)
            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(np.arange(n_samp)[pslc], tdata[pslc])
            ax.set_title("Time Domain Data")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            plt.savefig(plotfile)
            plt.close()


class AlgorithmNumpy(AlgorithmBase):
    """Numpy RFFT algorithm."""

    def __init__(
        self,
        n_tod,
        n_samp,
        rate,
        flags,
        flag_mask,
        kernel_freq,
        kernels,
        kernel_func,
        deconvolve,
    ):
        super().__init__(
            n_tod,
            n_samp,
            rate,
            flags,
            flag_mask,
            kernel_freq,
            kernels,
            kernel_func,
            deconvolve,
        )
        # Compute Fourier domain resolution
        order = int(np.ceil(np.log(n_samp) / np.log(2)))
        self.n_fft = 2 ** (order + 1)

        self.n_psd = self.n_fft // 2 + 1
        self.freq = np.fft.rfftfreq(self.n_fft, d=1.0 / self.rate)
        self.n_buffer = (self.n_fft - self.n_samp) // 2
        self.n_reflect = min(self.n_buffer, self.n_samp)
        self.common_kernel = None
        if self.kernel_func is None and len(self.kernels.shape) == 1:
            # We have a common kernel, interpolate it now
            self.common_kernel = self.interpolate_rfft_kernel(
                self.kernel_freq, self.kernels, self.freq, self.deconvolve
            )

        # Compute apodization window
        self.apodize = self.apodization(self.n_reflect)

    def _convolve(self, data, debug):
        # Buffers we will re-use
        tdata = np.empty(self.n_fft)
        fdata = np.empty(self.n_psd, dtype=np.complex128)

        # Loop over FFTs.
        for itod in range(self.n_tod):
            handle = data[itod]
            self.set_rfft_input(
                handle, tdata, self.n_buffer, self.n_reflect, self.apodize
            )
            if debug is not None:
                # Plot the initial time domain buffer
                self._debug_plot_tod(self.n_fft, tdata, f"{debug}_tin_{itod}")

            # Forward transform
            fdata[:] = np.fft.rfft(tdata, norm="backward")

            if debug is not None:
                # Plot the initial fourier domain buffer
                self._debug_plot_fourier(self.freq, fdata, f"{debug}_fin_{itod}")

            if self.common_kernel is None:
                if self.kernel_func is None:
                    krn = self.interpolate_rfft_kernel(
                        self.kernel_freq, self.kernels[itod], self.freq, self.deconvolve
                    )
                else:
                    krn = self.kernel_func(itod, self.freq)
            else:
                krn = self.common_kernel
            if debug is not None:
                self._debug_plot_fourier(self.freq, krn, f"{debug}_kernel_{itod}")

            # Convolve with kernel
            if self.deconvolve:
                fdata[:] /= krn
            else:
                fdata[:] *= krn
            # For a real transform, the nyquist frequency element is real
            fdata.imag[-1] = 0
            # Remove DC level
            fdata[0] = 0
            if debug is not None:
                # Plot the final fourier domain buffer
                self._debug_plot_fourier(self.freq, fdata, f"{debug}_fout_{itod}")

            # Inverse transform
            tdata[:] = np.fft.irfft(fdata, norm="backward")

            # Copy back to input array
            if debug is not None:
                # Plot the final time domain buffer
                self._debug_plot_tod(self.n_fft, tdata, f"{debug}_tout_{itod}")
            handle[:] = tdata[self.n_buffer : self.n_buffer + self.n_samp]


class AlgorithmInternal(AlgorithmBase):
    """Internal batched RFFTW algorithm."""

    def __init__(
        self,
        n_tod,
        n_samp,
        rate,
        flags,
        flag_mask,
        kernel_freq,
        kernels,
        kernel_func,
        deconvolve,
    ):
        super().__init__(
            n_tod,
            n_samp,
            rate,
            flags,
            flag_mask,
            kernel_freq,
            kernels,
            kernel_func,
            deconvolve,
        )
        # Compute Fourier domain resolution
        order = int(np.ceil(np.log(n_samp) / np.log(2)))
        self.n_fft = 2 ** (order + 1)

        self.n_psd = self.n_fft // 2 + 1
        self.freq = np.fft.rfftfreq(self.n_fft, d=1.0 / self.rate)
        self.n_buffer = (self.n_fft - self.n_samp) // 2
        self.n_reflect = min(self.n_buffer, self.n_samp)
        self.common_kernel = None
        if self.kernel_func is None and len(self.kernels.shape) == 1:
            # We have a common kernel, interpolate it now
            self.common_kernel = self.interpolate_rfft_kernel(
                self.kernel_freq, self.kernels, self.freq, self.deconvolve
            )
        # Compute apodization window
        self.apodize = self.apodization(self.n_reflect)

    def _convolve(self, data, debug):
        # We are using the internal FFT tools, so we batch-process all transforms.
        # This uses several times the memory of the detector data.
        store = FFTPlanReal1DStore.get()
        fplan = store.forward(self.n_fft, self.n_tod)
        rplan = store.backward(self.n_fft, self.n_tod)

        # Copy input into buffers
        for itod in range(self.n_tod):
            td = fplan.tdata(itod)
            handle = data[itod]
            self.set_rfft_input(handle, td, self.n_buffer, self.n_reflect, self.apodize)
            if debug is not None:
                # Plot the initial time domain buffer
                self._debug_plot_tod(self.n_fft, td, f"{debug}_tin_{itod}")

        # Forward transform
        fplan.exec()

        # Convolve with kernels
        for itod in range(self.n_tod):
            rplan.fdata(itod)[:] = fplan.fdata(itod)
            fd = rplan.fdata(itod)

            if debug is not None:
                # Plot the initial fourier domain buffer
                fcomplex = np.zeros(self.n_psd, dtype=np.complex128)
                fcomplex.real[:] = fd[: self.n_psd]
                fcomplex.imag[1:-1] = fd[-1 : self.n_psd - 1 : -1]
                self._debug_plot_fourier(self.freq, fcomplex, f"{debug}_fin_{itod}")

            if self.common_kernel is None:
                if self.kernel_func is None:
                    krn = self.interpolate_rfft_kernel(
                        self.kernel_freq, self.kernels[itod], self.freq, self.deconvolve
                    )
                else:
                    krn = self.kernel_func(itod, self.freq)
            else:
                krn = self.common_kernel
            if debug is not None:
                self._debug_plot_fourier(self.freq, krn, f"{debug}_kernel_{itod}")

            # The real and imaginary parts of the input and kernel,
            # excluding the zero and nyquist frequencies.
            kre = np.real(krn[1:-1])
            kim = np.imag(krn[1:-1])
            fre = np.array(fd[1 : self.n_psd - 1])
            fim = np.array(fd[-1 : self.n_psd - 1 : -1])
            nyq = fd[self.n_psd - 1]
            # We handle the zero and nyquist frequencies separately
            if self.deconvolve:
                denom = kre**2 + kim**2
                # Real values
                fd[1 : self.n_psd - 1] = (fre * kre + fim * kim) / denom
                # Nyquist
                fd[self.n_psd - 1] = (nyq * krn.real[-1]) / (krn.real[-1] ** 2)
                # Imaginary values
                fd[-1 : self.n_psd - 1 : -1] = (kre * fim - fre * kim) / denom
            else:
                # Real values
                fd[1 : self.n_psd - 1] = fre * kre - fim * kim
                # Nyquist
                fd[self.n_psd - 1] = nyq * krn.real[-1]
                # Imaginary values
                fd[-1 : self.n_psd - 1 : -1] = kre * fim + fre * kim
            # Remove DC level
            fd[0] = 0
            if debug is not None:
                # Plot the final fourier domain buffer
                fcomplex = np.zeros(self.n_psd, dtype=np.complex128)
                fcomplex.real[:] = fd[: self.n_psd]
                fcomplex.imag[1:-1] = fd[-1 : self.n_psd - 1 : -1]
                self._debug_plot_fourier(self.freq, fcomplex, f"{debug}_fout_{itod}")

        # Reverse transform
        rplan.exec()

        # Copy out data
        for itod in range(self.n_tod):
            td = rplan.tdata(itod)
            if debug is not None:
                # Plot the final time domain buffer
                self._debug_plot_tod(self.n_fft, td, f"{debug}_tout_{itod}")
            handle = data[itod]
            handle[:] = td[self.n_buffer : self.n_buffer + self.n_samp]

        # Save memory by clearing the fft plans
        store.clear()


class AlgorithmNonUniform(AlgorithmBase):
    """Non-uniform sampling algorithm."""

    def __init__(
        self,
        n_tod,
        n_samp,
        rate,
        flags,
        flag_mask,
        kernel_freq,
        kernels,
        kernel_func,
        deconvolve,
        flag_buffer,
    ):
        super().__init__(
            n_tod,
            n_samp,
            rate,
            flags,
            flag_mask,
            kernel_freq,
            kernels,
            kernel_func,
            deconvolve,
        )
        # Compute Fourier domain resolution
        order = int(np.ceil(np.log(n_samp) / np.log(2)))
        self.n_fft = 2**order

        self.flag_buffer = flag_buffer
        dt = 1.0 / self.rate

        self.raw_freq = np.fft.rfftfreq(self.n_fft, d=1.0 / self.rate)

        self.common_kernel = None
        if self.kernel_func is None and len(self.kernels.shape) == 1:
            # We have a common kernel, interpolate it now
            self.common_kernel = self._interpolate_kernel(
                self.kernel_freq, self.kernels, self.deconvolve
            )

        # Scale data times to be in [-Pi, Pi)
        times = dt * np.arange(self.n_samp, dtype=np.float64)
        tspan = times[-1] + dt
        scale_factor = 2 * np.pi / tspan
        self.scaled_times = times * scale_factor - np.pi

        # Build up the normalizations for round-trip transforms with flags
        self.norm = np.zeros(self.n_tod, dtype=np.float64)
        ttemp = np.empty(self.n_samp, dtype=np.float64)
        ftemp = np.empty(self.n_fft, dtype=np.complex128)
        for itod in range(self.n_tod):
            ttemp[:] = 1.0
            # Interpolate kernel
            if self.common_kernel is None:
                if self.kernel_func is None:
                    krn = self._interpolate_kernel(
                        self.kernel_freq, self.kernels[itod], self.deconvolve
                    )
                else:
                    krn = self._eval_kernel_func(itod, self.kernel_func)
            else:
                krn = self.common_kernel
            if self.flags is None:
                flg = None
            else:
                flg = self.flags[itod]
            self._convolve_single(ttemp, ftemp, flg, self.flag_mask, krn)

            # The result will have ringing around flagged regions.  We extend the
            # flags before evaluating the mean.
            if self.flags is None:
                good = np.ones(len(ttemp), dtype=bool)
            else:
                temp_flags = np.copy(flags[itod])
                extend_flags(temp_flags, self.flag_mask, self.flag_buffer[itod])
                good = temp_flags == 0
            bad = np.logical_not(good)
            ttemp[bad] = 0
            self.norm[itod] = 1.0 / np.mean(ttemp[good])

    def _interpolate_kernel(self, kernel_freq, kern, deconvolve):
        """Helper function to interpolate a single kernel.

        Only used when interpolating an explicit kernel.  Not used with a kernel.
        function.

        This populates a full finufft-compatible buffer.

        """
        # If we are deconvolving, ensure that zero-values are set to something
        # reasonable and small.
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

        # Interpolate to the "usual" rfft frequencies
        mag_out = mag_interp(self.raw_freq)
        ang_out = ang_interp(self.raw_freq)

        # The full Fourier frequencies are kept in CMCL ordering, and include
        # negative frequencies as well.  Populate the complex array that will
        # be used for the product / quotient.
        half_kern = mag_out * np.exp(1j * ang_out)
        out = np.empty(self.n_fft, dtype=np.complex128)
        mid = self.n_fft // 2
        # Positive frequencies, not including nyquist
        out[mid:] = half_kern[:-1]
        # Negative frequencies, starting at -nyquist
        out[:mid] = np.conjugate(half_kern[-1:0:-1])
        return out

    def _eval_kernel_func(self, itod, kfunc):
        """Helper function to evaluate a single kernel function.

        Only used when interpolating an explicit kernel.  Not used with a kernel.
        function.

        This populates a full finufft-compatible buffer.

        """
        # Evaluate the kernel at the "usual" RFFT frequencies.
        half_kern = kfunc(itod, self.raw_freq)

        # The full Fourier frequencies are kept in CMCL ordering, and include
        # negative frequencies as well.  Populate the complex array that will
        # be used for the product / quotient.
        out = np.empty(self.n_fft, dtype=np.complex128)
        mid = self.n_fft // 2
        # Positive frequencies, not including nyquist
        out[mid:] = half_kern[:-1]
        # Negative frequencies, starting at -nyquist
        out[:mid] = np.conjugate(half_kern[-1:0:-1])
        return out

    def _convolve_single(self, td, fd, flags, flag_mask, kernel, norm=1.0, debug=None):
        if flags is None:
            good = np.ones(len(td), dtype=bool)
        else:
            good = (flags & flag_mask) == 0
        n_good = np.count_nonzero(good)
        good_times = self.scaled_times[good]
        good_data = td[good] + 1j * np.zeros(n_good)
        if debug is not None:
            self._debug_plot_fourier(good_times, good_data, f"{debug}_in")
        finufft.nufft1d1(good_times, good_data, n_modes=self.n_fft, out=fd, eps=1.0e-15)

        if debug is not None:
            self._debug_plot_fourier(np.arange(self.n_fft), fd, f"{debug}_fourier")
            self._debug_plot_fourier(np.arange(self.n_fft), kernel, f"{debug}_kernel")
        if self.deconvolve:
            fd[:] /= kernel
        else:
            fd[:] *= kernel
        # Remove DC level
        fd[0] = 0.0

        if debug is not None:
            self._debug_plot_fourier(np.arange(self.n_fft), fd, f"{debug}_postkern")
        finufft.nufft1d2(good_times, fd, out=good_data, eps=1e-15)
        td[good] = good_data.real
        td[good] *= norm
        if debug is not None:
            self._debug_plot_fourier(good_times, good_data, f"{debug}_adjoint")
            self._debug_plot_tod(len(td), td, f"{debug}_out")

    def _convolve(self, data, debug):
        ftemp = np.empty(self.n_fft, dtype=np.complex128)
        for itod in range(self.n_tod):
            # Interpolate kernel
            if debug is None:
                debug_tod = None
            else:
                debug_tod = f"{debug}_{itod}"
            if self.common_kernel is None:
                if self.kernel_func is None:
                    krn = self._interpolate_kernel(
                        self.kernel_freq, self.kernels[itod], self.deconvolve
                    )
                else:
                    krn = self._eval_kernel_func(itod, self.kernel_func)
            else:
                krn = self.common_kernel
            if debug is not None:
                self._debug_plot_tod(self.n_samp, data[itod], f"{debug_tod}_tod")
                self._debug_plot_fourier(
                    np.arange(self.n_fft), krn, f"{debug_tod}_kernel"
                )
            if self.flags is None:
                flg = None
            else:
                flg = self.flags[itod]
            self._convolve_single(
                data[itod],
                ftemp,
                flg,
                self.flag_mask,
                krn,
                norm=self.norm[itod],
                debug=debug_tod,
            )


def convolve(
    data,
    rate,
    flags=None,
    flag_mask=None,
    kernel_freq=None,
    kernels=None,
    kernel_func=None,
    deconvolve=False,
    algorithm="numpy",
    debug=None,
):
    """Convolve timestream data with Fourier domain kernels.

    data should be an iterable, and the first dimension is the number of timestreams.
    The Fourier domain resolution is chosen to be the smallest radix-2 value sufficient
    to store twice the data length.

    If `kernels` is a 2D array, there should be one per timestream, and all kernels
    should be specified on the same frequency grid given by `kernel_freq`.  If
    `kernels` is 1D, the same kernel will be applied to all timestreams.  The kernels
    will be interpolated in frequency space to match the FFT resolution.
    Alternatively, if `kernel_func` is specified, it will be called with the detector
    index and the Fourier domain frequencies.  The function should return the complex
    kernel values at these frequencies.

    If `algorithm` = "numpy", then numpy rfft methods are used.  If `algorithm`
    is "internal", the compiled FFTW internal wrappers are used and this will thread
    over detectors (and use N_detector times as much memory).  In both these cases,
    the input data is centered in the time-domain buffer and is reflected about the
    endpoints to avoid discontinuities.  Any additional samples on either end are zero
    padded.  The reflected data is further apodized smoothly to zero with a Gaussian
    window.

    If `algorithm` = "nonuniform", a non-uniform, type-1 transform is performed to go
    from real space to Fourier domain using only valid time domain samples.  After
    multiplication by the kernel, a type-2 transform is performed to go from from
    Fourier domain back to real space.  This adjoint transform is not an inverse. This
    roundtrip will introduce an arbitrary normalization factor.  In order to compute
    the normalization, a unit-valued input with the same flagging structure is passed
    through the convolution.

    If `algorithm` is None, then the technique is selected based on the input flags.
    If flags is None, the numpy method is chosen and otherwise the nonuniform path is
    selected.

    If `flags` is not None, then some level of ringing will be produced by the
    convolution near the edges of flagged samples.  The flagged regions will be
    enlarged, and the number of samples required to do this is computed by looking
    at the spread in samples produced by passing an impulse (delta function) through
    the convolution.

    The input data is modified in-place.

    Args:
        data (iterable):  Sequence of timestreams.
        rate (float):  Sample rate in Hz.
        flags (iterable):  If not None, extend the per-detector flags according to
            the kernel size.
        flag_mask (int):  The value to bitwise AND with the flags to identify
            bad samples.
        kernel_freq (array):  The common frequency values for all input kernels.
        kernels (array):  Array of Fourier domain kernels.
        deconvolve (bool):  If True, divide by the kernel instead of multiplying.
        algorithm (str):  Possible values are None, "numpy", "internal", and
            "nonuniform".
        debug (str):  If not None, this should be the root file name of debugging
            plots that will be generated.

    Returns:
        None

    """
    log = Logger.get()
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

    if (flags is None and flag_mask is not None) or (
        flags is not None and flag_mask is None
    ):
        msg = "Both flags and flag_mask must be specified or set to None"
        raise RuntimeError(msg)

    n_tod = len(data)
    n_samp = len(data[0])
    for itod in range(1, n_tod):
        if len(data[itod]) != n_samp:
            msg = "All detector arrays should be the same length"
            raise RuntimeError(msg)

    timer = Timer()
    timer.start()

    extend = np.zeros(n_tod, dtype=np.int32)
    if flags is not None:
        # We will be extending the flagged regions.  Compute the spread induced by
        # the convolution on a delta function impulse.  We use the numpy algorithm
        # for this to save memory.
        spread_algo = AlgorithmNumpy(
            n_tod,
            n_samp,
            rate,
            flags,
            flag_mask,
            kernel_freq,
            kernels,
            kernel_func,
            deconvolve,
        )
        mid = n_samp // 2
        peak = 100.0
        threshold = 0.02
        temp = np.zeros_like(data)
        temp[:, mid] = peak
        spread_algo.convolve(temp)
        for itod in range(n_tod):
            atemp = np.absolute(temp[itod])
            ipeak = np.argmax(atemp)
            apeak = atemp[ipeak]
            imin = ipeak
            while imin > 0 and atemp[imin] > threshold * apeak:
                imin -= 1
            imax = ipeak
            while imax < n_samp and atemp[imax] > threshold * apeak:
                imax += 1
            extend[itod] = imax - imin
            if extend[itod] == n_samp:
                msg = "Impulse response spreads to all samples"
                raise RuntimeError(msg)
        msg = "Convolve: compute impulse response in"
        log.debug_rank(msg, comm=None, timer=timer)

    if algorithm is None:
        # FIXME: Choose based on flagging.
        # Until the non-uniform case receives more testing, do not use it
        # unless explicitly requested.
        # if flags is None:
        #     algorithm = "numpy"
        # else:
        #     algorithm = "nonuniform"
        algorithm = "numpy"

    if algorithm == "numpy":
        algo = AlgorithmNumpy(
            n_tod,
            n_samp,
            rate,
            flags,
            flag_mask,
            kernel_freq,
            kernels,
            kernel_func,
            deconvolve,
        )
    elif algorithm == "internal":
        algo = AlgorithmInternal(
            n_tod,
            n_samp,
            rate,
            flags,
            flag_mask,
            kernel_freq,
            kernels,
            kernel_func,
            deconvolve,
        )
    elif algorithm == "nonuniform":
        if not have_finufft:
            msg = "Please install the 'finufft' package"
            raise RuntimeError(msg)
        algo = AlgorithmNonUniform(
            n_tod,
            n_samp,
            rate,
            flags,
            flag_mask,
            kernel_freq,
            kernels,
            kernel_func,
            deconvolve,
            extend,
        )
    else:
        msg = f"Unknown algorithm '{algorithm}'.  Allowed values are "
        msg += "'numpy', 'internal', 'nonuniform' and None."
        raise RuntimeError(msg)

    # Do the data convolution
    algo.convolve(data, debug)
    msg = "Convolve: compute algorithm convolve in"
    log.debug_rank(msg, comm=None, timer=timer)

    # Extend flagging if needed.  Also flag samples at either end.
    if flags is not None:
        for itod in range(n_tod):
            ext = extend[itod]
            extend_flags(flags[itod], flag_mask, ext)
            flags[itod][:ext] |= flag_mask
            flags[itod][-ext:] |= flag_mask
        msg = "Convolve: extended flagged regions in"
        log.debug_rank(msg, comm=None, timer=timer)
