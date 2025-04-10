# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import scipy.signal

from ..fft import convolve, r1d_backward, r1d_forward
from ..rng import random
from ..vis import set_matplotlib_backend
from .helpers import create_outdir
from .mpi import MPITestCase


class FFTTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.length = 65536
        self.input_one = random(self.length, counter=[0, 0], key=[0, 0])
        self.compare_one = np.copy(self.input_one)
        self.nbatch = 5
        self.input_batch = np.zeros((self.nbatch, self.length), dtype=np.float64)
        for b in range(self.nbatch):
            self.input_batch[b, :] = random(self.length, counter=[0, 0], key=[0, b])
        self.compare_batch = np.array(self.input_batch)
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def test_roundtrip(self):
        output = r1d_forward(self.input_one)
        check = r1d_backward(output)

        if (self.comm is None) or (self.comm.rank == 0):
            # One process dumps debugging info
            import matplotlib.pyplot as plt

            savefile = os.path.join(self.outdir, "out_one_fft.txt")
            np.savetxt(savefile, np.transpose([self.compare_one, check]), delimiter=" ")
            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(np.arange(self.length), check, c="red", label="Output")
            ax.plot(np.arange(self.length), self.compare_one, c="black", label="Input")
            ax.legend(loc=1)
            plt.title("FFT One Input and Output")
            savefile = os.path.join(self.outdir, "out_one_fft.png")
            plt.savefig(savefile)
            plt.close()

        np.testing.assert_array_almost_equal(check, self.compare_one)

        output = r1d_forward(self.input_batch)
        check = r1d_backward(output)

        if (self.comm is None) or (self.comm.rank == 0):
            # One process dumps debugging info
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            for b in range(self.nbatch):
                savefile = os.path.join(self.outdir, "out_batch_{}_fft.txt".format(b))
                np.savetxt(
                    savefile,
                    np.transpose([self.compare_batch[b], check[b]]),
                    delimiter=" ",
                )
                fig = plt.figure(figsize=(12, 8), dpi=72)
                ax = fig.add_subplot(1, 1, 1, aspect="auto")
                ax.plot(np.arange(self.length), check[b], c="red", label="Output")
                ax.plot(
                    np.arange(self.length),
                    self.compare_batch[b],
                    c="black",
                    label="Input",
                )
                ax.legend(loc=1)
                plt.title("FFT Batch {} Input and Output".format(b))
                savefile = os.path.join(self.outdir, "out_batch_{}_fft.png".format(b))
                plt.savefig(savefile)
                plt.close()

        np.testing.assert_array_almost_equal(check, self.compare_batch)
        return

    def _conv_create_kernel(self, rate, order, kfreqs, delay_freqs):
        b, a = scipy.signal.butter(
            order, rate / 10, btype="low", analog=True, output="ba"
        )
        _, kvals = scipy.signal.freqs(b, a, worN=kfreqs)
        _, delay = scipy.signal.group_delay((b, a), w=delay_freqs, fs=rate)
        return kvals, delay

    def _conv_plot_kernel(self, kfreqs, kvals, outfile):
        set_matplotlib_backend()
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8), dpi=72)
        ax = fig.add_subplot(1, 1, 1, aspect="auto")
        ax.semilogx(kfreqs, kvals)
        ax.set_title("Butterworth filter frequency response")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.margins(0, 0.1)
        ax.grid(which="both", axis="both")
        plt.savefig(outfile)
        plt.close()

    def _conv_plot_signals(self, data, outfile, plot_start, plot_stop):
        set_matplotlib_backend()
        import matplotlib.pyplot as plt

        pslc = slice(plot_start, plot_stop, 1)
        fig = plt.figure(figsize=(12, 8), dpi=72)
        ax = fig.add_subplot(1, 1, 1, aspect="auto")
        for label, times, vals in data:
            ax.plot(times[pslc], vals[pslc], label=label)
        ax.set_title("Signal Timestream")
        ax.set_xlabel("Time [seconds]")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="best")
        plt.savefig(outfile)
        plt.close()

    def _conv_create_signals(self, rate, n_tod, n_samp, flow, fhigh):
        t = (1 / rate) * np.arange(n_samp)
        lowf = np.sin(2 * np.pi * flow * t)
        highf = np.sin(2 * np.pi * fhigh * t)
        sig = lowf + highf
        return (
            np.tile(sig, n_tod).reshape((n_tod, -1)),
            np.tile(lowf, n_tod).reshape((n_tod, -1)),
        )

    def test_convolve_common(self):
        rate = 200.0
        n_samp = 12345
        n_tod = 5
        filter_order = 4
        fftorder = int(np.ceil(np.log(n_samp) / np.log(2)))
        n_fft = 2 ** (fftorder + 1)
        kfreqs = np.fft.rfftfreq(n_fft, d=1.0 / rate)

        # Timestamps
        times = (1 / rate) * np.arange(n_samp)

        # Make some signals consisting of a sine wave above and below
        # the cutoff.
        flow = 5.0
        fhigh = 50.0
        original, lowf = self._conv_create_signals(rate, n_tod, n_samp, flow, fhigh)
        if (self.comm is None) or (self.comm.rank == 0):
            for prange in [(0, n_samp), (0, 500)]:
                savefile = os.path.join(
                    self.outdir, f"conv_common_in_{prange[0]}-{prange[1]}.pdf"
                )
                self._conv_plot_signals(
                    [("Input", times, original[0])],
                    savefile,
                    prange[0],
                    prange[1],
                )

        # Build a lowpass kernel
        kvals, sample_shift = self._conv_create_kernel(
            rate,
            filter_order,
            kfreqs,
            np.array([flow]),
        )

        if (self.comm is None) or (self.comm.rank == 0):
            savefile = os.path.join(self.outdir, "conv_common_kernel.pdf")
            self._conv_plot_kernel(kfreqs, kvals, savefile)

        # The butterworth lowpass introduces a phase shift.  Compute the expected
        # magnitude of this sample shift at our low frequency for later comparision.
        times_shifted = times[:] + (sample_shift[0] / rate)

        # Make a finely-sampled time grid to interpolate the input and phase-shift
        # corrected outputs to in order to compare.
        times_compare = np.linspace(times[100], times[200], num=1000)
        lowf_compare = np.zeros((n_tod, len(times_compare)))
        for itod in range(n_tod):
            lowf_compare[itod] = np.interp(times_compare, times, lowf[itod])

        # Convolve
        signals = dict()
        signals_compare = dict()
        for algo in ["numpy", "internal"]:
            signals[algo] = np.array(original)
            signals_compare[algo] = np.zeros((n_tod, len(times_compare)))
            debug_root = None
            if self.make_plots:
                debug_root = os.path.join(self.outdir, f"conv_common_interp_{algo}")
            convolve(
                signals[algo],
                rate,
                kernel_freq=kfreqs,
                kernels=kvals,
                algorithm=algo,
                debug=debug_root,
            )

            # Remove the sample shift in the output, for easier comparison with
            # the input.
            for itod in range(n_tod):
                signals_compare[algo][itod] = np.interp(
                    times_compare, times_shifted, signals[algo][itod]
                )

            # Check result
            for itod in range(n_tod):
                diff = signals_compare[algo][itod] - lowf_compare[itod]
                passed = np.all(np.absolute(diff) < 0.2)
                if not passed:
                    print(f"{algo} fail: max diff = {np.amax(np.absolute(diff))}")
                    self.assertTrue(False)

        if (self.comm is None) or (self.comm.rank == 0):
            for prange in [(0, n_samp), (0, 500)]:
                savefile = os.path.join(
                    self.outdir,
                    f"conv_common_compare_output_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        ("Input", times, lowf[0]),
                        ("Numpy", times, signals["numpy"][0]),
                        ("Internal", times, signals["internal"][0]),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )
                savefile = os.path.join(
                    self.outdir,
                    f"conv_common_compare_diff_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        (
                            "Numpy - Internal",
                            times,
                            signals["numpy"][0] - signals["internal"][0],
                        ),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )
                savefile = os.path.join(
                    self.outdir,
                    f"conv_common_compare_shifted_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        ("Input", times_compare, lowf_compare[0]),
                        (
                            "Numpy, Shift Removed",
                            times_compare,
                            signals_compare["numpy"][0],
                        ),
                        (
                            "Internal, Shift Removed",
                            times_compare,
                            signals_compare["internal"][0],
                        ),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )

    def test_convolve_kfunc(self):
        rate = 200.0
        n_samp = 12345
        n_tod = 5
        filter_order = 4
        fftorder = int(np.ceil(np.log(n_samp) / np.log(2)))
        n_fft = 2 ** (fftorder + 1)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / rate)

        # Timestamps
        times = (1 / rate) * np.arange(n_samp)

        # Create a function which generates a kernel on demand
        def _kernel_func(indx, kfreqs):
            """Function which generates a kernel on demand.

            `indx` is ignored- the same kernel is always returned.
            """
            b, a = scipy.signal.butter(
                filter_order, rate / 10, btype="low", analog=True, output="ba"
            )
            _, kvals = scipy.signal.freqs(b, a, worN=kfreqs)
            return kvals

        # Make some signals consisting of a sine wave above and below
        # the cutoff.
        flow = 5.0
        fhigh = 50.0
        original, lowf = self._conv_create_signals(rate, n_tod, n_samp, flow, fhigh)
        if (self.comm is None) or (self.comm.rank == 0):
            for prange in [(0, n_samp), (0, 500)]:
                savefile = os.path.join(
                    self.outdir, f"conv_kfunc_in_{prange[0]}-{prange[1]}.pdf"
                )
                self._conv_plot_signals(
                    [("Input", times, original[0])],
                    savefile,
                    prange[0],
                    prange[1],
                )

        # The butterworth lowpass introduces a phase shift.  Compute the expected
        # magnitude of this sample shift at our low frequency for later comparision.
        temp_kvals, sample_shift = self._conv_create_kernel(
            rate,
            filter_order,
            freqs,
            np.array([flow]),
        )
        times_shifted = times[:] + (sample_shift[0] / rate)

        # Make a finely-sampled time grid to interpolate the input and phase-shift
        # corrected outputs to in order to compare.
        times_compare = np.linspace(times[100], times[200], num=1000)
        lowf_compare = np.zeros((n_tod, len(times_compare)))
        for itod in range(n_tod):
            lowf_compare[itod] = np.interp(times_compare, times, lowf[itod])

        # Convolve
        signals = dict()
        signals_compare = dict()
        for algo in ["numpy", "internal"]:
            signals[algo] = np.array(original)
            signals_compare[algo] = np.zeros((n_tod, len(times_compare)))
            debug_root = None
            if self.make_plots:
                debug_root = os.path.join(self.outdir, f"conv_kfunc_interp_{algo}")
            convolve(
                signals[algo],
                rate,
                kernel_func=_kernel_func,
                algorithm=algo,
                debug=debug_root,
            )

            # Remove the sample shift in the output, for easier comparison with
            # the input.
            for itod in range(n_tod):
                signals_compare[algo][itod] = np.interp(
                    times_compare, times_shifted, signals[algo][itod]
                )

            # Check result
            for itod in range(n_tod):
                diff = signals_compare[algo][itod] - lowf_compare[itod]
                passed = np.all(np.absolute(diff) < 0.2)
                if not passed:
                    print(f"{algo} fail: max diff = {np.amax(np.absolute(diff))}")
                    self.assertTrue(False)

        if (self.comm is None) or (self.comm.rank == 0):
            for prange in [(0, n_samp), (0, 500)]:
                savefile = os.path.join(
                    self.outdir,
                    f"conv_kfunc_compare_output_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        ("Input", times, lowf[0]),
                        ("Numpy", times, signals["numpy"][0]),
                        ("Internal", times, signals["internal"][0]),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )
                savefile = os.path.join(
                    self.outdir,
                    f"conv_kfunc_compare_shifted_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        ("Input", times_compare, lowf_compare[0]),
                        (
                            "Numpy, Shift Removed",
                            times_compare,
                            signals_compare["numpy"][0],
                        ),
                        (
                            "Internal, Shift Removed",
                            times_compare,
                            signals_compare["internal"][0],
                        ),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )

    def test_convolve_deconvolve(self):
        rate = 200.0
        n_samp = 12345
        n_tod = 5
        filter_order = 4

        # Timestamps
        times = (1 / rate) * np.arange(n_samp)

        # Create a function which generates a kernel on demand
        def _kernel_func(indx, kfreqs):
            """Function which just attenuates all frequencies"""
            kvals = np.ones(len(kfreqs), dtype=np.complex128)
            kvals.imag[1:-1] = 1.0
            kvals *= 0.5
            return kvals

        # Make some gaussian noise
        original = np.empty((n_tod, n_samp), dtype=np.float64)
        for itod in range(n_tod):
            np.random.seed(itod * n_samp + 12345)
            original[itod] = np.random.normal(loc=0, scale=1.0, size=n_samp)

        if (self.comm is None) or (self.comm.rank == 0):
            for prange in [(0, n_samp), (0, 100)]:
                savefile = os.path.join(
                    self.outdir, f"conv_roundtrip_in_{prange[0]}-{prange[1]}.pdf"
                )
                self._conv_plot_signals(
                    [("Input", times, original[0])],
                    savefile,
                    prange[0],
                    prange[1],
                )

        # Convolve
        signals = dict()
        for algo in ["numpy", "internal"]:
            signals[algo] = np.array(original)
            debug_root = None
            if self.make_plots:
                debug_root = os.path.join(self.outdir, f"conv_roundtrip_interp1_{algo}")
            convolve(
                signals[algo],
                rate,
                kernel_func=_kernel_func,
                deconvolve=False,
                algorithm=algo,
                debug=debug_root,
            )

        # Deconvolve
        signals_compare = dict()
        for algo in ["numpy", "internal"]:
            signals_compare[algo] = np.array(signals[algo])
            debug_root = None
            if self.make_plots:
                debug_root = os.path.join(self.outdir, f"conv_roundtrip_interp2_{algo}")
            convolve(
                signals_compare[algo],
                rate,
                kernel_func=_kernel_func,
                deconvolve=True,
                algorithm=algo,
                debug=debug_root,
            )

        if (self.comm is None) or (self.comm.rank == 0):
            for prange in [(0, n_samp), (0, 100)]:
                savefile = os.path.join(
                    self.outdir,
                    f"conv_roundtrip_compare_convolved_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        ("Input", times, original[0]),
                        ("Numpy", times, signals["numpy"][0]),
                        ("Internal", times, signals["internal"][0]),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )
                savefile = os.path.join(
                    self.outdir,
                    f"conv_roundtrip_compare_deconvolved_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        ("Input", times, original[0]),
                        ("Numpy", times, signals_compare["numpy"][0]),
                        ("Internal", times, signals_compare["internal"][0]),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )
                savefile = os.path.join(
                    self.outdir,
                    f"conv_roundtrip_compare_resid_{prange[0]}-{prange[1]}.pdf",
                )
                self._conv_plot_signals(
                    [
                        ("Numpy", times, (signals_compare["numpy"][0] - original[0])),
                        (
                            "Internal",
                            times,
                            (signals_compare["internal"][0] - original[0]),
                        ),
                    ],
                    savefile,
                    prange[0],
                    prange[1],
                )

        # Check result
        for algo in ["numpy", "internal"]:
            for itod in range(n_tod):
                cslc = slice(50, -50, 1)
                orig = original[itod][cslc]
                comp = signals_compare[algo][itod][cslc]
                bad = np.absolute(orig) < 0.001
                absdiff = np.absolute(comp - orig)
                absdiff[bad] = 0
                if np.amax(absdiff) > 0.1:
                    print(
                        f"{algo}[{itod}] fail: max absdiff = {np.amax(absdiff)}",
                        flush=True,
                    )
                    self.assertTrue(False)
