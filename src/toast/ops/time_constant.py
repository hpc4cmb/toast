# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from .. import rng
from ..fft import FFTPlanReal1DStore
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


@trait_docs
class TimeConstant(Operator):
    """Simple time constant filtering without flag checks."""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to",
    )

    tau = Quantity(
        None,
        allow_none=True,
        help="Time constant to apply to all detectors.  Overrides `tau_name`",
    )

    tau_sigma = Float(
        None,
        allow_none=True,
        help="Randomized fractional error to add to each time constant.",
    )

    tau_name = Unicode(
        None,
        allow_none=True,
        help="Key to use to find time constants in the Focalplane.",
    )

    tau_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Detector flag mask for cutting detectors with invalid Tau values.",
    )

    batch = Bool(False, help="If True, batch all detectors and process at once")

    deconvolve = Bool(False, help="Deconvolve the time constant instead.")

    realization = Int(0, help="Realization ID, only used if tau_sigma is nonzero")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_tau(self, obs, det):
        focalplane = obs.telescope.focalplane
        if self.tau is None:
            tau = focalplane[det][self.tau_name]
            try:
                tau = tau.to(u.second)
            except AttributeError:
                # The value is just a float in seconds (or NaN)
                tau = tau * u.second
        else:
            tau = self.tau
        if self.tau_sigma:
            # randomize tau in a reproducible manner
            counter1 = obs.session.uid
            counter2 = self.realization
            key1 = focalplane[det]["uid"]
            key2 = 123456

            x = rng.random(
                1,
                sampler="gaussian",
                key=(key1, key2),
                counter=(counter1, counter2),
            )[0]
            tau = tau * (1 + x * self.tau_sigma)
        return tau

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        if self.tau is None and self.tau_name is None:
            raise RuntimeError("Either tau or tau_name must be set.")

        # The store of FFT plans which we will re-use for all observations
        store = FFTPlanReal1DStore.get()

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                continue
            fsample = obs.telescope.focalplane.sample_rate

            nsample = obs.n_local_samples

            # Compute the radix-2 FFT length to use
            fftlen = 2
            while fftlen <= 2 * nsample:
                fftlen *= 2
            npsd = fftlen // 2 + 1
            npad = fftlen - nsample
            nfirsthalf = nsample // 2
            if nsample % 2 == 0:
                nsecondhalf = nfirsthalf
            else:
                nsecondhalf = nfirsthalf + 1

            freqs = np.fft.rfftfreq(fftlen, 1 / fsample.to_value(u.Hz))

            # The forward and reverse plans
            if self.batch:
                fplan = store.forward(fftlen, len(dets))
                rplan = store.backward(fftlen, len(dets))
            else:
                fplan = store.forward(fftlen, 1)
                rplan = store.backward(fftlen, 1)

            # Temporary buffer for the imaginary terms of the filter
            imgfilt = np.zeros(len(freqs) - 2)
            imgsq = np.zeros(len(freqs) - 2)

            def _fill_forward(fp, indx, sig):
                """Fill the forward buffer for one detector.

                To avoid boundary effects, we continue the signal with a
                time-reversed copy of itself

                """
                # Original signal
                #print(f"DBG: 0:{nsample} = sig[0:{len(sig)}]")
                fp.tdata(indx)[:nsample] = sig
                # Continue with time-reversed second half
                #print(f"DBG: {nsample}:{nsample+nsecondhalf} = sig[-1:{-nsecondhalf-1}]")
                fp.tdata(indx)[nsample:nsample + nsecondhalf] = sig[:-nsecondhalf-1:-1]
                # Pad with the center value
                #print(f"DBG: {nsample+nsecondhalf}:{-nfirsthalf} = pad")
                fp.tdata(indx)[nsample + nsecondhalf:-nfirsthalf] = sig[nfirsthalf]
                # End with the time-reversed first half
                #print(f"DBG: {-nfirsthalf}:-1 = sig[{nfirsthalf-1}:0]")
                fp.tdata(indx)[-nfirsthalf:] = sig[nfirsthalf - 1::-1]

            def _convolve_tau(fp, rp, indx, taudet):
                """Multiply a single fourier domain buffer with the kernel"""
                # Our complex filter kernel is:
                #
                #   1 + 2 * pi * tau * freqs
                #
                # So the real part is "1" and we do not need to store that.  We
                # just need to compute the imaginary terms:
                imgfilt[:] = 2.0 * np.pi * taudet * freqs[1:-1]

                # Apply the kernel.  Here we are multiplying or dividing complex
                # numbers.  The simplified steps here are the operations needed
                # if one writes down the algebra and uses the fact that the real part
                # of the filter is one.  We store the result in the fourier domain
                # buffer of the reverse plan.
                #
                # If X = a + b*i, Y = c + d*i, and c == 1, then:
                #   X * Y = (a - b*d) + (a*d + b)i
                #   X / Y = (a + b*d) / (1 + d^2) + (b - a*d)i / (1 + d^2)
                #
                # Helper views that put the frequencies in the right order.  We
                # handle the zero and Nyquist frequencies separately.
                redata = fp.fdata(indx)[1 : npsd - 1]
                imgdata = fp.fdata(indx)[-1 : npsd - 1 : -1]

                if self.deconvolve:
                    rp.fdata(indx)[0] = fp.fdata(indx)[0]
                    rp.fdata(indx)[1 : npsd - 1] = redata - imgdata * imgfilt
                    rp.fdata(indx)[npsd - 1] = fp.fdata(indx)[npsd - 1]
                    rp.fdata(indx)[-1 : npsd - 1 : -1] = redata * imgfilt + imgdata
                else:
                    imgsq[:] = 1.0 / (1.0 + imgfilt * imgfilt)
                    rp.fdata(indx)[0] = fp.fdata(indx)[0]
                    rp.fdata(indx)[1 : npsd - 1] = (
                        redata + imgdata * imgfilt
                    ) * imgsq
                    rp.fdata(indx)[npsd - 1] = fp.fdata(indx)[npsd - 1]
                    rp.fdata(indx)[-1 : npsd - 1 : -1] = (
                        imgdata - redata * imgfilt
                    ) * imgsq

            if self.batch:
                # We are processing all local detectors at once.  This allows us
                # to thread over detectors, but require 4x the amount of detector
                # memory.  It should only be used if running fewer processes and
                # more threads.
                # Fill forward buffer
                for idet, det in enumerate(dets):
                    signal = obs.detdata[self.det_data][det]
                    _fill_forward(fplan, idet, signal)
                # Execute forward transform
                fplan.exec()
                # Convolve with filter
                for idet, det in enumerate(dets):
                    tau = self._get_tau(obs, det)
                    if np.isnan(tau):
                        old_flag = obs.local_detector_flags[det]
                        obs.update_local_detector_flags(
                            {det: old_flag | self.tau_flag_mask}
                        )
                        continue
                    tau_s = tau.to_value(u.s)
                    _convolve_tau(fplan, rplan, idet, tau_s)
                # Inverse transform
                rplan.exec()
                # Copy result into place
                for idet, det in enumerate(dets):
                    obs.detdata[self.det_data][det] = rplan.tdata(idet)[:nsample]
            else:
                # Process each detector one at a time.
                for idet, det in enumerate(dets):
                    signal = obs.detdata[self.det_data][det]
                    # Fill forward buffer
                    _fill_forward(fplan, 0, signal)
                    # Execute forward transform
                    fplan.exec()
                    # Convolve with filter
                    tau = self._get_tau(obs, det)
                    if np.isnan(tau):
                        old_flag = obs.local_detector_flags[det]
                        obs.update_local_detector_flags(
                            {det: old_flag | self.tau_flag_mask}
                        )
                        continue
                    tau_s = tau.to_value(u.s)
                    _convolve_tau(fplan, rplan, 0, tau_s)
                    # Inverse transform
                    rplan.exec()
                    # Copy result into place
                    obs.detdata[self.det_data][det] = rplan.tdata(0)[:nsample]

            # # Original signal
            # fplan.tdata(idet)[:nsample] = signal
            # # Continue with time-reversed second half
            # fplan.tdata(idet)[nsample : nsample + nhalf] = signal[: nhalf - 1 : -1]
            # # Pad with the center value
            # fplan.tdata(idet)[nsample + nhalf : -nsample - nhalf] = signal[nhalf]
            # # End with the time-reversed first half
            # fplan.tdata(idet)[-nhalf:] = signal[nhalf - 1 :: -1]

            # Clear the FFT plans after each observation to save memory.  This means
            # we are just using the plan store for convenience and not re-using the
            # allocated memory.
            store.clear()

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": list(),
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        return dict()
