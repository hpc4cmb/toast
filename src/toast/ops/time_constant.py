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
from ..utils import Environment, GlobalTimers, Logger, Timer, dtype_to_aligned
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

    deconvolve = Bool(False, help="Deconvolve the time constant instead.")

    realization = Int(0, help="Realization ID, only used if tau_sigma is nonzero")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_tau(self, obs, det):
        focalplane = obs.telescope.focalplane
        if self.tau is None:
            tau = focalplane[det][self.tau_name]
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
            nhalf = nsample // 2

            freqs = np.fft.rfftfreq(fftlen, 1 / fsample.to_value(u.Hz))

            # The forward and reverse plans
            fplan = store.forward(fftlen, len(dets))
            rplan = store.backward(fftlen, len(dets))

            # Fill the forward buffers.  To avoid boundary effects, we continue
            # the signal with a time-reversed copy of itself
            for idet, det in enumerate(dets):
                signal = obs.detdata[self.det_data][det]
                # Original signal
                fplan.tdata(idet)[:nsample] = signal
                # Continue with time-reversed second half
                fplan.tdata(idet)[nsample : nsample + nhalf] = signal[: nhalf - 1 : -1]
                # Pad with the center value
                fplan.tdata(idet)[nsample + nhalf : -nsample - nhalf] = signal[nhalf]
                # End with the time-reversed first half
                fplan.tdata(idet)[-nhalf:] = signal[nhalf - 1 :: -1]

            # Do the forward transforms
            fplan.exec()

            # Temporary buffer for the imaginary terms of the filter
            imgfilt = np.zeros(len(freqs) - 2)
            imgsq = np.zeros(len(freqs) - 2)

            # Apply Fourier domain filter and fill the frequency-domain
            # buffers of the reverse transform.
            for idet, det in enumerate(dets):
                tau = self._get_tau(obs, det)
                tau_s = tau.to_value(u.s)

                # Our complex filter kernel is:
                #
                #   1 + 2 * pi * tau * freqs
                #
                # So the real part is "1" and we do not need to store that.  We
                # just need to compute the imaginary terms:
                imgfilt[:] = 2.0 * np.pi * tau_s * freqs[1:-1]

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
                redata = fplan.fdata(idet)[1 : npsd - 1]
                imgdata = fplan.fdata(idet)[-1 : npsd - 1 : -1]

                if self.deconvolve:
                    rplan.fdata(idet)[0] = fplan.fdata(idet)[0]
                    rplan.fdata(idet)[1 : npsd - 1] = redata - imgdata * imgfilt
                    rplan.fdata(idet)[npsd - 1] = fplan.fdata(idet)[npsd - 1]
                    rplan.fdata(idet)[-1 : npsd - 1 : -1] = redata * imgfilt + imgdata
                else:
                    imgsq[:] = 1.0 / (1.0 + imgfilt * imgfilt)
                    rplan.fdata(idet)[0] = fplan.fdata(idet)[0]
                    rplan.fdata(idet)[1 : npsd - 1] = (
                        redata + imgdata * imgfilt
                    ) * imgsq
                    rplan.fdata(idet)[npsd - 1] = fplan.fdata(idet)[npsd - 1]
                    rplan.fdata(idet)[-1 : npsd - 1 : -1] = (
                        imgdata - redata * imgfilt
                    ) * imgsq

            # Do the inverse transforms
            rplan.exec()

            # Copy timestreams back
            for idet, det in enumerate(dets):
                obs.detdata[self.det_data][det] = rplan.tdata(idet)[:nsample]

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
