# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from astropy import units as u
import numpy as np
import traitlets

from ..mpi import MPI, MPI_Comm, use_mpi, Comm

from .operator import Operator
from .. import qarray as qa
from .. import rng
from ..timing import function_timer
from ..traits import trait_docs, Int, Unicode, Bool, Dict, Quantity, Float
from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned
from ..observation import default_names as obs_names


@trait_docs
class TimeConstant(Operator):
    """Simple time constant filtering without flag checks."""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        obs_names.det_data,
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
        return

    def _get_tau(self, obs, det):
        focalplane = obs.telescope.focalplane
        if self.tau is None:
            tau = focalplane[det][self.tau_name]
        else:
            tau = self.tau
        if self.tau_sigma:
            # randomize tau in a reproducible manner
            counter1 = obs.uid
            counter2 = self.realization
            key1 = focalplane[det]["uid"]
            key2 = 123456

            x = rng.random(
                1,
                sampler="gaussian",
                key=(key1, key2),
                counter=(counter1, counter2),
            )[0]
            tau *= 1 + x * self.tau_sigma
        return tau

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        if self.tau is None and self.tau_name is None:
            raise RuntimeError("Either tau or tau_name must be set.")

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                continue
            fsample = obs.telescope.focalplane.sample_rate

            # FIXME : if the FFTs here ever become a bottle neck, we will
            # need to switch to using the TOAST batched FFT instead.

            nsample = obs.n_local_samples
            freqs = np.fft.rfftfreq(nsample, 1 / fsample.to_value(u.Hz))

            for det in dets:
                signal = obs.detdata[self.det_data][det]

                tau = self._get_tau(obs, det)
                if self.deconvolve:
                    taufilter = 1 + 2.0j * np.pi * freqs * tau.to_value(u.s)
                else:
                    taufilter = 1.0 / (1 + 2.0j * np.pi * freqs * tau.to_value(u.s))

                signal[:] = np.fft.irfft(taufilter * np.fft.rfft(signal), n=nsample)

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

    def _accelerators(self):
        return list()
