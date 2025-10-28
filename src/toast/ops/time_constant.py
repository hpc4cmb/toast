# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from .. import rng
from ..fft import convolve
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

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
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

    edge_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Sample flag mask for cutting samples at the ends due to filter effects.",
    )

    batch = Bool(False, help="If True, batch all detectors and process at once")

    deconvolve = Bool(False, help="Deconvolve the time constant instead.")

    realization = Int(0, help="Realization ID, only used if tau_sigma is nonzero")

    debug = Unicode(
        None,
        allow_none=True,
        help="Path to directory for generating debug plots",
    )

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

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors, flagmask=self.tau_flag_mask)
            if len(dets) == 0:
                continue

            fsample = obs.telescope.focalplane.sample_rate.to_value(u.Hz)

            # Get the timeconstants for all detectors
            tau_det = dict()
            for idet, det in enumerate(dets):
                tau = self._get_tau(obs, det)
                if np.isfinite(tau):
                    tau_det[idet] = tau
                else:
                    # Tau is NaN. Flag the detector
                    obs.local_detector_flags[det] |= defaults.det_mask_invalid
                    tau_det[idet] = 0 * u.s

            def _filter_kernel(indx, kfreqs):
                """Function to generate the filter kernel on demand.

                Our complex filter kernel is:
                    1 + j * (2 * pi * tau * freqs)

                """
                tau = tau_det[indx].to_value(u.second)
                kernel = np.zeros(len(kfreqs), dtype=np.complex128)
                kernel.real[:] = 1
                kernel.imag[:] = 2.0 * np.pi * tau * kfreqs
                if not self.deconvolve:
                    kernel = 1.0 / kernel
                return kernel

            # The slice of detector data we will use
            signal = obs.detdata[self.det_data][dets, :]
            if len(dets) == 1:
                # Corner case, signal is a vector, not a list of vectors
                signal = [signal]

            if self.batch:
                # Use the internal batched (threaded) implementation.  This
                # is likely faster, but at the cost of memory use equal to
                # at least 8 times the detector timestream memory for
                # a given observation.
                algo = "internal"
            else:
                # Use numpy, one detector at a time.
                algo = "numpy"

            if self.debug is not None:
                debug_root = os.path.join(self.debug, f"{self.name}_{algo}")
            else:
                debug_root = None

            convolve(
                signal,
                fsample,
                kernel_func=_filter_kernel,
                deconvolve=False,
                algorithm=algo,
                debug=debug_root,
            )

            # Flag 5 time-constants of data at the beginning and end
            for idet, det in enumerate(dets):
                tau = tau_det[idet].to_value(u.second)
                n_edge = int(5 * tau * fsample)
                if n_edge == 0:
                    continue
                obs.detdata[self.det_flags][det][:n_edge] |= self.edge_flag_mask
                obs.detdata[self.det_flags][det][-n_edge:] |= self.edge_flag_mask

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
