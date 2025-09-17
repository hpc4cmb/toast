# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..intervals import IntervalList
from ..mpi import MPI
from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, name_UID
from .operator import Operator


@trait_docs
class SignalDiffNoiseModel(Operator):
    """Evaluate a simple white noise model based on consecutive sample
    differences.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(defaults.det_data, help="Observation detdata key to analyze")

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional shared flagging",
    )

    noise_model = Unicode(
        "noise_model", help="The observation key containing the output noise model"
    )

    view = Unicode(
        None,
        allow_none=True,
        help="Evaluate the sample differences in this view",
    )

    fmin = Quantity(1e-6 * u.Hz, help="Minimum frequency to use for noise model.")

    fknee = Quantity(1e-6 * u.Hz, help="Knee frequency to use for noise model.")

    alpha = Float(1.0, help="Slope of the 1/f noise model")

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net_factors = []
        self.total_factors = []
        self.weights_in = []
        self.weights_out = []
        self.rates = []

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if detectors is not None:
            msg = "You must run this operator on all detectors at once"
            log.error(msg)
            raise RuntimeError(msg)

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)
            focalplane = ob.telescope.focalplane
            fsample = focalplane.sample_rate

            shared_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
            signal_units = ob.detdata[self.det_data].units

            # Create the noise model for all detectors, even flagged ones.
            dets = []
            fmin = {}
            fknee = {}
            alpha = {}
            NET = {}
            rates = {}
            indices = {}
            for name in ob.local_detectors:
                dets.append(name)
                rates[name] = fsample
                fmin[name] = self.fmin
                fknee[name] = self.fknee
                alpha[name] = self.alpha
                NET[name] = 0.0 * signal_units / np.sqrt(fsample)
                indices[name] = focalplane[name]["uid"]

            # Set the NET for the good detectors
            for name in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                # Estimate white noise from consecutive sample differences.
                # Neither of the samples can have flags raised.
                sig = ob.detdata[self.det_data][name]
                det_flags = ob.detdata[self.det_flags][name] & self.det_flag_mask
                good = np.logical_and(shared_flags == 0, det_flags == 0)
                sig_diff = sig[1:] - sig[:-1]
                good_diff = np.logical_and(good[1:], good[:-1])
                sigma = np.std(sig_diff[good_diff]) / np.sqrt(2) * signal_units
                net = sigma / np.sqrt(fsample)
                # Store the estimate in a noise model
                NET[name] = net

            ob[self.noise_model] = AnalyticNoise(
                rate=rates,
                fmin=fmin,
                detectors=dets,
                fknee=fknee,
                alpha=alpha,
                NET=NET,
                indices=indices,
            )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": [self.noise_model],
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return prov
