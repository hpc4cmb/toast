# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import traitlets

from astropy import units as u

from ..utils import Environment, Logger

from ..timing import function_timer, Timer

from ..noise_sim import AnalyticNoise

from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity

from .operator import Operator


@trait_docs
class DefaultNoiseModel(Operator):
    """Create a default noise model from focalplane parameters.

    A noise model is used by other operations such as simulating noise timestreams
    and also map making.  This operator uses the detector properties from the
    focalplane in each observation to create a simple AnalyticNoise model.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode(
        "noise_model", help="The observation key for storing the noise model"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        comm = data.comm

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # The focalplane for this observation
            focalplane = ob.telescope.focalplane

            # Every process has a copy of the focalplane, and every process may want
            # the noise model for all detectors (not just our local detectors).
            # So we simply have every process generate the same noise model locally.

            fmin = {}
            fknee = {}
            alpha = {}
            NET = {}
            rates = {}
            for d in dets:
                rates[d] = focalplane.sample_rate
                fmin[d] = focalplane[d]["psd_fmin"]
                fknee[d] = focalplane[d]["psd_fknee"]
                alpha[d] = focalplane[d]["psd_alpha"]
                NET[d] = focalplane[d]["psd_net"]

            noise = AnalyticNoise(
                rate=rates, fmin=fmin, detectors=dets, fknee=fknee, alpha=alpha, NET=NET
            )

            ob[self.noise_model] = noise

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {"meta": [self.noise_model]}
        return prov

    def _accelerators(self):
        return list()
