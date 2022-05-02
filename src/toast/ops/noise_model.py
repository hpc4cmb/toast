# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..noise_sim import AnalyticNoise
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger
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

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            if ob.telescope.focalplane.noise is None:
                raise RuntimeError("Focalplane does not have a noise model")
            ob[self.noise_model] = ob.telescope.focalplane.noise

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {"meta": [self.noise_model]}
        return prov
