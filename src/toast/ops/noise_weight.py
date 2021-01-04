# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import traitlets

from ..utils import Environment, Logger

from ..timing import function_timer, Timer

from ..noise_sim import AnalyticNoise

from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity

from .operator import Operator


@trait_docs
class NoiseWeight(Operator):
    """Apply diagonal noise weighting to detector data.

    This simple operator takes the detector weight from the specified noise model and
    applies it to the timestream values.

    """

    # Class traits

    API = traitlets.Int(0, help="Internal interface version for this operator")

    noise_model = traitlets.Unicode(
        "noise_model", help="The observation key containing the noise model"
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Check that the noise model exists
            if self.noise_model not in ob:
                msg = "Noise model {} does not exist in observation {}".format(
                    self.noise_model, ob.name
                )
                raise RuntimeError(msg)

            noise = ob[self.noise_model]

            for d in dets:
                # Get the detector weight from the noise model.
                detweight = noise.detector_weight(det)

                # Apply
                ob.detdata[self.det_data][d] *= detweight
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {"meta": [self.noise_model], "detdata": [self.det_data]}
        return req

    def _provides(self):
        return dict()

    def _accelerators(self):
        return list()
