# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ...accelerator import ImplementationType
from ...noise_sim import AnalyticNoise
from ...timing import Timer, function_timer
from ...traits import Int, Unicode, UseEnum, trait_docs
from ...utils import Environment, Logger
from ..operator import Operator
from .kernels import noise_weight


@trait_docs
class NoiseWeight(Operator):
    """Apply diagonal noise weighting to detector data.

    This simple operator takes the detector weight from the specified noise model and
    applies it to the timestream values.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode(
        "noise_model", help="The observation key containing the noise model"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            data_input_units = ob.detdata[self.det_data].units
            data_invcov_units = 1.0 / data_input_units**2
            data_output_units = 1.0 / data_input_units

            # Check that the noise model exists
            if self.noise_model not in ob:
                msg = "Noise model {} does not exist in observation {}".format(
                    self.noise_model, ob.name
                )
                raise RuntimeError(msg)

            # Compute the noise for each detector (using the correct units)
            noise = ob[self.noise_model]
            detector_weights = np.array(
                [
                    noise.detector_weight(detector).to(data_invcov_units).value
                    for detector in dets
                ],
                dtype=np.float64,
            )

            # Multiply detectors by their respective noise weight
            intervals = ob.intervals[self.view].data
            det_data = ob.detdata[self.det_data].data
            det_data_indx = ob.detdata[self.det_data].indices(dets)
            noise_weight(
                det_data,
                det_data_indx,
                intervals,
                detector_weights,
                impl=implementation,
                use_accel=use_accel,
            )

            # Update the units of the output
            ob.detdata[self.det_data].update_units(data_output_units)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.noise_model],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        return {"detdata": [self.det_data]}

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        return True
