# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Int, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class CalibrateDetectors(Operator):
    """Multiply detector data by factors in the observation dictionary.

    Given a dictionary in each observation, apply the per-detector scaling factors
    to the timestreams.  Detectors that do not exist in the dictionary are flagged.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    cal_name = Unicode(
        "calibration", help="The observation or focalplane key containing the gains"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for data to calibrate",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    cal_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to apply to detectors with no calibration information",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("cal_mask")
    def _check_cal_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Calibration mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            if self.det_data not in ob.detdata:
                continue

            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            if len(dets) == 0:
                continue

            fp = ob.telescope.focalplane
            if self.cal_name in ob:
                # Observation has a separate calibration table
                cal = ob[self.cal_name]
            elif self.cal_name in fp.properties:
                # Gains are in the focalplane database
                cal = {}
                for det in dets:
                    cal[det] = fp[det][self.cal_name]
            else:
                msg = f"{ob.name}: Gains '{self.cal_name}' do not exist "
                msg += f"as a dictionary nor in the focalplane database"
                raise RuntimeError(msg)

            # Process all detectors
            det_flags = dict(ob.local_detector_flags)
            for det in dets:
                if det not in cal:
                    # Flag this detector
                    det_flags[det] |= self.cal_mask
                    continue
                ob.detdata[self.det_data][det] *= cal[det]

            # Update flags
            ob.update_local_detector_flags(det_flags)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        return {"detdata": [self.det_data]}
