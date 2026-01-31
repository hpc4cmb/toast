# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Float, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class CalibrateDetectors(Operator):
    """Multiply detector data by calibration factors.

    The calibration factor can be specified as a fixed value for all detectors (for
    example when converting from raw ADC values) or per-detector values can be
    supplied as a dictionary within the observation metadata or a column of the
    focalplane data table.  Detectors which are missing in the calibration dictionary
    are flagged.

    This operator is frequently the first one applied to "raw" data loaded from disk.
    If the dtype of the input DetectorData is an integer type, it will be promoted
    to float64 before applying the calibration.

    The units of the DetectorData can be updated with the `cal_units` trait.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    cal_name = Unicode(
        "calibration", help="The observation or focalplane key containing the gains"
    )

    cal_value = Float(
        None,
        allow_none=True,
        help="Apply this constant value to all detectors.  Overrides `cal_name`",
    )

    cal_units = Unit(
        None,
        allow_none=True,
        help="Update the detector data units",
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
            if self.cal_value is not None:
                cal = {x: self.cal_value for x in dets}
            else:
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
                    msg += "as a dictionary nor in the focalplane database"
                    raise RuntimeError(msg)

            # Check dtype of detector data
            input_dtype = ob.detdata[self.det_data].dtype
            if input_dtype == np.dtype(np.int32) or input_dtype == np.dtype(np.int64):
                # Create a new DetectorData object and copy the original.  Then delete
                # the original and move the new one into place.
                temp_name = f"{self.name}_{self.det_data}_TEMPORARY"
                if self.cal_units is None:
                    temp_units = ob.detdata[self.det_data].units
                else:
                    temp_units = self.cal_units
                ob.detdata.create(
                    temp_name,
                    sample_shape=ob.detdata[self.det_data].sample_shape,
                    dtype=np.float64,
                    detectors=ob.detdata[self.det_data].detectors,
                    units=temp_units,
                )
                for det in ob.detdata[self.det_data].detectors:
                    ob.detdata[temp_name][det] = ob.detdata[self.det_data][det].astype(
                        np.float64
                    )
                del ob.detdata[self.det_data]
                ob.detdata.rename(temp_name, self.det_data)
            else:
                # Just update units if needed
                if self.cal_units is not None:
                    ob.detdata[self.det_data].update_units(self.cal_units)

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
