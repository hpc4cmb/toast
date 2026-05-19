# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import ephem
import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..coordinates import to_DJD
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, List, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator
from .pipeline import Pipeline

XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class FlagNaNs(Operator):
    """Operator which flags NaN values in detector data and replaces them with 0"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(defaults.det_mask_invalid, help="Bit mask to raise flags with")

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            input_det_flags = obs.local_detector_flags
            output_det_flags = {}
            for det in dets:
                signal = obs.detdata[self.det_data][det]
                flags = obs.detdata[self.det_flags][det]

                good = np.isfinite(signal)
                bad = np.logical_not(good)

                if np.all(bad):
                    # Flag the detector and all data
                    signal[:] = 0
                    flags |= self.det_flag_mask
                    output_det_flags[det] = input_det_flags[det] | self.det_mask
                else:
                    # Flag bad samples
                    signal[bad] = 0
                    flags[bad] |= self.det_flag_mask

            obs.update_local_detector_flags(output_det_flags)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data, self.det_flags],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data, self.det_flags],
        }
        return prov
