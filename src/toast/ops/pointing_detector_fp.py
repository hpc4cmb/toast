# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import astropy.units as u
import numpy as np
import traitlets

from .. import qarray as qa
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, Quantity, Unicode, UseEnum, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class PointingDetectorFP(Operator):
    """Operator which returns detector pointing in the Focalplane frame
    (where boresight is constantly pointed at the zenith)

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    boresight = Unicode(
        None,
        allow_none=True,
        help="Boresight is not used by this Operator but is needed to "
        "implement a pointing operator API."
    )

    quats = Unicode(
        defaults.quats,
        allow_none=True,
        help="Observation detdata key for output quaternions",
    )

    coord_in = Unicode(
        None,
        allow_none=True,
        help="The input boresight coordinate system ('C', 'E', 'G')",
    )

    coord_out = Unicode(
        None,
        allow_none=True,
        help="The output coordinate system ('C', 'E', 'G')",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        for trait in "boresight", "coord_in", "coord_out":
            value = getattr(self, trait)
            if value is not None:
                log.warning(
                    f"PointingDetectorFP will not use the provided "
                    f"{trait} = {value}"
                )

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            exists = ob.detdata.ensure(
                self.quats,
                sample_shape=(4,),
                dtype=np.float64,
                detectors=dets,
            )

            if exists:
                if data.comm.group_rank == 0:
                    msg = (
                        f"Group {data.comm.group}, ob {ob.name}, det quats "
                        f"already computed for {dets}"
                    )
                    log.verbose(msg)
                continue

            focalplane = ob.telescope.focalplane
            for det in dets:
                detquat = focalplane[det]["quat"]
                ob.detdata[self.quats][det] = detquat

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.quats],
        }
        return prov
