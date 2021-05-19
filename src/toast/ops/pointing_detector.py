# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Environment, Logger

from ..traits import trait_docs, Int, Unicode, Bool

from ..timing import function_timer

from .. import qarray as qa

from .operator import Operator


@trait_docs
class PointingDetectorSimple(Operator):
    """Operator which translates boresight pointing into detector frame"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional flagging")

    boresight = Unicode("boresight_radec", help="Observation shared key for boresight")

    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output quaternions (for debugging)",
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

    @traitlets.validate("coord_in")
    def _check_coord_in(self, proposal):
        check = proposal["value"]
        if check is not None:
            if check not in ["E", "C", "G"]:
                raise traitlets.TraitError("coordinate system must be 'E', 'C', or 'G'")
        return check

    @traitlets.validate("coord_out")
    def _check_coord_out(self, proposal):
        check = proposal["value"]
        if check is not None:
            if check not in ["E", "C", "G"]:
                raise traitlets.TraitError("coordinate system must be 'E', 'C', or 'G'")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        coord_rot = None
        if self.coord_in is None:
            if self.coord_out is not None:
                msg = "Input and output coordinate systems should both be None or valid"
                raise RuntimeError(msg)
        else:
            if self.coord_out is None:
                msg = "Input and output coordinate systems should both be None or valid"
                raise RuntimeError(msg)
            if self.coord_in == "C":
                if self.coord_out == "E":
                    coord_rot = qa.equ2ecl
                elif self.coord_out == "G":
                    coord_rot = qa.equ2gal
            elif self.coord_in == "E":
                if self.coord_out == "G":
                    coord_rot = qa.ecl2gal
                elif self.coord_out == "C":
                    coord_rot = qa.inv(qa.equ2ecl)
            elif self.coord_in == "G":
                if self.coord_out == "C":
                    coord_rot = qa.inv(qa.equ2gal)
                if self.coord_out == "E":
                    coord_rot = qa.inv(qa.ecl2gal)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Do we already have pointing for all requested detectors?
            if self.quats in ob.detdata:
                quat_dets = ob.detdata[self.quats].detectors
                for d in dets:
                    if d not in quat_dets:
                        break
                else:  # no break
                    # We already have pointing for all specified detectors
                    if data.comm.group_rank == 0:
                        log.verbose(
                            f"Group {data.comm.group}, ob {ob.name}, detector pointing "
                            f"already translated for {dets}"
                        )
                    continue

            # Create (or re-use) output data for the detector quaternions.

            ob.detdata.ensure(
                self.quats,
                sample_shape=(4,),
                dtype=np.float64,
                detectors=dets,
            )

            # Loop over views
            views = ob.view[self.view]
            for vw in range(len(views)):
                # Get the flags if needed
                flags = None
                if self.shared_flags is not None:
                    flags = np.array(views.shared[self.shared_flags][vw])
                    flags &= self.shared_flag_mask

                # Boresight pointing quaternions
                in_boresight = views.shared[self.boresight][vw]

                # Coordinate transform if needed
                boresight = in_boresight
                if coord_rot is not None:
                    boresight = qa.mult(coord_rot, in_boresight)

                # Focalplane for this observation
                focalplane = ob.telescope.focalplane

                for det in dets:
                    # Detector quaternion offset from the boresight.
                    # Real experiments may require additional information
                    # such as parameters of a physical pointing model or
                    # observatory velocity vector (for aberration correction).
                    # In such cases, the detector quaternion can depend on
                    # time and the observing direction and a custom detector
                    # pointing operator needs to be implemented.
                    detquat = np.array(focalplane[det]["quat"], dtype=np.float64)

                    # Timestream of detector quaternions
                    quats = qa.mult(boresight, detquat)
                    if flags is not None:
                        quats[flags != 0] = qa.null_quat
                    views.detdata[self.quats][vw][det] = quats

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.boresight],
            "detdata": list(),
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.quats],
        }
        return prov

    def _accelerators(self):
        return list()
