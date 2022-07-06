# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from .. import qarray as qa
from .._libtoast import pointing_detector
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, Unicode, trait_docs
from ..utils import Environment, Logger
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
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    boresight = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight"
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

    use_python = Bool(False, help="If True, use python implementation")

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
    def _exec(self, data, detectors=None, use_accel=False, **kwargs):
        log = Logger.get()

        if self.use_python and use_accel:
            raise RuntimeError("Cannot use accelerator with pure python implementation")

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

            exists = ob.detdata.ensure(
                self.quats,
                sample_shape=(4,),
                dtype=np.float64,
                detectors=dets,
                accel=use_accel,
            )

            if exists:
                if data.comm.group_rank == 0:
                    msg = (
                        f"Group {data.comm.group}, ob {ob.name}, det quats "
                        f"already computed for {dets}"
                    )
                    log.verbose(msg)
                continue

            # FIXME:  temporary hack until instrument classes are also pre-staged
            # to GPU
            focalplane = ob.telescope.focalplane
            fp_quats = np.zeros((len(dets), 4), dtype=np.float64)
            for idet, d in enumerate(dets):
                fp_quats[idet, :] = focalplane[d]["quat"]

            quat_indx = ob.detdata[self.quats].indices(dets)

            if use_accel:
                if not ob.detdata.accel_exists(self.quats):
                    ob.detdata.accel_create(self.quats)

            if self.shared_flags is None:
                flags = np.zeros(1, dtype=np.uint8)
            else:
                flags = ob.shared[self.shared_flags].data

            log.verbose_rank(
                f"Operator {self.name}, observation {ob.name}, use_accel = {use_accel}",
                comm=data.comm.comm_group,
            )

            # FIXME: handle coordinate transforms here too...

            if self.use_python:
                self._py_pointing_detector(
                    fp_quats,
                    ob.shared[self.boresight].data,
                    quat_indx,
                    ob.detdata[self.quats].data,
                    ob.intervals[self.view].data,
                    flags,
                )
            else:
                pointing_detector(
                    fp_quats,
                    ob.shared[self.boresight].data,
                    quat_indx,
                    ob.detdata[self.quats].data,
                    ob.intervals[self.view].data,
                    flags,
                    self.shared_flag_mask,
                    use_accel,
                )

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

    def _supports_accel(self):
        return True

    def _py_pointing_detector(
        self,
        fp_quats,
        bore_data,
        quat_indx,
        quat_data,
        intr_data,
        flag_data,
    ):
        """Internal python implementation for comparison tests."""
        for idet in range(len(quat_indx)):
            qidx = quat_indx[idet]
            for vw in intr_data:
                samples = slice(vw.first, vw.last + 1, 1)
                bore = np.array(bore_data[samples])
                if self.shared_flags is not None:
                    good = (flag_data[samples] & self.shared_flag_mask) == 0
                    bore[np.invert(good)] = np.array([0, 0, 0, 1], dtype=np.float64)
                quat_data[qidx][samples] = qa.mult(bore, fp_quats[idet])
