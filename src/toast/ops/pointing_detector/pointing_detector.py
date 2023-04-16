# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ... import qarray as qa
from ...accelerator import ImplementationType
from ...observation import default_values as defaults
from ...timing import function_timer
from ...traits import Bool, Int, Unicode, UseEnum, trait_docs
from ...utils import Logger
from ..operator import Operator
from .kernels import pointing_detector


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
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

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

            # FIXME:  temporary hack until instrument classes are also pre-staged to GPU
            focalplane = ob.telescope.focalplane
            fp_quats = np.zeros((len(dets), 4), dtype=np.float64)
            for idet, d in enumerate(dets):
                fp_quats[idet, :] = focalplane[d]["quat"]

            quat_indx = ob.detdata[self.quats].indices(dets)

            if self.shared_flags is None:
                flags = np.zeros(1, dtype=np.uint8)
            else:
                flags = ob.shared[self.shared_flags].data

            log.verbose_rank(
                f"Operator {self.name}, observation {ob.name}, use_accel = {use_accel}",
                comm=data.comm.comm_group,
            )

            # FIXME: handle coordinate transforms here too...

            pointing_detector(
                fp_quats,
                ob.shared[self.boresight].data,
                quat_indx,
                ob.detdata[self.quats].data,
                ob.intervals[self.view].data,
                flags,
                self.shared_flag_mask,
                impl=implementation,
                use_accel=use_accel,
            )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.boresight],
            "detdata": [self.quats],
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

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        return True
