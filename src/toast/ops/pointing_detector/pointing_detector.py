# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import astropy.units as u
import numpy as np
import traitlets

from ... import qarray as qa
from ...accelerator import ImplementationType
from ...observation import default_values as defaults
from ...timing import function_timer
from ...traits import Bool, Int, Quantity, Unicode, UseEnum, trait_docs
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

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    boresight = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight"
    )

    hwp_angle = Unicode(
        defaults.hwp_angle, allow_none=True, help="Observation shared key for HWP angle"
    )

    hwp_angle_offset = Quantity(
        0 * u.deg, help="HWP angle offset to apply when constructing deflection"
    )

    hwp_deflection_radius = Quantity(
        None,
        allow_none=True,
        help="If non-zero, nominal detector pointing will be deflected in a circular "
        "pattern according to HWP phase.",
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
        bore_suffix = ""
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
                    coord_rot = qa.equ2ecl()
                    bore_suffix = "_C2E"
                elif self.coord_out == "G":
                    coord_rot = qa.equ2gal()
                    bore_suffix = "_C2G"
            elif self.coord_in == "E":
                if self.coord_out == "G":
                    coord_rot = qa.ecl2gal()
                    bore_suffix = "_E2G"
                elif self.coord_out == "C":
                    coord_rot = qa.inv(qa.equ2ecl())
                    bore_suffix = "_E2C"
            elif self.coord_in == "G":
                if self.coord_out == "C":
                    coord_rot = qa.inv(qa.equ2gal())
                    bore_suffix = "_G2C"
                if self.coord_out == "E":
                    coord_rot = qa.inv(qa.ecl2gal())
                    bore_suffix = "_G2E"

        # Ensure that we have boresight pointing in the required coordinate
        # frame.  We will potentially re-use this boresight pointing for every
        # iteration of the amplitude solver, so it makes sense to compute and
        # store this.
        bore_name = self.boresight
        if bore_suffix != "":
            bore_name = f"{self.boresight}{bore_suffix}"
            for ob in data.obs:
                if bore_name not in ob.shared:
                    # Does not yet exist, create it
                    ob.shared.create_column(
                        bore_name,
                        ob.shared[self.boresight].shape,
                        ob.shared[self.boresight].dtype,
                    )
                    # First process in each column computes the quaternions
                    bore_rot = None
                    if ob.comm_col_rank == 0:
                        bore_rot = qa.mult(coord_rot, ob.shared[self.boresight].data)
                    ob.shared[bore_name].set(bore_rot, fromrank=0)

        # Ensure that our boresight data is on the right device.  In the case of
        # no coordinate rotation, this would already be done by the outer pipeline.
        for ob in data.obs:
            if use_accel:
                if not ob.shared.accel_in_use(bore_name):
                    # Not currently on the device
                    if not ob.shared.accel_exists(bore_name):
                        # Does not even exist yet on the device
                        ob.shared.accel_create(bore_name)
                    ob.shared.accel_update_device(bore_name)
            else:
                if ob.shared.accel_in_use(bore_name):
                    # Back to host
                    ob.shared.accel_update_host(bore_name)

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

            # Optionally apply HWP deflection.  This is effectively a deflection
            # of the boresight prior to the rotation by the detector quaternion.
            if (
                self.hwp_deflection_radius is not None
                and self.hwp_deflection_radius.value != 0
            ):
                if use_accel:
                    # The data objects are on an accelerator.  Raise an exception
                    # until we can move this code into the kernel.
                    raise NotImplementedError("HWP deflection only works on CPU")
                # Copy node-shared object so that we can modify it.  Starting point
                # is the HWP fast axis.
                deflection_orientation = np.array(ob.shared[self.hwp_angle].data)

                # Apply any phase offset from the fast axis.
                deflection_orientation += self.hwp_angle_offset.to_value(u.rad)

                # The orientation of the deflection is 90 degrees from
                # the axis of rotation going from the boresight to the deflected
                # boresight.
                deflection_orientation += np.pi / 2

                # The rotation axis of the deflection
                deflection_axis = np.zeros(
                    3 * len(deflection_orientation),
                    dtype=np.float64,
                ).reshape((len(deflection_orientation), 3))
                deflection_axis[:, 0] = np.cos(deflection_orientation)
                deflection_axis[:, 1] = np.sin(deflection_orientation)

                # Angle of deflection
                deflection_angle = self.hwp_deflection_radius.to_value(u.radian)

                # Deflection quaternion
                deflection = qa.rotation(
                    deflection_axis,
                    deflection_angle,
                )

                # Apply deflection to the boresight
                boresight = qa.mult(ob.shared[bore_name].data, deflection)
            else:
                boresight = ob.shared[bore_name].data

            pointing_detector(
                fp_quats,
                boresight,
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
