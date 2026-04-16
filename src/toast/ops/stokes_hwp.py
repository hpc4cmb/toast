# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..observation import default_values as defaults
from .. import qarray as qa
from ..timing import function_timer
from ..traits import Bool, Instance, Int, List, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator


def stokes_weights_hwp_model_nominal(
    quat_index,
    quats,
    weight_index,
    weights,
    hwp,
    intervals,
    epsilon,
    gamma,
    cal,
    IAU,
):
    """This implements the math for the "nominal" model.

    In this model, all detectors have the same Mueller matrix coefficients,
    regardless of incidence angle.

    There are 9 non-zero elements in the pointing matrix in this case.

    """
    if IAU:
        U_sign = 1.0
    else:
        U_sign = -1.0
    print('AO gamma =', gamma)
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)
    for idet in range(len(quat_index)):
        qidx = quat_index[idet]
        widx = weight_index[idet]
        eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet])
        for vw in intervals:
            samples = slice(vw.first, vw.last, 1)
            vd = qa.rotate(quats[qidx][samples], zaxis)
            vo = qa.rotate(quats[qidx][samples], xaxis)

            # The vector orthogonal to the line of sight that is parallel
            # to the local meridian.
            dir_ang = np.arctan2(vd[:, 1], vd[:, 0])
            dir_r = np.sqrt(1.0 - vd[:, 2] * vd[:, 2])
            vm_z = -dir_r
            vm_x = vd[:, 2] * np.cos(dir_ang)
            vm_y = vd[:, 2] * np.sin(dir_ang)

            # Compute the rotation angle from the meridian vector to the
            # orientation vector.  The direction vector is normal to the plane
            # containing these two vectors, so the rotation angle is:
            #
            # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
            #
            alpha_y = (
                vd[:, 0] * (vm_y * vo[:, 2] - vm_z * vo[:, 1])
                - vd[:, 1] * (vm_x * vo[:, 2] - vm_z * vo[:, 0])
                + vd[:, 2] * (vm_x * vo[:, 1] - vm_y * vo[:, 0])
            )
            alpha_x = vm_x * vo[:, 0] + vm_y * vo[:, 1] + vm_z * vo[:, 2]

            # This is the final detector alpha angle on the sky
            alpha = np.arctan2(alpha_y, alpha_x)

            # The HWP "omega" angle is alpha + gamma_hwp(t) - gamma_det
            omega = alpha + hwp - gamma[idet]

            # Compute all intermediate trig arrays we need.
            cos2alpha = np.cos(2.0 * alpha)
            sin2alpha = np.sin(2.0 * alpha)
            cos2omega = np.cos(2.0 * omega)
            sin2omega = np.sin(2.0 * omega)
            cos2alpha2omega = np.cos(2.0 * alpha - 2.0 * omega)
            sin2alpha2omega = np.sin(2.0 * alpha - 2.0 * omega)
            cos2alpha4omega = np.cos(2.0 * alpha - 4.0 * omega)
            sin2alpha4omega = np.sin(2.0 * alpha - 4.0 * omega)

            # FIXME:  Ignore the cross-polar response (eta) for now.
            weights[widx][samples, 0] = 1.0
            weights[widx][samples, 1] = sin2alpha
            weights[widx][samples, 2] = cos2alpha
            weights[widx][samples, 3] = sin2omega
            weights[widx][samples, 4] = cos2omega
            weights[widx][samples, 5] = cos2alpha2omega
            weights[widx][samples, 6] = sin2alpha2omega
            weights[widx][samples, 7] = cos2alpha4omega
            weights[widx][samples, 8] = sin2alpha4omega

            # Apply overall calibration
            weights[widx][samples, :] *= cal[idet]


def stokes_weights_hwp_model_mueller(
    quat_index,
    quats,
    weight_index,
    weights,
    hwp,
    intervals,
    epsilon,
    gamma,
    mueller,
    cal,
    IAU,
    include_V=False,
):
    """This implements the math for the "mueller" model.

    In this model, all detectors have the same Mueller matrix coefficients
    and this is specified by the user.

    There are 3 non-zero elements in the pointing matrix unless `include_V`
    is True, in which case the stokes V term is included.

    """
    if IAU:
        U_sign = 1.0
    else:
        U_sign = -1.0

    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)
    for idet in range(len(quat_index)):
        qidx = quat_index[idet]
        widx = weight_index[idet]
        eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet])
        for vw in intervals:
            samples = slice(vw.first, vw.last, 1)
            vd = qa.rotate(quats[qidx][samples], zaxis)
            vo = qa.rotate(quats[qidx][samples], xaxis)

            # The vector orthogonal to the line of sight that is parallel
            # to the local meridian.
            dir_ang = np.arctan2(vd[:, 1], vd[:, 0])
            dir_r = np.sqrt(1.0 - vd[:, 2] * vd[:, 2])
            vm_z = -dir_r
            vm_x = vd[:, 2] * np.cos(dir_ang)
            vm_y = vd[:, 2] * np.sin(dir_ang)

            # Compute the rotation angle from the meridian vector to the
            # orientation vector.  The direction vector is normal to the plane
            # containing these two vectors, so the rotation angle is:
            #
            # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
            #
            alpha_y = (
                vd[:, 0] * (vm_y * vo[:, 2] - vm_z * vo[:, 1])
                - vd[:, 1] * (vm_x * vo[:, 2] - vm_z * vo[:, 0])
                + vd[:, 2] * (vm_x * vo[:, 1] - vm_y * vo[:, 0])
            )
            alpha_x = vm_x * vo[:, 0] + vm_y * vo[:, 1] + vm_z * vo[:, 2]

            # This is the final detector alpha angle on the sky
            alpha = np.arctan2(alpha_y, alpha_x)

            # The HWP "omega" angle is alpha + gamma_hwp(t) - gamma_det
            omega = alpha + hwp - gamma[idet]

            # Compute all intermediate trig arrays we need in multiple places.

            cos2alpha = np.cos(2.0 * alpha)
            sin2alpha = np.sin(2.0 * alpha)
            cos2omega = np.cos(2.0 * omega)
            sin2omega = np.sin(2.0 * omega)
            cos2alpha2omega = np.cos(2.0 * alpha - 2.0 * omega)
            sin2alpha2omega = np.sin(2.0 * alpha - 2.0 * omega)
            cos2alpha4omega = np.cos(2.0 * alpha - 4.0 * omega)
            sin2alpha4omega = np.sin(2.0 * alpha - 4.0 * omega)

            # Assign values of the pointing matrix. (see notebook doc)

            # Stokes I
            weights[widx][samples, 0] = (
                mueller[0, 0]
                + eta  * mueller[1, 0] * cos2alpha2omega
                - eta * mueller[2, 0] * sin2alpha2omega
            )

            # Stokes Q
            weights[widx][samples, 1] = (
                cos2omega * mueller[0, 1]
                + sin2omega * mueller[0, 2]
                + 0.5 * eta * cos2alpha * (mueller[1, 1] + mueller[2, 2])
                + 0.5 * eta * sin2alpha * (mueller[1, 2] - mueller[2, 1])
                + 0.5 * eta * cos2alpha4omega * (mueller[1, 1] - mueller[2, 2])
                - 0.5 * eta * sin2alpha4omega * (mueller[1, 2] + mueller[2, 1])
            )

            # Stokes U
            weights[widx][samples, 2] = (
                cos2omega * mueller[0, 2]
                - sin2omega * mueller[0, 1]
                + 0.5 * eta * cos2alpha * (mueller[1, 2] - mueller[2, 1])
                - 0.5 * eta * sin2alpha * (mueller[1, 1] + mueller[2, 2])
                + 0.5 * eta * cos2alpha4omega * (mueller[1, 2] + mueller[2, 1])
                + 0.5 * eta * sin2alpha4omega * (mueller[1, 1] - mueller[2, 2])
            ) * U_sign

            # Stokes V
            if include_V:
                weights[widx][samples, 3] = (
                    mueller[0, 3]
                    + mueller[1, 3] * eta * cos2alpha2omega
                    - mueller[2, 3] * eta * sin2alpha2omega
                )

            # Apply overall calibration
            weights[widx][samples, :] *= cal[idet]


@trait_docs
class StokesWeightsHWP(Operator):
    """Operator which generates HWP systematics pointing weights.

    Given the individual detector pointing, this computes the pointing weights
    assuming that... (To-Do: add description and references)

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels
    and weights.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    mode = Unicode("nominal", help="The data model to use: 'nominal', 'mueller'")

    mueller = List([], help="In 'mueller' mode, the common Mueller matrix to use")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    hwp_angle = Unicode(defaults.hwp_angle, help="Observation shared key for HWP angle")

    fp_gamma = Unicode("gamma", help="Focalplane key for detector gamma offset angle")

    weights = Unicode(
        defaults.weights, help="Observation detdata key for output weights"
    )

    single_precision = Bool(False, help="If True, use 32bit float in output")

    IAU = Bool(False, help="If True, use the IAU convention rather than COSMO")

    cal = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of pointing weight "
        "calibration for each det",
    )

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "det_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        check = proposal["value"]
        if check not in ["nominal", "mueller"]:
            raise traitlets.TraitError("Invalid mode (must be 'nominal' or 'mueller')")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Compute the number of non-zeros in the pointing matrix
        if self.mode == "nominal":
            self._nnz = 9
        elif self.mode == "mueller":
            self._nnz = 3

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        # Detector pointing quaternions
        quats_name = self.detector_pointing.quats

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        # Expand detector pointing
        self.detector_pointing.apply(data, detectors=detectors)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(
                detectors, flagmask=self.detector_pointing.det_mask
            )
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Check that our view is fully covered by detector pointing.  If the
            # detector_pointing view is None, then it has all samples.  If our own
            # view was None, then it would have been set to the detector_pointing
            # view above.
            if (view is not None) and (self.detector_pointing.view is not None):
                if ob.intervals[view] != ob.intervals[self.detector_pointing.view]:
                    # We need to check intersection
                    intervals = ob.intervals[self.view]
                    detector_intervals = ob.intervals[self.detector_pointing.view]
                    intersection = detector_intervals & intervals
                    if intersection != intervals:
                        msg = (
                            f"view {self.view} is not fully covered by valid "
                            "detector pointing"
                        )
                        raise RuntimeError(msg)

            # Create (or re-use) output data for the weights
            if self.single_precision:
                exists = ob.detdata.ensure(
                    self.weights,
                    sample_shape=(self._nnz,),
                    dtype=np.float32,
                    detectors=dets,
                    accel=use_accel,
                )
            else:
                exists = ob.detdata.ensure(
                    self.weights,
                    sample_shape=(self._nnz,),
                    dtype=np.float64,
                    detectors=dets,
                    accel=use_accel,
                )

            quat_indx = ob.detdata[quats_name].indices(dets)
            weight_indx = ob.detdata[self.weights].indices(dets)

            # Do we already have pointing for all requested detectors?
            if exists:
                # Yes
                if data.comm.group_rank == 0:
                    msg = (
                        f"Group {data.comm.group}, ob {ob.name}, Stokes weights "
                        f"already computed for {dets}"
                    )
                    log.verbose(msg)
                continue

            focalplane = ob.telescope.focalplane
            det_epsilon = np.zeros(len(dets), dtype=np.float64)

            # Get the cross polar response from the focalplane
            if "pol_leakage" in focalplane.detector_data.colnames:
                for idet, d in enumerate(dets):
                    det_epsilon[idet] = focalplane[d]["pol_leakage"]

            # Get the per-detector calibration
            if self.cal is None:
                cal = np.array([1.0 for x in dets], np.float64)
            else:
                cal = np.array([ob[self.cal][x] for x in dets], np.float64)

            det_gamma = np.zeros(len(dets), dtype=np.float64)
            hwp_data = ob.shared[self.hwp_angle].data
            for idet, d in enumerate(dets):
                det_gamma[idet] = focalplane[d]["gamma"].to_value(u.rad)
            weight_data = ob.detdata[self.weights].data

            if self.mode == "nominal":
                stokes_weights_hwp_model_nominal(
                    quat_indx,
                    ob.detdata[quats_name].data,
                    weight_indx,
                    weight_data,
                    hwp_data,
                    ob.intervals[self.view].data,
                    det_epsilon,
                    det_gamma,
                    cal,
                    bool(self.IAU),
                )
            elif self.mode == "mueller":
                # Use a constant (local) Mueller matrix.  Ignore the Stokes V weights
                # for now.
                stokes_weights_hwp_model_mueller(
                    quat_indx,
                    ob.detdata[quats_name].data,
                    weight_indx,
                    weight_data,
                    hwp_data,
                    ob.intervals[self.view].data,
                    det_epsilon,
                    det_gamma,
                    np.array(self.mueller),
                    cal,
                    bool(self.IAU),
                    include_V=False,
                )
            else:
                raise RuntimeError(f"Unexpected mode: {self.mode}")

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        if "detdata" not in req:
            req["detdata"] = list()
        req["detdata"].append(self.weights)
        if self.cal is not None:
            req["meta"].append(self.cal)
        if self.hwp_angle is not None:
            req["shared"].append(self.hwp_angle)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prov["detdata"].append(self.weights)
        return prov

    def _supports_accel(self):
        return False
