# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Environment, Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance

from ..healpix import HealpixPixels

from ..timing import function_timer

from .. import qarray as qa

from ..pixels import PixelDistribution

from ..observation import default_values as defaults

from .._libtoast import stokes_weights, stokes_weights_I, stokes_weights_IQU

from .operator import Operator

from .delete import Delete


@trait_docs
class StokesWeights(Operator):
    """Operator which generates I/Q/U pointing weights.

    Given the individual detector pointing, this computes the pointing weights
    assuming that the detector is a linear polarizer followed by a total
    power measurement.  An optional dictionary of pointing weight calibration factors
    may be specified for each observation.

    For each observation, the cross-polar response for every detector is obtained from
    the Focalplane, and if a HWP angle timestream exists, then a perfect HWP Mueller
    matrix is included in the response.

    The timestream model is then (see Jones, et al, 2006):

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a} + U \\sin{2a}\\right]\\right]

    Or, if a HWP is included in the response with time varying angle "w", then
    the total response is:

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a+4w} + U \\sin{2a+4w}\\right]\\right]

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

    mode = Unicode("I", help="The Stokes weights to generate (I or IQU)")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    hwp_angle = Unicode(
        None, allow_none=True, help="Observation shared key for HWP angle"
    )

    weights = Unicode(
        defaults.weights, help="Observation detdata key for output weights"
    )

    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output quaternions",
    )

    single_precision = Bool(False, help="If True, use 32bit float in output")

    cal = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of pointing weight "
        "calibration for each det",
    )

    use_python = Bool(False, help="If True, use python implementation")

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
        if check not in ["I", "IQU"]:
            raise traitlets.TraitError("Invalid mode (must be 'I' or 'IQU')")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=False, **kwargs):
        env = Environment.get()
        log = Logger.get()

        self._nnz = len(self.mode)

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        if self.use_python and use_accel:
            raise RuntimeError("Cannot use accelerator with pure python implementation")

        # Expand detector pointing
        if self.quats is not None:
            quats_name = self.quats
        else:
            if self.detector_pointing.quats is not None:
                quats_name = self.detector_pointing.quats
            else:
                quats_name = "quats"

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        # Expand detector pointing
        self.detector_pointing.quats = quats_name
        self.detector_pointing.apply(data, detectors=detectors, use_accel=use_accel)

        cal = self.cal
        if cal is None:
            cal = 1.0

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
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

            if use_accel:
                if not ob.detdata.accel_exists(self.weights):
                    ob.detdata.accel_create(self.weights)

            # FIXME:  temporary hack until instrument classes are also pre-staged
            # to GPU
            focalplane = ob.telescope.focalplane
            det_epsilon = np.zeros(len(dets), dtype=np.float64)

            # Get the cross polar response from the focalplane
            if "pol_leakage" in focalplane.detector_data.colnames:
                for idet, d in enumerate(dets):
                    det_epsilon[idet] = focalplane[d]["pol_leakage"]

            if self.use_python:
                hwp_data = None
                if self.hwp_angle is not None:
                    hwp_data = ob.shared[self.hwp_angle].data
                self._py_stokes_weights(
                    quat_indx,
                    ob.detdata[quats_name].data,
                    weight_indx,
                    ob.detdata[self.weights].data,
                    ob.intervals[self.view].data,
                    cal,
                    det_epsilon,
                    hwp_data,
                )
            else:
                if self.mode == "IQU":
                    if self.hwp_angle is None:
                        hwp_data = np.zeros((0,), dtype=np.float64)
                    else:
                        hwp_data = ob.shared[self.hwp_angle].data
                    stokes_weights_IQU(
                        quat_indx,
                        ob.detdata[quats_name].data,
                        weight_indx,
                        ob.detdata[self.weights].data,
                        hwp_data,
                        ob.intervals[self.view].data,
                        det_epsilon,
                        cal,
                        use_accel,
                    )
                else:
                    stokes_weights_I(
                        weight_indx,
                        ob.detdata[self.weights].data,
                        ob.intervals[self.view].data,
                        cal,
                        use_accel,
                    )
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        if self.cal is not None:
            req["meta"].append(self.cal)
        if self.hwp_angle is not None:
            req["shared"].append(self.hwp_angle)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"detdata": [self.weights]}
        return prov

    def _supports_accel(self):
        return self.detector_pointing.supports_accel()

    def _py_stokes_weights(
        self,
        quat_indx,
        quat_data,
        weight_indx,
        weight_data,
        intr_data,
        cal,
        det_epsilon,
        hwp_data,
    ):
        """Internal python implementation for comparison tests."""
        zaxis = np.array([0, 0, 1], dtype=np.float64)
        xaxis = np.array([1, 0, 0], dtype=np.float64)
        if self.mode == "IQU":
            for idet in range(len(quat_indx)):
                qidx = quat_indx[idet]
                widx = weight_indx[idet]
                eta = (1.0 - det_epsilon[idet]) / (1.0 + det_epsilon[idet])
                for vw in intr_data:
                    samples = slice(vw.first, vw.last + 1, 1)
                    dir = qa.rotate(quat_data[qidx][samples], zaxis)
                    orient = qa.rotate(quat_data[qidx][samples], xaxis)
                    ay = np.multiply(orient[:, 0], dir[:, 1]) - np.multiply(
                        orient[:, 1], dir[:, 0]
                    )
                    ax = (
                        np.multiply(
                            orient[:, 0], np.multiply(-1.0 * dir[:, 2], dir[:, 0])
                        )
                        + np.multiply(
                            orient[:, 1], np.multiply(-1.0 * dir[:, 2], dir[:, 1])
                        )
                        + np.multiply(
                            orient[:, 2],
                            np.multiply(dir[:, 0], dir[:, 0])
                            + np.multiply(dir[:, 1], dir[:, 1]),
                        )
                    )
                    ang = np.arctan2(ay, ax)
                    if hwp_data is not None:
                        ang += 2.0 * hwp_data[samples]
                    ang *= 2.0
                    weight_data[widx][samples, 0] = cal
                    weight_data[widx][samples, 1] = cal * eta * np.cos(ang)
                    weight_data[widx][samples, 2] = cal * eta * np.sin(ang)
        else:
            for idet in range(len(quat_indx)):
                widx = weight_indx[idet]
                for vw in intr_data:
                    samples = slice(vw.first, vw.last + 1, 1)
                    weight_data[widx][samples] = cal
