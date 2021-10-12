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

from .._libtoast import stokes_weights

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
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        self._nnz = len(self.mode)

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

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
        self.detector_pointing.apply(data, detectors=detectors)

        # We do the calculation over buffers of timestream samples to reduce memory
        # overhead from temporary arrays.
        tod_buffer_length = env.tod_buffer_length()

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

            # Do we already have pointing for all requested detectors?
            if self.weights in ob.detdata:
                wt_dets = ob.detdata[self.weights].detectors
                for d in dets:
                    if d not in wt_dets:
                        break
                else:  # no break
                    # We already have pointing for all specified detectors
                    if data.comm.group_rank == 0:
                        msg = (
                            f"Group {data.comm.group}, ob {ob.name}, pointing "
                            f"already computed for {dets}"
                        )
                        log.verbose(msg)
                    continue

            # Create (or re-use) output data for the weights

            if self.single_precision:
                ob.detdata.ensure(
                    self.weights,
                    sample_shape=(self._nnz,),
                    dtype=np.float32,
                    detectors=dets,
                )
            else:
                ob.detdata.ensure(
                    self.weights,
                    sample_shape=(self._nnz,),
                    dtype=np.float64,
                    detectors=dets,
                )

            # Focalplane for this observation
            focalplane = ob.telescope.focalplane

            # Loop over views
            views = ob.view[view]
            for vw in range(len(views)):
                # Get the flags if needed.  Use the same flags as
                # detector pointing.
                flags = None
                if self.detector_pointing.shared_flags is not None:
                    flags = np.array(
                        views.shared[self.detector_pointing.shared_flags][vw]
                    )
                    flags &= self.detector_pointing.shared_flag_mask

                # HWP angle if needed
                hwp_angle = None
                if self.hwp_angle is not None:
                    hwp_angle = views.shared[self.hwp_angle][vw]

                # Optional calibration
                cal = None
                if self.cal is not None:
                    cal = ob[self.cal]

                for det in dets:
                    props = focalplane[det]

                    # Get the cross polar response from the focalplane
                    if "pol_leakage" in props.colnames:
                        epsilon = props["pol_leakage"]
                    else:
                        epsilon = 0.0

                    # Timestream of detector quaternions
                    quats = views.detdata[quats_name][vw][det]
                    view_samples = len(quats)

                    # Cal for this detector
                    if cal is not None:
                        dcal = cal[det]
                    else:
                        dcal = 1.0

                    # Buffered pointing calculation
                    buf_off = 0
                    buf_n = tod_buffer_length
                    while buf_off < view_samples:
                        if buf_off + buf_n > view_samples:
                            buf_n = view_samples - buf_off

                        bslice = slice(buf_off, buf_off + buf_n)

                        # This buffer of detector quaternions
                        detp = quats[bslice, :].reshape(-1)

                        # Buffer of HWP angle
                        hslice = None
                        if hwp_angle is not None:
                            hslice = hwp_angle[bslice].reshape(-1)

                        # Buffer of flags
                        fslice = None
                        if flags is not None:
                            fslice = flags[bslice].reshape(-1)

                        # Weight buffer
                        wtslice = views.detdata[self.weights][vw][det, bslice].reshape(
                            -1
                        )

                        if self.single_precision:
                            wbuf = np.zeros(len(wtslice), dtype=np.float64)
                        else:
                            wbuf = wtslice

                        stokes_weights(
                            epsilon,
                            dcal,
                            self.mode,
                            detp,
                            hslice,
                            fslice,
                            wbuf,
                        )

                        if self.single_precision:
                            wtslice[:] = wbuf.astype(np.float32)

                        buf_off += buf_n

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
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.weights],
        }
        if self.quats is not None:
            prov["detdata"].append(self.quats)
        return prov

    def _accelerators(self):
        return list()
