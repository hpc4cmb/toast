# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np

from astropy import units as u

import traitlets

from ..utils import Environment, Logger

from ..timing import function_timer, Timer

from ..noise_sim import AnalyticNoise

from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity

from .. import qarray as qa

from .operator import Operator


@trait_docs
class ElevationNoise(Operator):
    """Modify detector noise model based on elevation.

    This adjusts the detector PSDs in a noise model based on the median elevation of
    each detector in each observation.

    The new NET value is given by:

    .. math::
        NET_{new} = (a / sin(el) + b)

    The Full PSD is then scaled by

    .. math::
        PSD *= NET_{new}^2 / NET_{old}^2

    Where the old NET is estimated from the high frequency samples of the PSD.

    NOTE: since this operator generates a new noise model for all detectors, you
    should specify all detectors you intend to use downstream when calling exec().

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels
    and weights.

    If the output model is not specified, then the input is modified in place.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode(
        "noise_model", help="The observation key containing the input noise model"
    )

    out_model = Unicode(
        None, allow_none=True, help="Create a new noise model with this name"
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az / El pointing into detector frame",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    noise_a = Quantity(
        0.0 * u.K * np.sqrt(1 * u.second),
        help="Parameter 'a' in (a / sin(el) + b)",
    )

    noise_b = Float(
        0.0 * u.K * np.sqrt(1 * u.second),
        help="Parameter 'b' in (a / sin(el) + b)",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.detector_pointing is None:
            msg = "You must set the detector_pointing trait before calling exec()"
            log.error(msg)
            raise RuntimeError(msg)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Check that the noise model exists
            if self.noise_model not in ob:
                msg = "Noise model {} does not exist in observation {}".format(
                    self.noise_model, ob.name
                )
                raise RuntimeError(msg)

            # If both the A and B values are unset, the noise model is not modified.
            if self.noise_a.value == 0 and self.noise_b.value == 0:
                ob[self.out_model] = ob[self.noise_model]
                continue

            # Check that the view in the detector pointing operator covers
            # all the samples needed by this operator

            view = self.view
            if view is None:
                # Use the same data view as detector pointing
                view = self.detector_pointing.view
            elif self.detector_pointing.view is not None:
                # Check that our view is fully covered by detector pointing
                intervals = ob.intervals[self.view]
                detector_intervals = ob.intervals[self.detector_pointing.view]
                intersection = detector_intervals & intervals
                if intersection != intervals:
                    msg = "view {} is not fully covered by valid detector pointing".format(
                        self.view
                    )
                    raise RuntimeError(msg)

            noise = ob[self.noise_model]

            noise_dets = list(noise.detectors)
            noise_keys = list(noise.keys)

            out_noise = None
            if self.out_model is None:
                out_noise = noise
            else:
                ob[self.out_model] = copy.deepcopy(noise)
                out_noise = ob[self.out_model]

            # Focalplane for this observation
            focalplane = ob.telescope.focalplane

            # We are building up a data product (a noise model) which has values for
            # all detectors.  For each detector we need to expand the detector pointing.
            # Since the contributions for all views contribute to the scaling for each
            # detector, we loop over detectors first and then views.

            views = ob.view[view]

            # The flags are common to all detectors, so we compute them once.

            view_flags = list()
            for vw in range(len(views)):
                # Get the flags if needed.  Use the same flags as detector pointing.
                flags = None
                if self.detector_pointing.shared_flags is not None:
                    flags = np.array(
                        views.shared[self.detector_pointing.shared_flags][vw]
                    )
                    flags &= self.detector_pointing.shared_flag_mask
                view_flags.append(flags)

            for det in dets:
                # Compute detector quaternions one detector at a time.
                self.detector_pointing.apply(data, detectors=[det])
                el_view = list()
                for vw in range(len(views)):
                    # Detector elevation
                    theta, _ = qa.to_position(
                        views.detdata[self.detector_pointing.quats][vw][det]
                    )

                    # Apply flags and convert to elevation
                    if view_flags[vw] is None:
                        el_view.append(np.pi / 2 - theta)
                    else:
                        el_view.append(np.pi / 2 - theta[view_flags[vw] == 0])

                el = np.median(np.concatenate(el_view))

                # PSD
                psd_ksq = noise.psd(det).to_value(u.K ** 2 * u.second)
                rate = noise.rate(det)
                old_net_sq = np.median(psd_ksq[-10:])
                new_net = self.noise_a.to_value(u.K * (1.0 * u.second)) / np.sin(
                    el
                ) + self.noise_b.to_value(u.K * (1.0 * u.second))
                psd[:] *= new_net ** 2 / old_net_sq
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req["meta"].append(self.noise_model)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        if self.out_model is None:
            prov["meta"].append(self.out_model)
        return prov

    def _accelerators(self):
        return list()
