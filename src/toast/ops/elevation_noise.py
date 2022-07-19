# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..intervals import IntervalList
from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


@trait_docs
class ElevationNoise(Operator):
    """Modify detector noise model based on elevation.
    Optionally include PWV modulation.

    This adjusts the detector PSDs in a noise model based on the median elevation of
    each detector in each observation.

    The PSD value scaled by:

    .. math::
        PSD_{new} = PSD_{old} * (a / sin(el) + c)^2

    NOTE: since this operator generates a new noise model for all detectors, you
    should specify all detectors you intend to use downstream when calling exec().

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels
    and weights.

    If the output model is not specified, then the input is modified in place.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

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
        None,
        allow_none=True,
        help="Use this view of the data in all observations.  "
        "Use 'middle' if the middle 10 seconds of each observation is enough to "
        "determine the effective observing elevation",
    )

    noise_a = Float(
        None,
        allow_none=True,
        help="Parameter 'a' in (a / sin(el) + c).  "
        "If not set, look for one in the Focalplane.",
    )

    noise_c = Float(
        None,
        allow_none=True,
        help="Parameter 'c' in (a / sin(el) + c).  "
        "If not set, look for one in the Focalplane.",
    )

    pwv_a0 = Float(
        None,
        allow_none=True,
        help="Parameter 'a0' in (a0 + pwv * a1 + pwv ** 2 * a2). "
        " If not set, look for one in the Focalplane.",
    )

    pwv_a1 = Float(
        None,
        allow_none=True,
        help="Parameter 'a1' in (a0 + pwv * a1 + pwv ** 2 * a2). "
        " If not set, look for one in the Focalplane.",
    )

    pwv_a2 = Float(
        None,
        allow_none=True,
        help="Parameter 'a2' in (a0 + pwv * a1 + pwv ** 2 * a2). "
        " If not set, look for one in the Focalplane.",
    )

    modulate_pwv = Bool(False, help="If True, modulate the NET based on PWV")

    extra_factor = Float(
        None,
        allow_none=True,
        help="Extra multiplier to the NET scaling",
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

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.detector_pointing is None:
            msg = "You must set the detector_pointing trait before calling exec()"
            log.error(msg)
            raise RuntimeError(msg)

        for obs in data.obs:
            obs_data = data.select(obs_uid=obs.uid)
            focalplane = obs.telescope.focalplane

            if self.view == "middle" and self.view not in obs.intervals:
                # Create a view that is just one minute in the middle
                length = 10.0  # in seconds
                times = obs.shared[self.times]
                t_start = times[0]
                t_stop = times[-1]
                t_middle = np.mean([t_start, t_stop])
                t_start = max(t_start, t_middle - length / 2)
                t_stop = min(t_stop, t_middle + length / 2)
                obs.intervals[self.view] = IntervalList(
                    timestamps=times, timespans=[(t_start, t_stop)]
                )

            # Get the detectors we are using for this observation
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Check that the noise model exists
            if self.noise_model not in obs:
                msg = (
                    "Noise model {self.noise_model} does not exist in "
                    "observation {obs.name}"
                )
                raise RuntimeError(msg)

            # Check that the view in the detector pointing operator covers
            # all the samples needed by this operator

            view = self.view
            detector_pointing_view = self.detector_pointing.view
            if view is None:
                # Use the same data view as detector pointing
                view = self.detector_pointing.view
            elif self.detector_pointing.view is not None:
                # Check that our view is fully covered by detector pointing
                intervals = obs.intervals[self.view]
                detector_intervals = obs.intervals[self.detector_pointing.view]
                intersection = detector_intervals & intervals
                if intersection != intervals:
                    msg = (
                        f"view {self.view} is not fully covered by valid "
                        "detector pointing"
                    )
                    raise RuntimeError(msg)
            self.detector_pointing.view = view

            noise = obs[self.noise_model]

            # Create a new base-class noise object with the same PSDs and
            # mixing matrix as the input.  Then modify those values.  If the
            # output name is the same as the input, then delete the input
            # and replace it with the new model.

            nse_keys = noise.keys
            nse_dets = noise.detectors
            nse_freqs = {x: noise.freq(x) for x in nse_keys}
            nse_psds = {x: noise.psd(x) for x in nse_keys}
            nse_indx = {x: noise.index(x) for x in nse_keys}
            out_noise = Noise(
                detectors=nse_dets,
                freqs=nse_freqs,
                psds=nse_psds,
                indices=nse_indx,
                mixmatrix=noise.mixing_matrix,
            )

            # We are building up a data product (a noise model) which has values for
            # all detectors.  For each detector we need to expand the detector pointing.
            # Since the contributions for all views contribute to the scaling for each
            # detector, we loop over detectors first and then views.

            views = obs.view[view]

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
                    # If there are no valid samples, ignore detector flags and
                    # *hope* that we can still get an approximate elevation
                    if np.all(flags != 0):
                        flags = None
                view_flags.append(flags)

            for det in dets:
                # If both the A and C values are unset, the noise model is not modified.
                if self.noise_a is not None:
                    noise_a = self.noise_a
                    noise_c = self.noise_c
                elif "elevation_noise_a" in focalplane[det].colnames:
                    noise_a = focalplane[det]["elevation_noise_a"]
                    noise_c = focalplane[det]["elevation_noise_c"]
                else:
                    continue

                if self.modulate_pwv and self.pwv_a0 is not None:
                    pwv_a0 = self.pwv_a0
                    pwv_a1 = self.pwv_a1
                    pwv_a2 = self.pwv_a2
                    modulate_pwv = True
                elif self.modulate_pwv and "pwv_noise_a0" in focalplane[det].colnames:
                    pwv_a0 = focalplane[det]["pwv_noise_a0"]
                    pwv_a1 = focalplane[det]["pwv_noise_a1"]
                    pwv_a2 = focalplane[det]["pwv_noise_a2"]
                    modulate_pwv = True
                else:
                    modulate_pwv = False

                # Compute detector quaternions one detector at a time.
                self.detector_pointing.apply(obs_data, detectors=[det])

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

                # Scale the PSD

                net_factor = noise_a / np.sin(el) + noise_c

                if modulate_pwv:
                    pwv = obs.telescope.site.weather.pwv.to_value(u.mm)
                    net_factor *= pwv_a0 + pwv_a1 * pwv + pwv_a2 * pwv**2

                if self.extra_factor is not None:
                    net_factor *= self.extra_factor

                out_noise.psd(det)[:] *= net_factor**2

            self.detector_pointing.view = detector_pointing_view

            # Store the new noise model in the observation.

            if self.out_model is None or self.noise_model == self.out_model:
                # We are replacing the input
                del obs[self.noise_model]
                obs[self.noise_model] = out_noise
            else:
                # We are storing this in a new key
                obs[self.out_model] = out_noise
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
