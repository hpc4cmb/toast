# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import traitlets
from astropy import units as u

from .. import rng as rng
from ..mpi import MPI
from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Int, List, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, name_UID
from .operator import Operator


@trait_docs
class CommonModeNoise(Operator):
    """Modify noise model to include common modes

    If the output model is not specified, then the input is modified in place.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    realization = Int(0, help="The noise component index")

    component = Int(0, help="The noise component index")

    noise_model = Unicode(
        "noise_model", help="The observation key containing the input noise model"
    )

    out_model = Unicode(
        None, allow_none=True, help="Create a new noise model with this name"
    )

    focalplane_key = Unicode(
        None,
        allow_none=True,
        help="Detectors sharing the focalplane key will have the same common mode",
    )

    detset = List(
        [],
        help="List of detectors to add the common mode to.  "
        "Only used if `focalplane_key` is None",
    )

    fmin = Quantity(
        None,
        allow_none=True,
        help="",
    )

    fknee = Quantity(
        None,
        allow_none=True,
        help="",
    )

    alpha = Float(
        None,
        allow_none=True,
        help="",
    )

    NET = Quantity(
        None,
        allow_none=True,
        help="",
    )

    coupling_strength_center = Float(
        1,
        help="Mean coupling strength between the detectors and the common mode",
    )

    coupling_strength_width = Float(
        0,
        help="Width of the coupling strength distribution "
        "about `coupling_strength_center`",
    )

    static_coupling = Bool(
        False,
        help="If True, coupling to the common mode is not randomized over "
        "observations and realizations",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in ("fmin", "fknee", "alpha", "NET"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if detectors is not None:
            msg = "You must run this operator on all detectors at once"
            log.error(msg)
            raise RuntimeError(msg)

        for obs in data.obs:
            if obs.comm_row_size != 1:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)
            focalplane = obs.telescope.focalplane
            fsample = focalplane.sample_rate.to_value(u.Hz)

            # Check that the noise model exists
            if self.noise_model not in obs:
                msg = f"Noise model {self.noise_model} does not exist in "
                msg += f"observation {obs.name}"
                raise RuntimeError(msg)

            noise = obs[self.noise_model]
            # The noise simulation tools require frequencies to agree
            freqs = noise.freq(noise.keys[0]).to_value(u.Hz)

            # Find the unique values of focalplane keys

            dets_by_key = {}
            if self.focalplane_key is None:
                dets_by_key[None] = []
                for det in obs.all_detectors:
                    if len(self.detset) != 0 and det not in self.detset:
                        continue
                    dets_by_key[None].append(det)
            else:
                if self.focalplane_key not in focalplane.detector_data.colnames:
                    msg = f"Focalplane does not have column for '{self.focalplane_key}'.  "
                    msg += f"Available columns are {focalplane.detector_data.colnames}"
                    raise RuntimeError(msg)
                for det in obs.all_detectors:
                    key = focalplane[det][self.focalplane_key]
                    if key not in dets_by_key:
                        dets_by_key[key] = []
                    dets_by_key[key].append(det)

            # Create a new base-class noise object with the same PSDs and
            # mixing matrix as the input.  Then modify those values.  If the
            # output name is the same as the input, then delete the input
            # and replace it with the new model.

            nse_keys = noise.keys
            nse_dets = noise.detectors
            nse_freqs = {x: noise.freq(x) for x in nse_keys}
            nse_psds = {x: noise.psd(x) for x in nse_keys}
            nse_indx = {x: noise.index(x) for x in nse_keys}
            mixing_matrix = noise.mixing_matrix

            # Add the common mode noise PSDs

            fmin = self.fmin.to_value(u.Hz)
            fknee = self.fknee.to_value(u.Hz)
            alpha = self.alpha
            net = self.NET

            if self.static_coupling:
                obs_id = 0
                realization = 0
            else:
                obs_id = obs.uid
                realization = self.realization

            for key, dets in dets_by_key.items():
                if key is None:
                    noise_key = f"{self.name}_{self.component}"
                else:
                    noise_key = f"{self.name}_{self.component}_{key}"
                mixing_matrix[noise_key] = {}
                noise_uid = name_UID(noise_key)
                nse_keys.append(noise_key)
                nse_freqs[noise_key] = freqs * u.Hz
                nse_psds[noise_key] = net**2 * (
                    (freqs**alpha + fknee**alpha) / (freqs**alpha + fmin**alpha)
                )
                nse_indx[noise_key] = noise_uid

                # Draw coupling strengths and record them in the mixing matrix
                for det in dets:
                    key1 = noise_uid + obs.telescope.uid * 3956215
                    key2 = obs_id
                    counter1 = realization
                    counter2 = focalplane[det]["uid"]
                    gaussian = rng.random(
                        1,
                        sampler="gaussian",
                        key=(key1, key2),
                        counter=(counter1, counter2),
                    )[0]
                    coupling = (
                        self.coupling_strength_center
                        + gaussian * self.coupling_strength_width
                    )
                    mixing_matrix[det][noise_key] = coupling

            out_noise = Noise(
                detectors=nse_dets,
                freqs=nse_freqs,
                psds=nse_psds,
                indices=nse_indx,
                mixmatrix=mixing_matrix,
            )

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
        return

    def _requires(self):
        req = {"meta": self.noise_model}
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
