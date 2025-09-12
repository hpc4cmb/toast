# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
import os

import numpy as np
from astropy import units as u

from ..fft import convolve
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Int, Unicode, Quantity, trait_docs
from .noise_model import estimate_net
from .operator import Operator


@trait_docs
class NoiseFilter(Operator):
    """Apply the time domain inverse noise filter (N_tt`^-1) to timestreams.

    This operator uses a specified noise model and computes the inverse time domain
    noise covariance (1 / PSD^2).  It then applies this to the detector timestreams.

    Historically this sort of operation was used to transform data to a basis where
    the noise is white for purposes of solving a least squares problem (e.g. the
    classic GLS mapmaking equation).  However, in the case of using a series of
    timestream filters and making a binned map, this operation might be useful as
    an additional filter in the stack.

    NOTE:  This assumes that flagged samples have already been filled with a
    constrained noise realization (or something similar).  See the FillGaps operator.

    """

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional shared flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for detector sample flagging",
    )

    noise_model = Unicode(
        defaults.noise_model, help="Observation key containing the noise model"
    )

    white_noise_min = Quantity(
        None,
        allow_none=True,
        help="The minimum frequency to consider for the white noise plateau",
    )

    white_noise_max = Quantity(
        None,
        allow_none=True,
        help="The maximum frequency to consider for the white noise plateau",
    )

    debug = Unicode(
        None,
        allow_none=True,
        help="Path to directory for generating debug plots",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if self.white_noise_max is not None:
            # Ensure that the min is also set
            if self.white_noise_min is None:
                msg = "You must set both of the min / max values or neither of them"
                raise RuntimeError(msg)

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            if len(dets) == 0:
                continue

            # Sample rate for this observation
            rate = obs.telescope.focalplane.sample_rate.to_value(u.Hz)

            # The signal array: this is a list of detector array references.
            signal = obs.detdata[self.det_data][dets, :]

            # Construct the N_tt'^-1 kernels for these detectors
            kernels = list()
            kern_freq = None
            psd = None
            for d in dets:
                nse = obs[self.noise_model]
                freq = nse.freq(d)
                if kern_freq is None:
                    kern_freq = freq
                    psd = np.zeros(len(kern_freq), dtype=np.float64)
                else:
                    if not np.allclose(kern_freq, freq):
                        msg = "All detectors in the noise model must have the same"
                        msg += " frequency binning"
                        raise RuntimeError(msg)
                psd[:] = nse.psd(d).value

                # Estimate the NET, so that we can properly treat the 1/f in the
                # kernel.
                if self.white_noise_max is None:
                    net = estimate_net(freq.value, psd)
                else:
                    plateau_samples = np.logical_and(
                        (freq > self.white_noise_min.to(u.Hz)),
                        (freq < self.white_noise_max.to(u.Hz)),
                    )
                    net = np.sqrt(np.mean(psd[plateau_samples]))

                # The white noise value sets the scale things we care about.  If
                # other frequencies have been filtered out, make sure that these
                # components do not blow up in the kernel.  Also normalize the
                # kernel so that the white noise levels are not impacted.
                net_sq = net**2
                psd_limit = 1.0e-3 * net_sq
                cut = psd < psd_limit
                psd[cut] = psd_limit
                psd[:] = 1 / psd
                norm = net_sq
                psd *= norm
                psd[0] = 0
                kernels.append(psd)
            kernels = np.array(kernels)

            # Convolve this inverse noise covariance with the detector timestreams
            debug_root = None
            if self.debug is not None:
                debug_root = os.path.join(
                    self.debug, f"noise_filter_{obs.name}_{dets[0]}-{dets[-1]}"
                )
            convolve(
                signal,
                rate,
                kernel_freq=kern_freq,
                kernels=kernels,
                algorithm="numpy",
                debug=debug_root,
            )

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.noise_model],
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()
