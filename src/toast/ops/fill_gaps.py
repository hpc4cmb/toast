# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
import os

import numpy as np
import traitlets
from astropy import units as u

from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, flagged_noise_fill
from .operator import Operator


@trait_docs
class FillGaps(Operator):
    """Operator that fills flagged samples with noise.

    Currently this operator just fills flagged samples with a simple polynomial
    plus white noise.  It is mostly used for visualization.  No attempt is made
    yet to fill the gaps with a constrained noise realization.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key",
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

    buffer = Quantity(
        1.0 * u.s,
        help="Buffer of time on either side of each gap",
    )

    poly_order = Int(
        1,
        help="Order of the polynomial to fit across each gap",
    )

    @traitlets.validate("poly_order")
    def _check_poly_order(self, proposal):
        check = proposal["value"]
        if check <= 0:
            raise traitlets.TraitError("poly_order should be >= 1")
        return check

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        for ob in data.obs:
            timer = Timer()
            timer.start()

            # Sample rate for this observation
            rate = ob.telescope.focalplane.sample_rate.to_value(u.Hz)

            # The buffer size in samples
            buf_samp = int(self.buffer.to_value(u.second) * rate)

            # Check that parameters make sense
            if self.poly_order > buf_samp + 1:
                msg = f"Cannot fit an order {self.poly_order} polynomial "
                msg += f"to {buf_samp} samples"
                raise RuntimeError(msg)

            if buf_samp > ob.n_local_samples // 4:
                msg = f"Using {buf_samp} samples of buffer around gaps is"
                msg += f" not reasonable for an observation with {ob.n_local_samples}"
                msg += " local samples"
                raise RuntimeError(msg)

            # Local detectors we are considering
            local_dets = ob.select_local_detectors(flagmask=self.det_mask)
            n_dets = len(local_dets)

            # The shared flags
            if self.shared_flags is None:
                shared_flags = np.zeros(ob.n_local_samples, dtype=bool)
            else:
                shared_flags = (
                    ob.shared[self.shared_flags].data & self.shared_flag_mask
                ) != 0

            for idet, det in enumerate(local_dets):
                if self.det_flags is None:
                    flags = shared_flags
                else:
                    flags = np.logical_or(
                        shared_flags,
                        (ob.detdata[self.det_flags][det, :] & self.det_flag_mask) != 0,
                    )
                flagged_noise_fill(
                    ob.detdata[self.det_data][det],
                    flags,
                    buf_samp,
                    poly_order=self.poly_order,
                )
            msg = f"FillGaps {ob.name}: completed in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # Note that the hwp_angle is not strictly required- this
        # is just a no-op.
        req = {
            "shared": [self.times],
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {
            "meta": [],
            "detdata": [self.det_data],
        }
        return prov
