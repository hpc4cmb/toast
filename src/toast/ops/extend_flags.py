# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Int, Quantity, Unicode, Bool, trait_docs
from ..utils import Logger, extend_flags
from .operator import Operator


@trait_docs
class ExtendFlags(Operator):
    """Operator that expands flagged regions.

    This operator takes a buffer size (in samples or time) and examines a particular
    bit value of the shared and/or detector flags.  Any flagged samples will be
    expanded by the buffer amount on either side.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(0, help="Bit mask value for detector sample flagging")

    shared_buffer_time = Quantity(
        None,
        allow_none=True,
        help="Flag shared gaps smaller than this time span",
    )

    shared_buffer_samples = Int(
        None,
        allow_none=True,
        help="Flag shared gaps smaller than this number of samples",
    )

    det_buffer_time = Quantity(
        None,
        allow_none=True,
        help="Flag detector gaps smaller than this time span",
    )

    det_buffer_samples = Int(
        None,
        allow_none=True,
        help="Flag detector gaps smaller than this number of samples",
    )

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
        log = Logger.get()

        if self.shared_flag_mask != 0:
            if (
                self.shared_buffer_samples is not None
                and self.shared_buffer_time is not None
            ):
                msg = "Cannot specify both shared_buffer_samples and shared_buffer_time"
                raise RuntimeError(msg)

            if self.shared_buffer_samples is None and self.shared_buffer_time is None:
                msg = "Must specify one of shared_buffer_samples or shared_buffer_time"
                raise RuntimeError(msg)

        if self.det_flag_mask != 0:
            if self.det_buffer_samples is not None and self.det_buffer_time is not None:
                msg = "Cannot specify both det_buffer_samples and det_buffer_time"
                raise RuntimeError(msg)

            if self.det_buffer_samples is None and self.det_buffer_time is None:
                msg = "Must specify one of det_buffer_samples or det_buffer_time"
                raise RuntimeError(msg)

        if self.det_flag_mask == 0 and self.shared_flag_mask == 0:
            msg = "det_flag_mask and shared_flag_mask are both zero- nothing to do."
            log.warning_rank(msg, comm=data.comm.comm_world)
            return

        for ob in data.obs:
            # Sample rate for this observation
            rate = ob.telescope.focalplane.sample_rate.to_value(u.Hz)

            # The buffer size in samples
            if self.shared_buffer_time is not None:
                shared_buf_samp = int(self.shared_buffer_time.to_value(u.second) * rate)
            else:
                shared_buf_samp = self.shared_buffer_samples
            if self.det_buffer_time is not None:
                det_buf_samp = int(self.det_buffer_time.to_value(u.second) * rate)
            else:
                det_buf_samp = self.det_buffer_samples

            if self.shared_flag_mask != 0:
                # We are updating shared flags.  Do this on the first process row and
                # then collectively set the new values.
                if ob.comm_col_rank == 0:
                    new_flags = np.copy(ob.shared[self.shared_flags].data)
                    extend_flags(
                        new_flags,
                        self.shared_flag_mask,
                        shared_buf_samp,
                    )
                else:
                    new_flags = None
                ob.shared[self.shared_flags].set(new_flags)
            if self.det_flag_mask != 0:
                # We are updating detector flags.  Each process works with its local
                # data.
                for det in ob.select_local_detectors(flagmask=self.det_mask):
                    extend_flags(
                        ob.detdata[self.det_flags][det],
                        self.det_flag_mask,
                        det_buf_samp,
                    )
                    if (
                        np.count_nonzero(ob.detdata[self.det_flags][det])
                        == ob.n_local_samples
                    ):
                        msg = f"All samples for detector {det} have been flagged"
                        log.warning(msg)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
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
