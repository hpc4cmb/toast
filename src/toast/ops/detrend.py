# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
import os

import numpy as np
import traitlets

from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Int, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator

@trait_docs
class Detrend(Operator):
    """Remove mean/median/slope of detector data.

    method='mean'   will subtract mean of each detector data.

    method='median' will subtract median of each detector data.

    method='linear' will subtract mean and slope of each detector data,
    matching `match_nsample` samples at the beginning and end of detector data.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply detrend to",
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
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    detrend_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to use when det_data cannot be detrended.",
    )

    method = Unicode(
        "linear",
        help="Method to detrend. Valid methods are 'linear', 'mean', 'median'",
    )

    edge_nsample = Int(
        100,
        help="When using 'linear' method, number of samples to calculate mean level at the edge",
    )

    edge_nsample_method = Unicode(
        'mean',
        help="When using 'linear' method, method to calculate mean level at the edge. "
        "Valid methods are 'mean' and 'median'",
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("method")
    def _check_method(self, proposal):
        check = proposal["value"]
        allowed = ["mean", "linear", "median"]
        if check not in allowed:
            msg = f"method must be one of {allowed}, not {check}"
            raise traitlets.TraitError(msg)
        return check

    @traitlets.validate("edge_nsample")
    def _check_edge_nsample(self, proposal):
        check = proposal["value"]
        if check < 1:
            raise traitlets.TraitError("edge_nsample should be an integer >= 1")
        return check

    @traitlets.validate("edge_nsample_method")
    def _check_edge_nsample(self, proposal):
        check = proposal["value"]
        allowed = ["mean", "median"]
        if check not in allowed:
            msg = f"method must be one of {allowed}, not {check}"
            raise traitlets.TraitError(msg)
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            local_dets = ob.select_local_detectors(selection=detectors, flagmask=self.det_mask)
            shared_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
            if len(local_dets) == 0:
                # Nothing to do for this observation
                continue
            for det in local_dets:
                sig = ob.detdata[self.det_data][det]
                det_flags = ob.detdata[self.det_flags][det]
                good_samples_flag = np.logical_and(
                    shared_flags == 0,
                    (det_flags & self.det_flag_mask) == 0,
                )
                good_samples_i = np.flatnonzero(good_samples_flag)
                if self.method == "mean":
                    sig -= np.mean(sig[good_samples_flag])
                elif self.method == "median":
                    sig -= np.median(sig[good_samples_flag])
                elif self.method == "linear":
                    start_match_slc = slice(
                            good_samples_i[0],
                            good_samples_i[0]+self.edge_nsample,
                            1)
                    end_match_slc = slice(
                            good_samples_i[-1]+1-self.edge_nsample,
                            good_samples_i[-1]+1,
                            1)
                    if start_match_slc.stop >= end_match_slc.start:
                        ob.local_detector_flags[det] |= self.detrend_flag_mask
                        log.info_rank(f"Not enough good samples, flagged observation {ob.name}")
                        continue
                    if self.edge_nsample_method == 'mean':
                        start_level = np.mean(
                                sig[start_match_slc][good_samples_flag[start_match_slc]]
                                )
                        end_level = np.mean(
                                sig[end_match_slc][good_samples_flag[end_match_slc]]
                                )
                    elif self.edge_nsample_method == 'median':
                        start_level = np.median(
                                sig[start_match_slc][good_samples_flag[start_match_slc]]
                                )
                        end_level = np.median(
                                sig[end_match_slc][good_samples_flag[end_match_slc]]
                                )
                    slope = (end_level - start_level)/(good_samples_i[-1] - good_samples_i[0] + 1.0 - self.edge_nsample)
                    # Remove slope, the slope at the middle point of edge_nsample shoule be zero
                    sig -= (np.arange(sig.size) - good_samples_i[0] - (self.edge_nsample-1.0)/2.0) * slope
                    # This assertion should be True if all the black magic indices counting above are correct
                    #assert np.mean(sig[start_match_slc][good_samples_flag[start_match_slc]]) == start_level
                    sig -= start_level
                else:
                    raise RuntimeError(f"Unknow method={self.method}.")

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()
