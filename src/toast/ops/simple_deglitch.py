# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..intervals import IntervalList
from ..mpi import MPI
from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, name_UID
from .operator import Operator


@trait_docs
class SimpleDeglitch(Operator):
    """An operator that flags extreme detector samples.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(defaults.det_data, help="Observation detdata key to analyze")

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
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

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional shared flagging",
    )

    view = Unicode(
        None,
        allow_none=True,
        help="Find glitches in this view",
    )

    glitch_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value to apply at glitch positions",
    )

    glitch_radius = Int(
        5,
        help="Number of additional samples to flag around a glitch",
    )

    rms_limit = Float(
        0.99,
        help="Relative improvement in RMS required to flag a glitch",
    )

    nsample_min = Int(
        100,
        help="Minimum number of good samples in an interval.",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net_factors = []
        self.total_factors = []
        self.weights_in = []
        self.weights_out = []
        self.rates = []

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            if ob.comm_row_size != 1:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)
            views = ob.intervals[self.view]
            focalplane = ob.telescope.focalplane

            local_dets = ob.select_local_detectors(flagmask=defaults.det_mask_invalid)
            shared_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
            for name in local_dets:
                sig = ob.detdata[self.det_data][name]
                det_flags = ob.detdata[self.det_flags][name] & self.det_flag_mask
                bad = np.logical_or(shared_flags != 0, det_flags != 0)
                for iview, view in enumerate(views):
                    nsample = view.last - view.first + 1
                    ind = slice(view.first, view.last + 1)
                    sig_view = sig[ind].copy()
                    sig_view[bad[ind]] = np.nan
                    sig_view -= np.nanmedian(sig_view)
                    rms_ref = np.nanstd(sig_view)
                    while True:
                        if np.isnan(rms_ref) or \
                           np.sum(np.isfinite(sig_view)) < self.nsample_min:
                            # flag the entire view.  Not enough statistics
                            sig_view[:] = np.nan
                            break
                        # See if masking the brightest sample reduces the RMS
                        i = np.nanargmax(np.abs(sig_view))
                        sig_view_test = sig_view.copy()
                        istart = max(0, i - self.glitch_radius)
                        istop = min(nsample, i + self.glitch_radius + 1)
                        sig_view_test[istart : istop] = np.nan
                        rms_test = np.nanstd(sig_view_test)
                        if rms_test > self.rms_limit * rms_ref:
                            # Not enough improvement
                            break
                        sig_view = sig_view_test
                        rms_ref = rms_test
                    bad_view = np.isnan(sig_view)
                    # DEBUG begin
                    # if np.sum(bad_view) < bad_view.size:
                    #     import pdb
                    #     pdb.set_trace()
                    # if name == "pa6_1117":
                    #     import pdb
                    #     import matplotlib.pyplot as plt
                    #     pdb.set_trace()
                    # DEBUG end
                    det_flags[ind][bad_view] |= self.glitch_mask

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": [self.noise_model],
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return prov
