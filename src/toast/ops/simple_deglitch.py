# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import traitlets
from astropy import units as u
from scipy.signal import medfilt

from .. import qarray as qa
from ..intervals import IntervalList
from ..mpi import MPI
from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, flagged_noise_fill, name_UID
from .operator import Operator


@trait_docs
class SimpleDeglitch(Operator):
    """An operator that flags extreme detector samples."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(defaults.det_data, help="Observation detdata key to analyze")

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
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

    reset_det_flags = Bool(
        False,
        help="Replace existing detector flags",
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

    glitch_limit = Float(
        5.0,
        help="Glitch detection threshold in units of RMS",
    )

    nglitch_limit = Int(
        10,
        help="Maximum number of glitches in a view",
    )

    nsample_min = Int(
        100,
        help="Minimum number of good samples in an interval.",
    )

    medfilt_kernel_size = Int(
        101,
        help="Median filter kernel width.  Either 0 (full interval) "
        "or a positive odd number",
    )

    fill_gaps = Bool(
        True,
        help="Fill gaps with a trend line and white noise",
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

    @traitlets.validate("medfilt_kernel_size")
    def _check_medfilt_kernel_size(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("medfilt_kernel_size cannot be negative")
        if check > 0 and check % 2 == 0:
            raise traitlets.TraitError("medfilt_kernel_size cannot be even")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net_factors = []
        self.total_factors = []
        self.weights_in = []
        self.weights_out = []
        self.rates = []

    @function_timer
    def _get_coupled_detectors(self, det, dets):
        """See if other detectors should be flagged symmetrically"""
        coupled = [det]
        if det.startswith("demod0"):
            qdet = det.replace("demod0", "demod4r")
            udet = det.replace("demod0", "demod4i")
            alt_dets = [qdet, udet]
        elif det.startswith("demod4r"):
            idet = det.replace("demod4r", "demod0")
            udet = det.replace("demod4r", "demod4i")
            alt_dets = [idet, udet]
        elif det.startswith("demod4i"):
            idet = det.replace("demod4i", "demod0")
            qdet = det.replace("demod4i", "demod4r")
            alt_dets = [idet, qdet]
        else:
            alt_dets = []
        for alt_det in alt_dets:
            if alt_det in dets:
                coupled.append(alt_det)
        return coupled

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        timer.start()

        nobs = 0
        nbad = 0
        ndet = 0
        obstimer = Timer()
        obstimer.start()
        for ob in data.obs:
            if ob.comm.comm_group is None or ob.comm.comm_group.rank == 0:
                nobs += 1
            if not ob.is_distributed_by_detector:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)
            views = ob.intervals[self.view]
            focalplane = ob.telescope.focalplane

            local_dets = ob.select_local_detectors(flagmask=self.det_mask)
            ndet += len(local_dets)
            shared_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
            if self.reset_det_flags:
                for det in local_dets:
                    ob.detdata[self.det_flags][det][:] = 0
            bad_detectors = set()
            for det in local_dets:
                if det in bad_detectors:
                    # This detector was coupled to a detector with too many glitches
                    continue
                coupled_detectors = self._get_coupled_detectors(det, local_dets)
                sig = ob.detdata[self.det_data][det]
                det_flags = ob.detdata[self.det_flags][det]
                bad = np.logical_or(
                    shared_flags != 0,
                    (det_flags & self.det_flag_mask) != 0,
                )
                for iview, view in enumerate(views):
                    nsample = view.last - view.first
                    ind = slice(view.first, view.last)
                    sig_view = sig[ind].copy()
                    w = self.medfilt_kernel_size
                    if w > 0 and nsample > 2 * w:
                        # Remove the running median
                        sig_view[w:-w] -= medfilt(sig_view, kernel_size=w)[w:-w]
                        # Special treatment for the ends
                        sig_view[:w] -= np.median(sig_view[:w])
                        sig_view[-w:] -= np.median(sig_view[-w:])
                    sig_view[bad[ind]] = np.nan
                    if np.all(np.isnan(sig_view)):
                        continue
                    offset = np.nanmedian(sig_view)
                    sig_view -= offset
                    rms = np.nanstd(sig_view)
                    nglitch = 0
                    while True:
                        if (
                            np.isnan(rms)
                            or np.sum(np.isfinite(sig_view)) < self.nsample_min
                        ):
                            # flag the entire view.  Not enough statistics
                            sig_view[:] = np.nan
                            break
                        # See if the brightest remaining sample still stands out
                        i = np.nanargmax(np.abs(sig_view))
                        sig_view_test = sig_view.copy()
                        istart = max(0, i - self.glitch_radius)
                        istop = min(nsample, i + self.glitch_radius + 1)
                        sig_view_test[istart:istop] = np.nan
                        rms_test = np.nanstd(sig_view_test)
                        if np.abs(sig_view[i]) < self.glitch_limit * rms_test:
                            # Not significant enough
                            break
                        nglitch += 1
                        if nglitch > self.nglitch_limit:
                            sig_view[:] = np.nan
                            break
                        sig_view[:] = sig_view_test
                        rms = rms_test
                    if nglitch == 0:
                        continue
                    bad_view = np.isnan(sig_view)
                    for alt_det in coupled_detectors:
                        # Raise the same flags in all coupled detectors
                        alt_flags = ob.detdata[self.det_flags][alt_det]
                        alt_flags[ind][bad_view] |= self.glitch_mask

                if np.all((det_flags & self.det_flag_mask) != 0):
                    # This detector is a total loss. Raise the detector flag
                    for alt_det in coupled_detectors:
                        ob.local_detector_flags[alt_det] |= defaults.det_mask_invalid
                        bad_detectors.add(alt_det)
                elif self.fill_gaps:
                    # 1 second buffer
                    buffer_ = int(focalplane.sample_rate.to_value(u.Hz))
                    for alt_det in coupled_detectors:
                        flagged_noise_fill(
                            ob.detdata[self.det_data][alt_det],
                            ob.detdata[self.det_flags][alt_det],
                            buffer_,
                            poly_order=1,
                        )
            nbad_obs = len(bad_detectors)
            ndet_obs = len(local_dets)
            nbad += nbad_obs
            if ob.comm.comm_group is not None:
                nbad_obs = ob.comm.comm_group.reduce(nbad_obs)
                ndet_obs = ob.comm.comm_group.reduce(ndet_obs)
            log.debug_rank(
                f"Flagged {nbad_obs} / {ndet_obs} badly glitched detectors "
                f"in {ob.name} in",
                comm=ob.comm.comm_group,
                timer=obstimer,
            )
        if data.comm.comm_world is not None:
            nobs = data.comm.comm_world.reduce(nobs)
            nbad = data.comm.comm_world.reduce(nbad)
            ndet = data.comm.comm_world.reduce(ndet)
        log.info_rank(
            f"Flagged {nbad} / {ndet} badly glitched detectors over {nobs} "
            f"observations in",
            comm=data.comm.comm_world,
            timer=timer,
        )

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
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return prov
