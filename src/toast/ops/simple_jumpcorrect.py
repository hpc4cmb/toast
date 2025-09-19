# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import traitlets
from astropy import units as u
from scipy.signal import convolve

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
class SimpleJumpCorrect(Operator):
    """An operator that identifies and corrects jumps in the data"""

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

    phase = Unicode(
        defaults.azimuth,
        allow_none=True,
        help="Observation shared key for scan phase (to reject scan-synchronous jumps)",
    )

    phase_tol = Float(
        np.radians(1.0),
        help="When `phase` is not None, jumps closer than `phase_tol` are "
        "synchronous and not corrected",
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
        help="Find jumps in this view",
    )

    jump_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value to apply at glitch positions",
    )

    jump_radius = Int(
        5,
        help="Number of additional samples to flag around a jump",
    )

    jump_limit = Float(
        5.0,
        help="Jump detection threshold in units of RMS",
    )

    filterlen = Int(
        100,
        help="Matched filter length",
    )

    nsample_min = Int(
        100,
        help="Minimum number of good samples in an interval",
    )

    njump_limit = Int(
        10,
        help="If the detector has more than `njump_limit` jumps the detector "
        "the detector and time stream will be flagged as invalid.",
    )

    save_jumps = Unicode(
        None,
        allow_none=True,
        help="Save the jump corrections to a dictionary of values per observation",
    )

    apply_jumps = Unicode(
        None,
        allow_none=True,
        help="Do not compute jumps, instead apply the specified dictionary of values",
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

    @traitlets.validate("njump_limit")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check <= 0:
            raise traitlets.TraitError("njump limit should be a positive integer")
        return check

    @traitlets.validate("phase_tol")
    def _check_phase_tol(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("phase_tol should not be negative")
        return check

    def __init__(self, **kwargs):
        self.stepfilter = self._get_stepfilter(self.filterlen)
        super().__init__(**kwargs)

    @function_timer
    def _get_stepfilter(self, m):
        """
        Return the time domain matched filter kernel of length m.
        """
        h = np.zeros(m)
        mid = m // 2
        h[: mid] = 1
        h[mid:] = -1
        # This turns the interpretation of the peak amplitude directly
        # into the step amplitude
        h /= m // 2
        return h

    @function_timer
    def _find_peaks(self, toi, flag, lim=3.0, tol=1e4, sigma_in=None):
        """
        Find the peaks and their amplitudes in the match-filtered TOI.
        Inputs:
        lim -- threshold for jump detection in units of filtered TOI RMS.
        tol -- radius of a region to mask from further peak finding upon
            detecting a peak.
        sigma_in -- an estimate of filtered TOI RMS that overrides the
             sample variance otherwise used.

        """
        peaks = []
        mytoi = np.ma.masked_array(toi)
        myflag = flag.copy()
        nsample = len(mytoi)
        # Do not accept jumps at the ends due to boundary effects
        lbound = tol
        rbound = tol
        mytoi[:lbound] = np.ma.masked
        mytoi[-rbound:] = np.ma.masked
        if sigma_in is None:
            sigma = self._get_sigma(mytoi, myflag, tol)
        else:
            sigma = sigma_in

        if np.isnan(sigma) or sigma == 0:
            npeak = 0
        else:
            npeak = np.ma.sum(np.abs(mytoi) > sigma * lim)

        # Only one jump per iteration
        # And skip remaining if find more than `njump_limit` jumps
        while (npeak > 0) and (len(peaks) <= self.njump_limit):
            imax = np.argmax(np.abs(mytoi))
            amplitude = mytoi[imax]
            significance = np.abs(amplitude) / sigma

            # mask out the vicinity not to have false detections near the peak
            istart = max(0, imax - tol)
            istop = min(nsample, imax + tol)
            mytoi[istart:istop] = np.ma.masked
            myflag[istart:istop] = True
            # Excessive flagging is a sign of false detection
            if significance > 5 or (
                float(np.sum(myflag[istart:istop])) / (istop - istart) < 0.5
            ):
                peaks.append((imax, significance, amplitude))

            # Find additional peaks
            if sigma_in is None:
                sigma = self._get_sigma(mytoi, myflag, tol)
            if np.isnan(sigma) or sigma == 0:
                npeak = 0
            else:
                npeak = np.ma.sum(np.abs(mytoi) > sigma * lim)

        return peaks

    @function_timer
    def _get_sigma(self, toi, flag, tol):
        """ Measure the median flagged standard deviation of toi over
        windows of size 2 * tol
        """
        full_flag = np.logical_or(flag, toi == 0)

        sigmas = []
        nn = len(toi)
        # Ignore tol samples at the edge
        for start in range(tol, nn - 3 * tol + 1, 2 * tol):
            stop = start + 2 * tol
            ind = slice(start, stop)
            x = toi[ind][full_flag[ind] == 0]
            if len(x) != 0:
                rms = np.sqrt(np.mean(x.data**2))
                sigmas.append(rms)

        if len(sigmas) != 0:
            sigma = np.median(sigmas)
        else:
            sigma = 0.0
        return sigma

    @function_timer
    def _remove_jumps(self, signal, flag, jumps):
        """
        Removes the jumps described by peaks from x.
        Adds a buffer of flags with radius of tol.

        """
        corrected_signal = signal.copy()
        nsample = len(signal)
        flag_out = flag.copy()
        for pos, _, amplitude in jumps:
            pstart = max(0, pos - self.jump_radius)
            pstop = min(nsample, pos + self.jump_radius + 1)
            flag_out[pstart:pstop] = True
            # The filter-based amplitude gets biased if there is any
            # ringing around the jump
            ind = slice(pos - self.filterlen // 2, pos)
            before = np.mean(signal[ind][flag_out[ind] == False])
            ind = slice(pos, pos + self.filterlen // 2)
            after = np.mean(signal[ind][flag_out[ind] == False])
            amplitude = after - before
            corrected_signal[pos:] -= amplitude
        return corrected_signal, flag_out

    @function_timer
    def _find_jumps(self, sig, bad, jumps=None, phase=None, offset=0):
        """ Locate all jumps in `sig` using a matched filter
        """
        bad_out = bad.copy()
        # Potential jumps show up as peaks in the match-filtered signal
        sig_filtered = convolve(sig, self.stepfilter, mode="same")
        peaks = self._find_peaks(
            sig_filtered,
            bad,
            lim=self.jump_limit,
            tol=self.filterlen // 2,
        )
        njump = len(peaks)
        # Strong scan-syncronous signal can cause false detections
        while njump > 0 and phase is not None:
            peak_phase = np.array([phase[p[0]] for p in peaks])
            med = np.sort(peak_phase)[njump // 2]
            sync = np.abs(peak_phase - med) < self.phase_tol
            nsync = np.sum(sync)
            if nsync == 1:
                break
            new_peaks = []
            for p, good in zip(peaks, np.logical_not(sync)):
                if good:
                    new_peaks.append(p)
            peaks = new_peaks
            njump = len(peaks)
        if jumps is not None:
            jumps.extend(
                [(x + offset, y, z) for x, y, z in peaks]
            )
        return peaks

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.save_jumps is not None and self.apply_jumps is not None:
            msg = "Cannot both save to and apply pre-existing jumps"
            raise RuntimeError(msg)

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)
            views = ob.intervals[self.view]
            focalplane = ob.telescope.focalplane

            local_dets = ob.select_local_detectors(flagmask=self.det_mask)
            shared_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask
            if self.phase is None:
                phase = ob.shared[self.phase].data
                phase = np.unwrap(phase)
            else:
                phase = None
            if self.save_jumps is not None:
                jump_props = dict()
            for det in local_dets:
                if self.save_jumps is None:
                    jumps = None
                else:
                    jumps = list()
                sig = ob.detdata[self.det_data][det]
                det_flags = ob.detdata[self.det_flags][det]
                if self.reset_det_flags:
                    det_flags[:] = 0
                bad = np.logical_or(
                    shared_flags != 0,
                    (det_flags & self.det_flag_mask) != 0,
                )
                if self.apply_jumps is not None:
                    corrected_signal, flag_out = self._remove_jumps(
                        sig, bad, ob[self.apply_jumps][det], self.jump_radius
                    )
                    sig[:] = corrected_signal
                    det_flags[flag_out] |= self.jump_mask
                else:
                    for iview, view in enumerate(views):
                        ind = slice(view.first, view.last)
                        jumps = self._find_jumps(
                            sig[ind],
                            bad[ind],
                            jumps=jumps,
                            phase=phase,
                            offset=view.first,
                        )
                        if len(jumps) == 0:
                            continue
                        if len(jumps) > self.njump_limit:
                            # This detector view has too many jumps
                            det_flags[ind] |= self.det_flag_mask
                            continue
                        corrected_signal, flag_out = self._remove_jumps(
                            sig[ind], bad[ind], jumps,
                        )
                        sig[ind] = corrected_signal
                        det_flags[ind][flag_out] |= self.jump_mask
                    if self.save_jumps is not None:
                        jump_props[det] = jumps
            if self.save_jumps is not None:
                ob[self.save_jumps] = jump_props
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
