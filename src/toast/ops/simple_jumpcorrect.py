# Copyright (c) 2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import traitlets
from astropy import units as u
from scipy.signal import fftconvolve

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
    """An operator that identifies and corrects jumps in the data
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

    medfilt_kernel_size = Int(
        101,
        help="Median filter kernel width.  Either 0 (full interval) "
        "or a positive odd number",
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

    def _get_stepfilter(self, m):
        """
        Return the time domain matched filter kernel of length m.
        """
        h = np.zeros(m)
        h[:m // 2] = 1
        h[m // 2:] = -1
        # This turns the interpretation of the peak amplitude directly
        # into the step amplitude
        h /= m // 2
        return h

    def _find_peaks(self, toi, flag, flag_out, lim=3.0, tol=1e4, sigma_in=None):
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
        # Do not accept jumps at the ends due to boundary effects
        lbound = tol
        rbound = tol
        mytoi[:lbound] = np.ma.masked
        mytoi[-rbound:] = np.ma.masked
        if sigma_in is None:
            sigma = self._get_sigma(mytoi, flag_out, tol)
        else:
            sigma = sigma_in

        if np.isnan(sigma) or sigma == 0:
            npeak = 0
        else:
            npeak = np.ma.sum(np.abs(mytoi) > sigma * lim)

        # Only one jump per iteration
        if npeak > 0:
            imax = np.argmax(np.abs(mytoi))
            amplitude = mytoi[imax]
            significance = np.abs(amplitude) / sigma

            # Mask the peak for taking mean and finding additional peaks
            istart = max(0, imax - tol)
            istop = min(len(mytoi), imax + tol)
            # mask out the vicinity not to have false detections near the peak
            mytoi[istart:istop] = np.ma.masked
            flag_out[istart:istop] = True
            if sigma_in is None:
                sigma = self._get_sigma(mytoi, flag_out, tol)

            # Excessive flagging is a sign of false detection
            if significance > 5 or (float(np.sum(flag[istart:istop]))
                                    / (istop - istart) < .5):
                peaks.append((imax, significance, amplitude))

            npeak = np.sum(np.abs(mytoi) > sigma * lim)
        return peaks

    def _get_sigma(self, toi, flag, tol):

        full_flag = np.logical_or(flag, toi == 0)

        sigmas = []
        nn = len(toi)
        for start in range(tol, nn - tol, 2 * tol):
            stop = start + 2 * tol
            if stop > nn - tol:
                break
            ind = slice(start, stop)
            x = toi[ind][full_flag[ind] == 0]
            if len(x) != 0:
                rms = np.sqrt(np.mean(x.data ** 2))
                sigmas.append(rms)

        if len(sigmas) != 0:
            sigma = np.median(sigmas)
        else:
            sigma = 0.
        return sigma

    def _remove_jumps(self, signal, flag, peaks, tol):
        """
        Removes the jumps described by peaks from x.
        Adds a buffer of flags with radius of tol.

        """
        corrected_signal = signal.copy()
        flag_out = flag.copy()
        for peak, _, amplitude in peaks:
            corrected_signal[peak:] -= amplitude
            flag_out[peak - int(tol):peak + int(tol)] = True
        return corrected_signal, flag_out

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        stepfilter = self._get_stepfilter(self.filterlen)

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
                det_flags = ob.detdata[self.det_flags][name]
                if self.reset_det_flags:
                    det_flags[:] = 0
                bad = np.logical_or(
                    shared_flags != 0,
                    (det_flags & self.det_flag_mask) != 0,
                )
                for iview, view in enumerate(views):
                    nsample = view.last - view.first + 1
                    ind = slice(view.first, view.last + 1)
                    sig_view = sig[ind].copy()
                    bad_view = bad[ind]
                    bad_view_out = bad_view.copy()
                    sig_filtered = fftconvolve(
                        sig_view, stepfilter, mode="same"
                    )
                    peaks = self._find_peaks(
                        sig_filtered,
                        bad_view,
                        bad_view_out,
                        lim=self.jump_limit,
                        tol=self.filterlen // 2,
                    )
                    
                    njump = len(peaks)
                    if njump == 0:
                        continue
                    if njump > 10:
                        raise RuntimeError(f"Found {njump} jumps!")

                    corrected_signal, flag_out = self._remove_jumps(
                        sig_view, bad_view, peaks, self.jump_radius)
                    sig[ind] = corrected_signal
                    det_flags[ind][flag_out] |= self.jump_mask

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
