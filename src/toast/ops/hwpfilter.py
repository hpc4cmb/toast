# Copyright (c) 2015-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
from astropy import units as u

from .._libtoast import add_templates, bin_invcov, bin_proj, fourier, legendre
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, Unicode, trait_docs
from ..utils import Environment, Logger, Timer
from .operator import Operator

# Wrappers for more precise timing


@function_timer
def bin_proj_fast(ref, templates, good, proj):
    return bin_proj(ref, templates, good, proj)


@function_timer
def bin_invcov_fast(templates, good, invcov):
    return bin_invcov(templates, good, invcov)


@function_timer
def get_rcond(invcov):
    return 1 / np.linalg.cond(invcov)


@function_timer
def get_inverse(invcov):
    return np.linalg.inv(invcov)


@function_timer
def get_pseudoinverse(invcov):
    return np.linalg.pinv(invcov, rcond=1e-12, hermitian=True)


@function_timer
def lstsq_coeff(invcov, proj):
    cov = np.linalg.inv(invcov)
    return np.linalg.lstsq(invcov, proj, rcond=1e-30)[0]


@trait_docs
class HWPFilter(Operator):
    """Operator that applies HWP-synchronous signal filtering."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid | defaults.det_mask_processing,
        help="Bit mask value for optional detector flagging",
    )

    hwp_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to use when adding flags based on HWP filter failures.",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle, allow_none=True, help="Observation shared key for HWP angle"
    )

    trend_order = Int(
        5, help="Order of a Legendre polynomial to fit along with the HWPSS template."
    )

    filter_order = Int(
        5, help="Order of a Fourier expansion to fit as a function of HWP angle."
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    detrend = Bool(
        False, help="Subtract the fitted trend along with the ground template"
    )

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

    @traitlets.validate("trend_order")
    def _check_trend_order(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Trend order should be a non-negative integer")
        return check

    @traitlets.validate("filter_order")
    def _check_filter_order(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Filter order should be a non-negative integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def build_templates(self, obs):
        """Construct the local HWPSS template hierarchy"""

        # Construct trend templates.  Full domain for x is [-1, 1]

        my_offset = obs.local_index_offset
        my_nsamp = obs.n_local_samples
        nsamp_tot = obs.n_all_samples
        x = np.arange(my_offset, my_offset + my_nsamp) / nsamp_tot * 2 - 1

        legendre_trend = np.zeros([self.trend_order + 1, my_nsamp])
        legendre(x, legendre_trend, 0, self.trend_order + 1)

        # Fourier templates

        hwp_angle = obs.shared[self.hwp_angle].data
        nfilter = 2 * self.filter_order
        fourier_templates = np.zeros([nfilter, my_nsamp])
        fourier(hwp_angle, fourier_templates, 1, self.filter_order + 1)

        templates = np.vstack([legendre_trend, fourier_templates])

        return templates, legendre_trend, fourier_templates

    @function_timer
    def fit_templates(
        self,
        obs,
        det,
        templates,
        ref,
        good,
        last_good,
        last_invcov,
        last_cov,
        last_rcond,
    ):
        log = Logger.get()
        # communicator for processes with the same detectors
        comm = obs.comm_row
        ngood = np.sum(good)
        ntask = 1
        if comm is not None:
            ngood = comm.allreduce(ngood)
            ntask = comm.size
        if ngood == 0:
            return None, None, None, None

        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        proj = np.zeros(ntemplate)

        bin_proj_fast(ref, templates, good.astype(np.uint8), proj)
        if last_good is not None and np.all(good == last_good) and ntask == 1:
            # Flags have not changed, we can re-use the last inverse covariance
            invcov = last_invcov
            cov = last_cov
            rcond = last_rcond
        else:
            bin_invcov_fast(templates, good.astype(np.uint8), invcov)
            if comm is not None:
                # Reduce the binned data.  The detector signal is
                # distributed across the group communicator.
                comm.Allreduce(MPI.IN_PLACE, invcov, op=MPI.SUM)
                comm.Allreduce(MPI.IN_PLACE, proj, op=MPI.SUM)
            rcond = get_rcond(invcov)
            cov = None

        self.rcondsum += rcond
        if rcond > 1e-6:
            self.ngood += 1
            if cov is None:
                cov = get_inverse(invcov)
        else:
            self.nsingular += 1
            log.debug(
                f"HWP template matrix is poorly conditioned, "
                f"rcond = {rcond}, using pseudoinverse."
            )
            if cov is None:
                cov = get_pseudoinverse(invcov)
        coeff = np.dot(cov, proj)

        return coeff, invcov, cov, rcond

    @function_timer
    def subtract_templates(self, ref, good, coeff, legendre_trend, fourier_filter):
        # Trend
        if self.detrend:
            trend = np.zeros_like(ref)
            add_templates(trend, legendre_trend, coeff[: self.trend_order + 1])
            ref -= trend
        # HWP template
        hwptemplate = np.zeros_like(ref)
        add_templates(hwptemplate, fourier_filter, coeff[self.trend_order + 1 :])
        ref -= hwptemplate
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        t0 = time()
        env = Environment.get()
        log = Logger.get()

        wcomm = data.comm.comm_world
        gcomm = data.comm.comm_group

        self.nsingular = 0
        self.ngood = 0
        self.rcondsum = 0

        # Each group loops over its own CES:es
        nobs = len(data.obs)
        for iobs, obs in enumerate(data.obs):
            # Prefix for logging
            log_prefix = f"{data.comm.group} : {obs.name} :"

            if data.comm.group_rank == 0:
                msg = (
                    f"{log_prefix} OpHWPFilter: "
                    f"Processing observation {iobs + 1} / {nobs}"
                )
                log.debug(msg)

            # Cache the output common flags
            if self.shared_flags is not None:
                common_flags = (
                    obs.shared[self.shared_flags].data & self.shared_flag_mask
                )
            else:
                common_flags = np.zeros(obs.n_local_samples, dtype=np.uint8)

            t1 = time()
            templates, legendre_trend, fourier_filter = self.build_templates(obs)
            if data.comm.group_rank == 0:
                msg = (
                    f"{log_prefix} OpHWPFilter: "
                    f"Built templates in {time() - t1:.1f}s"
                )
                log.debug(msg)

            last_good = None
            last_invcov = None
            last_cov = None
            last_rcond = None
            ndet = len(obs.local_detectors)
            for idet, det in enumerate(obs.local_detectors):
                if data.comm.group_rank == 0:
                    msg = (
                        f"{log_prefix} OpHWPFilter: "
                        f"Processing detector # {idet + 1} / {ndet}"
                    )
                    log.verbose(msg)

                ref = obs.detdata[self.det_data][idet]
                if self.det_flags is not None:
                    def_flags = obs.detdata[self.det_flags][idet] & self.det_flag_mask
                    good = np.logical_and(common_flags == 0, def_flags == 0)
                else:
                    good = common_flags == 0

                t1 = time()
                coeff, last_invcov, last_cov, last_rcond = self.fit_templates(
                    obs,
                    det,
                    templates,
                    ref,
                    good,
                    last_good,
                    last_invcov,
                    last_cov,
                    last_rcond,
                )
                last_good = good
                if data.comm.group_rank == 0:
                    msg = (
                        f"{log_prefix} OpHWPFilter: "
                        f"Fit templates in {time() - t1:.1f}s"
                    )
                    log.verbose(msg)

                if coeff is None:
                    continue

                t1 = time()
                self.subtract_templates(
                    ref, good, coeff, legendre_trend, fourier_filter
                )
                if data.comm.group_rank == 0:
                    msg = (
                        f"{log_prefix} OpGroundFilter: "
                        f"Subtract templates in {time() - t1:.1f}s"
                    )
                    log.verbose(msg)
            del last_good
            del last_invcov
            del last_cov
            del last_rcond

        if wcomm is not None:
            self.nsingular = wcomm.allreduce(self.nsingular)
            self.ngood = wcomm.allreduce(self.ngood)
            self.rcondsum = wcomm.allreduce(self.rcondsum)

        if wcomm is None or wcomm.rank == 0:
            rcond_mean = self.rcondsum / (self.nsingular + self.ngood)
            msg = (
                f"Applied ground filter in {time() - t0:.1f} s.  "
                f"Average rcond of template matrix was {rcond_mean}"
            )
            log.debug(msg)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": list(),
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.azimuth is not None:
            req["shared"].append(self.azimuth)
        if self.boresight_azel is not None:
            req["shared"].append(self.boresight_azel)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()
