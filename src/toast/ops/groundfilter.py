# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
import re
from astropy import units as u

from .. import qarray as qa
from .._libtoast import add_templates, bin_invcov, bin_proj, legendre_templates
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator

# Wrappers for more precise timing


@function_timer
def bin_proj_fast(ref, templates, good, proj):
    return bin_proj(ref.astype(np.float64), templates, good, proj)


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
class GroundFilter(Operator):
    """Operator that applies ground template filtering to azimuthal scans."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key",
    )

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are filtered.",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
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

    ground_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to use when adding flags based on ground filter failures.",
    )

    azimuth = Unicode(
        defaults.azimuth, allow_none=True, help="Observation shared key for Azimuth"
    )

    boresight_azel = Unicode(
        defaults.boresight_azel,
        allow_none=True,
        help="Observation shared key for boresight Az/El",
    )

    trend_order = Int(
        5,
        allow_none=True,
        help="Order of a Legendre polynomial to fit along with the ground template.",
    )

    filter_order = Int(
        5,
        allow_none=True,
        help="Order of a Legendre polynomial to fit as a function of azimuth.",
    )

    bin_width = Quantity(
        None,
        allow_none=True,
        help="Azimuthal bin width of ground filter",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    detrend = Bool(
        False, help="Subtract the fitted trend along with the ground template"
    )

    split_template = Bool(
        False, help="Apply a different template for left and right scans"
    )

    leftright_interval = Unicode(
        defaults.throw_leftright_interval,
        help="Intervals for left-to-right scans",
    )

    rightleft_interval = Unicode(
        defaults.throw_rightleft_interval,
        help="Intervals for right-to-left scans",
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

    @traitlets.validate("trend_order")
    def _check_trend_order(self, proposal):
        check = proposal["value"]
        if check is not None and check < 0:
            raise traitlets.TraitError("Trend order should be a non-negative integer")
        return check

    @traitlets.validate("filter_order")
    def _check_filter_order(self, proposal):
        check = proposal["value"]
        if check is not None and check < 0:
            raise traitlets.TraitError("Filter order should be a non-negative integer")
        return check

    @traitlets.validate("bin_width")
    def _check_filter_order(self, proposal):
        check = proposal["value"]
        if check is not None and check.to_value(u.radian) <= 0:
            raise traitlets.TraitError("bin_width should be a positive quantity")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _split_templates(self, templates, obs):
        """Create separate templates for left and right-going scans"""
        split_templates = []
        split_masks = []
        for name in self.leftright_interval, self.rightleft_interval:
            mask = np.zeros(obs.n_local_samples, dtype=bool)
            for ival in obs.intervals[name]:
                mask[ival.first : ival.last] = True
            split_masks.append(mask)
        for template in templates:
            for mask in split_masks:
                split_template = template.copy()
                split_template[mask] = 0
                split_templates.append(split_template)
        split_templates = np.vstack(split_templates)
        return split_templates

    @function_timer
    def _make_poly_templates(self, phase):
        """Evaluate Legendre polynomial templates up to the requested
        order"""
        norder = self.filter_order + 1
        ltemplates = np.zeros([norder, phase.size])
        legendre_templates(phase, ltemplates, 0, norder)
        return ltemplates

    @function_timer
    def _make_bin_templates(self, az):
        # Assign each time stamp to an azimuthal bin
        wbin = self.bin_width.to_value(u.radian)
        ibin = (az // wbin).astype(int)

        # Find the set of hit azimuthal bins
        bins = np.unique(ibin)

        if self.filter_order is not None:
            # Discard one bin.  This makes the rest of the templates
            # relative to it and breaks degeneracy with polynomial templates
            cut = np.argmax(counts)
            good = np.ones(len(counts), dtype=bool)
            good[cut] = False
            bins = bins[good]

        # Each template is just a boolean mask that is true when
        # boresight is in a specific bin
        bin_templates = []
        for bin_ in bins:
            bin_templates.append((ibin == bin_).astype(float))
        return bin_templates

    @function_timer
    def build_templates(self, obs):
        """Construct the local ground template hierarchy"""

        views = obs.view[self.view]

        # Construct trend templates.  Full domain for x is [-1, 1]

        nsample = obs.n_local_samples
        x = np.arange(nsample) / nsample * 2 - 1

        templates = []

        if self.trend_order is not None:
            # Do not include the offset in the trend.  It will be part of
            # of the ground template

            legendre_trend = np.zeros([self.trend_order, nsample])
            legendre_templates(x, legendre_trend, 1, self.trend_order + 1)
            templates.append(legendre_trend)

        # Get boresight azimuth

        try:
            if self.azimuth is not None:
                az = obs.shared[self.azimuth]
            else:
                quats = obs.shared[self.boresight_azel]
                theta, phi, _ = qa.to_iso_angles(quats)
                az = 2 * np.pi - phi
        except Exception as e:
            msg = (
                f"Failed to get boresight azimuth from TOD.  "
                f"Perhaps it is not ground TOD? '{e}'"
            )
            raise RuntimeError(msg)

        # Figure out the azimuth range, accounting for observations that
        # cross the zero meridian

        azmin = np.amin(az)
        azmax = np.amax(az)
        while azmin < 0:
            azmin += 2 * np.pi
            azmax += 2 * np.pi
        if azmax - azmin > 2 * np.pi:
            # Full wrap around
            azmin = 0
            azmax = 2 * np.pi
            az %= 2 * np.pi

        # phase maps azimuth to [-1, 1]

        phase = (az - azmin) / (azmax - azmin) * 2 - 1

        # Polynomial templates

        if self.filter_order is not None:
            ltemplates = self._make_poly_templates(phase)
            if self.split_template:
                ltemplates = self._split_templates(ltemplates, obs)
            templates.append(ltemplates)

        # Binned templates

        if self.bin_width is not None:
            bin_templates = self._make_bin_templates(az)
            if self.split_template:
                bin_templates = self._split_templates(bin_templates, obs)
            templates.append(bin_templates)

        templates = np.vstack(templates)

        return templates

    @function_timer
    def fit_templates(
        self,
        obs,
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
        ngood = np.sum(good)
        if ngood == 0:
            return None, None, None, None

        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        proj = np.zeros(ntemplate)

        bin_proj_fast(ref, templates, good.astype(np.uint8), proj)
        if last_good is not None and np.all(good == last_good):
            # Flags have not changed, we can re-use the last inverse covariance
            invcov = last_invcov
            cov = last_cov
            rcond = last_rcond
        else:
            bin_invcov_fast(templates, good.astype(np.uint8), invcov)
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
                f"Ground template matrix is poorly conditioned, "
                f"rcond = {rcond}, using pseudoinverse."
            )
            if cov is None:
                cov = get_pseudoinverse(invcov)
        coeff = np.dot(cov, proj)

        return coeff, invcov, cov, rcond

    @function_timer
    def subtract_templates(self, ref, good, coeff, templates):
        if self.detrend:
            offset = 0
        else:
            offset = self.trend_order
        fit = np.zeros(ref.size, dtype=np.float64)
        add_templates(fit, templates[offset:], coeff[offset:])
        ref -= fit
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
            
            pat = re.compile(self.pattern)
            if obs.comm_row is not None and obs.comm_row.size != 1:
                raise RuntimeError("GroundFilter assumes data is split by detector")

            # Prefix for logging
            log_prefix = f"{data.comm.group} : {obs.name} :"

            if data.comm.group_rank == 0:
                msg = (
                    f"{log_prefix} OpGroundFilter: "
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
            templates = self.build_templates(obs)
            if data.comm.group_rank == 0:
                msg = (
                    f"{log_prefix} OpGroundFilter: "
                    f"Built templates in {time() - t1:.1f}s"
                )
                log.debug(msg)

            last_good = None
            last_invcov = None
            last_cov = None
            last_rcond = None

            for det in obs.select_local_detectors(detectors, flagmask=self.det_mask):
                if pat.match(det) is None:
                    continue
                if data.comm.group_rank == 0:
                    msg = f"{log_prefix} OpGroundFilter: " f"Processing detector {det}"
                    log.verbose(msg)

                ref = obs.detdata[self.det_data][det]
                if self.det_flags is not None:
                    test_flags = obs.detdata[self.det_flags][det] & self.det_flag_mask
                    good = np.logical_and(common_flags == 0, test_flags == 0)
                else:
                    good = common_flags == 0

                t1 = time()
                coeff, last_invcov, last_cov, last_rcond = self.fit_templates(
                    obs,
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
                        f"{log_prefix} OpGroundFilter: "
                        f"Fit templates in {time() - t1:.1f}s"
                    )
                    log.verbose(msg)

                if coeff is None:
                    # All samples flagged or template fit failed.
                    curflag = obs.local_detector_flags[det]
                    obs.update_local_detector_flags(
                        {det: curflag | self.ground_flag_mask}
                    )
                    continue

                t1 = time()
                self.subtract_templates(
                    ref,
                    good,
                    coeff,
                    templates,  # legendre_trend, legendre_filter
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
