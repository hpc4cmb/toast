# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

from ..mpi import MPI

import traitlets

import numpy as np

from astropy import units as u

import healpy as hp

from ..timing import function_timer

from .. import qarray as qa

from ..data import Data

from ..traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance

from .operator import Operator

from .pipeline import Pipeline

from ..utils import Environment, Logger, Timer

from .._libtoast import bin_templates, add_templates, legendre


@trait_docs
class GroundFilter(Operator):
    """Operator that applies ground template filtering to azimuthal scans.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        "signal", help="Observation detdata key for accumulating atmosphere timestreams"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    ground_flag_mask = Int(
        1, help="Bit mask to use when adding flags based on ground filter failures."
    )

    trend_order = Int(
        5, help="Order of a Legendre polynomial to fit along with the ground template."
    )

    filter_order = Int(
        5, help="Order of a Legendre polynomial to fit as a function of azimuth."
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
            raise traitlets.TraitError("Filtere order should be a non-negative integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def build_templates(self, obs):
        """Construct the local ground template hierarchy"""
        if self._intervals in obs:
            intervals = obs[self._intervals]
        else:
            intervals = None
        local_intervals = tod.local_intervals(intervals)

        # Construct trend templates.  Full domain for x is [-1, 1]

        my_offset, my_nsamp = tod.local_samples
        nsamp_tot = tod.total_samples
        x = np.arange(my_offset, my_offset + my_nsamp) / nsamp_tot * 2 - 1

        # Do not include the offset in the trend.  It will be part of
        # of the ground template
        legendre_trend = np.zeros([self._trend_order, x.size])
        legendre(x, legendre_trend, 1, self._trend_order + 1)

        try:
            (azmin, azmax, _, _) = tod.scan_range
            az = tod.read_boresight_az()
        except Exception as e:
            raise RuntimeError(
                "Failed to get boresight azimuth from TOD.  Perhaps it is "
                'not ground TOD? "{}"'.format(e)
            )

        # The azimuth vector is assumed to be arranged so that the
        # azimuth increases monotonously even across the zero meridian.

        phase = (az - azmin) / (azmax - azmin) * 2 - 1
        nfilter = self._filter_order + 1
        legendre_templates = np.zeros([nfilter, phase.size])
        legendre(phase, legendre_templates, 0, nfilter)
        if not self._split_template:
            legendre_filter = legendre_templates
        else:
            # Create separate templates for alternating scans
            common_ref = tod.local_common_flags(self._common_flag_name)
            legendre_filter = []
            mask1 = common_ref & tod.LEFTRIGHT_SCAN == 0
            mask2 = common_ref & tod.RIGHTLEFT_SCAN == 0
            for template in legendre_templates:
                for mask in mask1, mask2:
                    temp = template.copy()
                    temp[mask] = 0
                    legendre_filter.append(temp)
            del common_ref
            legendre_filter = np.vstack(legendre_filter)

        templates = np.vstack([legendre_trend, legendre_filter])

        return templates, legendre_trend, legendre_filter

    @function_timer
    def bin_templates(self, ref, templates, good, invcov, proj):

        # Bin the local data

        for i in range(ntemplate):
            temp = templates[i] * good
            proj[i] = np.dot(temp, ref)
            for j in range(i, ntemplate):
                invcov[i, j] = np.dot(temp, templates[j])
                # Symmetrize invcov
                if i != j:
                    invcov[j, i] = invcov[i, j]
            del temp

        return

    @function_timer
    def fit_templates(self, tod, det, templates, ref, good):
        log = Logger.get()
        # communicator for processes with the same detectors
        comm = tod.grid_comm_row
        ngood = np.sum(good)
        if comm is not None:
            ngood = comm.allreduce(ngood)
        if ngood == 0:
            return None

        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        proj = np.zeros(ntemplate)

        bin_templates(ref, templates, good.astype(np.uint8), invcov, proj)

        if comm is not None:
            # Reduce the binned data.  The detector signals is
            # distributed across the group communicator.
            comm.Allreduce(MPI.IN_PLACE, invcov, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, proj, op=MPI.SUM)

        # Assemble the joint template
        rcond = 1 / np.linalg.cond(invcov)
        self.rcondsum += rcond
        if rcond > 1e-6:
            self.ngood += 1
            cov = np.linalg.inv(invcov)
            coeff = np.dot(cov, proj)
        else:
            self.nsingular += 1
            log.debug(
                f"Ground template matrix is poorly conditioned, "
                f"rcond = {rcond}, doing least squares fitting."
            )
            # np.linalg.lstsq will find a least squares minimum
            # even if the covariance matrix is not invertible
            coeff = np.linalg.lstsq(invcov, proj, rcond=1e-30)[0]
            if np.any(np.isnan(coeff)) or np.std(coeff) < 1e-30:
                raise RuntimeError("lstsq FAILED")

        return coeff

    @function_timer
    def subtract_templates(self, ref, good, coeff, legendre_trend, legendre_filter):
        # Trend
        if self._detrend:
            trend = np.zeros_like(ref)
            add_templates(trend, legendre_trend, coeff[: self._trend_order])
            ref[good] -= trend[good]
        # Ground template
        grtemplate = np.zeros_like(ref)
        add_templates(grtemplate, legendre_filter, coeff[self._trend_order :])
        ref[good] -= grtemplate[good]
        ref[np.logical_not(good)] = 0
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

            if gcomm.rank == 0:
                msg = f"{log_prefix} OpGroundFilter: " \
                    f"Processing observation {iobs + 1} / {nobs}"
                log.debug(msg)

            # Cache the output common flags
            common_flags = (obs.shared[self.shared_flags] & self.shared_flag_mask) != 1

            t1 = time()
            templates, legendre_trend, legendre_filter = self.build_templates(obs)
            if gcomm.rank == 0:
                msg = f"{log_prefix} OpGroundFilter: " \
                    f"Built templates in {time() - t1:.1f}s"
                log.debug(msg)

            ndet = len(tod.local_dets)
            for idet, det in enumerate(tod.local_dets):
                if gcomm.rank == 0:
                    msg = f"{log_prefix} OpGroundFilter: " \
                        f"Processing detector # {idet + 1} / {ndet}"

                ref = tod.local_signal(det, self._name)
                flag_ref = tod.local_flags(det, self._flag_name)
                good = np.logical_and(
                    common_ref & self._common_flag_mask == 0,
                    flag_ref & self._flag_mask == 0,
                )
                del flag_ref

                t1 = time()
                coeff = self.fit_templates(tod, det, templates, ref, good)
                if gcomm.rank == 0:
                    msg = f"{log_prefix} OpGroundFilter: " \
                        f"Fit templates in {time() - t1:.1f}s"
                    log.debug(msg)

                if coeff is None:
                    continue

                t1 = time()
                self.subtract_templates(
                    ref, good, coeff, legendre_trend, legendre_filter
                )
                if gcomm.rank == 0:
                    msg = f"{log_prefix} OpGroundFilter: " \
                        f"Subtract templates in {time() - t1:.1f}s"
                    log.debug(msg)

                del ref

            del common_ref

        if comm is not None:
            self.nsingular = comm.allreduce(self.nsingular)
            self.ngood = comm.allreduce(self.ngood)
            self.rcondsum = comm.allreduce(self.rcondsum)

        if wcomm.rank == 0:
            rcond_mean = self.rcondsum / (self.nsingular + self.ngood)
            msg =  f"Applied ground filter in {time() - t0:.1f} s.  " \
                f"Average rcond of template matrix was {rcond_mean}"
            log.debug(msg)

        return
