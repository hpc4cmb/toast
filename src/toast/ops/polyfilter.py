# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
from time import time
import warnings

from astropy import units as u
import numpy as np
import scipy
import traitlets

import jax
import jax.numpy as jnp
from jax.config import config as jax_config
# enable 64bits precision
jax_config.update("jax_enable_x64", True)

from ..mpi import MPI, MPI_Comm, use_mpi, Comm

from .operator import Operator
from .. import qarray as qa
from ..timing import function_timer
from ..traits import trait_docs, Int, Unicode, Bool, Dict, Quantity, Instance
from ..utils import (
    Logger,
    Environment,
    Timer,
    GlobalTimers,
    dtype_to_aligned,
    AlignedF64,
    AlignedU8,
)
from ..observation import default_values as defaults

from .._libtoast import filter_polynomial, filter_poly2D, sum_detectors, subtract_mean


XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class PolyFilter2D(Operator):
    """Operator to regress out 2D polynomials across the focal plane."""

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to",
    )

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are filtered.",
    )

    order = Int(1, allow_none=False, help="Polynomial order")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    poly_flag_mask = Int(1, help="Bit mask value for intervals that fail to filter")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    focalplane_key = Unicode(
        None, allow_none=True, help="Which focalplane key to match"
    )

    use_python = Bool(
        False, help="If True, use a pure python implementation for testing."
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        gt = GlobalTimers.get()

        if detectors is not None:
            raise RuntimeError("PolyFilter2D cannot be run on subsets of detectors")
        norder = self.order + 1
        nmode = norder**2
        pat = re.compile(self.pattern)

        for obs in data.obs:
            # Get the original number of process rows in the observation
            proc_rows = obs.dist.process_rows

            # Duplicate just the fields of the observation we will use
            gt.start("Poly2D:  Duplicate obs")
            dup_shared = list()
            if self.shared_flags is not None:
                dup_shared.append(self.shared_flags)
            dup_detdata = [self.det_data]
            if self.det_flags is not None:
                dup_detdata.append(self.det_flags)
            dup_intervals = list()
            if self.view is not None:
                dup_intervals.append(self.view)
            temp_ob = obs.duplicate(
                times=self.times,
                meta=list(),
                shared=dup_shared,
                detdata=dup_detdata,
                intervals=dup_intervals,
            )
            gt.stop("Poly2D:  Duplicate obs")

            # Redistribute this temporary observation to be distributed by samples
            gt.start("Poly2D:  Forward redistribution")
            temp_ob.redistribute(1, times=self.times, override_sample_sets=None)
            gt.stop("Poly2D:  Forward redistribution")

            gt.start("Poly2D:  Detector setup")

            # Detectors to process
            detectors = []
            for det in temp_ob.all_detectors:
                if pat.match(det) is None:
                    continue
                detectors.append(det)
            ndet = len(detectors)
            if ndet == 0:
                continue

            # Detector positions

            detector_position = {}
            for det in detectors:
                det_quat = temp_ob.telescope.focalplane[det]["quat"]
                x, y, z = qa.rotate(det_quat, ZAXIS)
                theta, phi = np.arcsin([x, y])
                detector_position[det] = [theta, phi]

            # Enumerate detector groups (e.g. wafers) to filter

            # The integer group ID for a given detector
            group_index = {}

            # The list of detectors for each group key
            groups = {}

            # The integer group ID for each group key
            group_ids = {}

            if self.focalplane_key is None:
                # We have just one group of all detectors
                groups[None] = []
                group_ids[None] = 0
                ngroup = 1
                for det in detectors:
                    group_index[det] = 0
                    groups[None].append(det)
            else:
                focalplane = temp_ob.telescope.focalplane
                if self.focalplane_key not in focalplane.detector_data.colnames:
                    msg = (
                        f"Cannot divide detectors by {self.focalplane_key} because "
                        "it is not defined in the focalplane detector data."
                    )
                    raise RuntimeError(msg)
                for det in detectors:
                    value = focalplane[det][self.focalplane_key]
                    if value not in groups:
                        groups[value] = []
                    groups[value].append(det)
                ngroup = len(groups)
                for igroup, group in enumerate(sorted(groups)):
                    group_ids[group] = igroup
                    for det in groups[group]:
                        group_index[det] = igroup

            # Enumerate detectors to process

            # Mapping from detector name to index
            detector_index = {y: x for x, y in enumerate(detectors)}

            # The integer group ID for a given detector index
            group_det = np.array([group_index[x] for x in detectors])

            # Measure offset for each group, translate and scale
            # detector positions to [-1, 1]

            group_offset = {}
            all_positions = []
            for group, detectors_group in groups.items():
                ndet_group = len(detectors_group)
                theta_offset, phi_offset = 0, 0
                for det in detectors_group:
                    theta, phi = detector_position[det]
                    theta_offset += theta
                    phi_offset += phi
                theta_offset /= ndet_group
                phi_offset /= ndet_group
                for det in detectors_group:
                    theta, phi = detector_position[det]
                    detector_position[det] = [theta - theta_offset, phi - phi_offset]
                    all_positions.append(detector_position[det])

            thetavec, phivec = np.vstack(all_positions).T
            thetamax = np.amax(np.abs(thetavec))
            phimax = np.amax(np.abs(phivec))
            scale = 0.999 / max(thetamax, phimax)

            for det in detector_position:
                theta, phi = detector_position[det]
                detector_position[det] = [theta * scale, phi * scale]

            # Now evaluate the polynomial templates at the sites of
            # each detector

            orders = np.arange(norder)
            xorders, yorders = np.meshgrid(orders, orders, indexing="ij")
            xorders = xorders.ravel()
            yorders = yorders.ravel()

            detector_templates = np.zeros([ndet, nmode])
            for det in temp_ob.local_detectors:
                if det not in detector_index:
                    continue
                idet = detector_index[det]
                theta, phi = detector_position[det]
                detector_templates[idet] = theta**xorders * phi**yorders

            gt.stop("Poly2D:  Detector setup")

            # Iterate over each interval

            # Aligned memory objects using C-allocated memory so that we
            # can explicitly free it after processing.
            template_mem = AlignedF64()
            mask_mem = AlignedU8()
            signal_mem = AlignedF64()
            coeff_mem = AlignedF64()

            views = temp_ob.view[self.view]
            for iview, view in enumerate(views):
                if view.start is None:
                    # This is a view of the whole obs
                    nsample = temp_ob.n_local_samples
                else:
                    nsample = view.stop - view.start

                # Accumulate the linear regression templates

                gt.start("Poly2D:  Accumulate templates")

                template_mem.resize(ndet * nmode)
                template_mem[:] = 0
                templates = template_mem.array().reshape((ndet, nmode))

                mask_mem.resize(nsample * ndet)
                mask_mem[:] = 0
                masks = mask_mem.array().reshape((nsample, ndet))

                signal_mem.resize(nsample * ndet)
                signal_mem[:] = 0
                signals = signal_mem.array().reshape((nsample, ndet))

                coeff_mem.resize(nsample * ngroup * nmode)
                coeff_mem[:] = 0
                coeff = coeff_mem.array().reshape((nsample, ngroup, nmode))

                det_groups = -1 * np.ones(ndet, dtype=np.int32)

                if self.shared_flags is not None:
                    shared_flags = views.shared[self.shared_flags][iview]
                    shared_mask = (shared_flags & self.shared_flag_mask) == 0
                else:
                    shared_mask = np.ones(nsample, dtype=bool)

                for idet, det in enumerate(temp_ob.local_detectors):
                    if det not in detector_index:
                        continue
                    ind_det = detector_index[det]
                    ind_group = group_index[det]
                    det_groups[ind_det] = ind_group

                    signal = views.detdata[self.det_data][iview][idet]
                    if self.det_flags is not None:
                        det_flags = views.detdata[self.det_flags][iview][idet]
                        det_mask = (det_flags & self.det_flag_mask) == 0
                        mask = np.logical_and(shared_mask, det_mask)
                    else:
                        mask = shared_mask

                    templates[ind_det, :] = detector_templates[ind_det]
                    masks[:, ind_det] = mask
                    signals[:, ind_det] = signal * mask

                gt.stop("Poly2D:  Accumulate templates")

                if self.use_python:
                    gt.start("Poly2D:  Solve templates (with python)")
                    for isample in range(nsample):
                        for group, igroup in group_ids.items():
                            good = group_det == igroup
                            mask = masks[isample, good]
                            t = templates[good].T.copy() * mask
                            proj = np.dot(t, signals[isample, good] * mask)
                            ccinv = np.dot(t, t.T)
                            coeff[isample, igroup] = np.linalg.lstsq(
                                ccinv, proj, rcond=None
                            )[0]
                    gt.stop("Poly2D:  Solve templates (with python)")
                else:
                    gt.start("Poly2D:  Solve templates")
                    filter_poly2D(det_groups, templates, signals, masks, coeff)
                    gt.stop("Poly2D:  Solve templates")

                gt.start("Poly2D:  Update detector flags")

                for igroup in range(ngroup):
                    local_dets = temp_ob.local_detectors
                    dets_in_group = np.zeros(len(local_dets), dtype=np.bool)
                    for idet, det in enumerate(local_dets):
                        if group_index[det] == igroup:
                            dets_in_group[idet] = True
                    if not np.any(dets_in_group):
                        continue
                    if self.det_flags is not None:
                        sample_flags = np.ones(
                            len(local_dets),
                            dtype=views.detdata[self.det_flags][0].dtype,
                        )
                        sample_flags *= self.poly_flag_mask
                        sample_flags *= dets_in_group
                        for isample in range(nsample):
                            if np.all(coeff[isample, igroup] == 0):
                                views.detdata[self.det_flags][iview][
                                    :, isample
                                ] |= sample_flags

                gt.stop("Poly2D:  Update detector flags")

                gt.start("Poly2D:  Clean timestreams")

                trcoeff = np.transpose(
                    np.array(coeff), [1, 0, 2]
                )  # ngroup x nsample x nmode
                trmasks = np.array(masks).T  # ndet x nsample
                for idet, det in enumerate(temp_ob.local_detectors):
                    if det not in detector_index:
                        continue
                    igroup = group_index[det]
                    ind = detector_index[det]
                    signal = views.detdata[self.det_data][iview][idet]
                    mask = trmasks[idet]
                    signal -= np.sum(trcoeff[igroup] * templates[ind], 1) * mask

                gt.stop("Poly2D:  Clean timestreams")

            # Redistribute back
            gt.start("Poly2D:  Backward redistribution")
            temp_ob.redistribute(
                proc_rows, times=self.times, override_sample_sets=obs.dist.sample_sets
            )
            gt.stop("Poly2D:  Backward redistribution")

            # Copy data to original observation
            gt.start("Poly2D:  Copy output")
            obs.detdata[self.det_data][:] = temp_ob.detdata[self.det_data][:]
            gt.stop("Poly2D:  Copy output")

            # Free data copy
            temp_ob.clear()
            del temp_ob

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
        }
        return prov


@trait_docs
class PolyFilter(Operator):
    """Operator which applies polynomial filtering to the TOD."""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key apply filtering to"
    )

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are filtered.",
    )

    order = Int(1, allow_none=False, help="Polynomial order")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    poly_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Shared flag bit mask for samples that are not filtered",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    view = Unicode(
        "throw", allow_none=True, help="Use this view of the data in all observations"
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.pattern is None:
            pat = None
        else:
            pat = re.compile(self.pattern)

        for obs in data.obs:
            # Get the detectors we are using for this observation
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            if self.view is not None:
                if self.view not in obs.intervals:
                    msg = (
                        f"PolyFilter is configured to apply in the '{self.view}' view "
                        f"but it is not defined for observation '{obs.name}'"
                    )
                    raise RuntimeError(msg)
                local_starts = []
                local_stops = []
                for interval in obs.intervals[self.view]:
                    local_starts.append(interval.first)
                    local_stops.append(interval.last)
            else:
                local_starts = [0]
                local_stops = [obs.n_local_samples - 1]

            local_starts = np.array(local_starts)
            local_stops = np.array(local_stops)

            if self.shared_flags is not None:
                shared_flags = (
                    obs.shared[self.shared_flags].data & self.shared_flag_mask
                )
            else:
                shared_flags = np.zeros(obs.n_local_samples, dtype=np.uint8)

            signals = []
            last_flags = None
            for idet, det in enumerate(dets):
                # Test the detector pattern
                if pat.match(det) is None:
                    continue

                signal = obs.detdata[self.det_data][idet]
                if self.det_flags is not None:
                    det_flags = obs.detdata[self.det_flags][idet] & self.det_flag_mask
                    flags = shared_flags | det_flags
                else:
                    flags = shared_flags

                if last_flags is None or np.all(last_flags == flags):
                    signals.append(signal)
                else:
                    filter_polynomial_switch(
                        self.order, last_flags, signals, local_starts, local_stops
                    )
                    signals = [signal]
                last_flags = flags.copy()

            if len(signals) > 0:
                filter_polynomial_switch(
                    self.order, last_flags, signals, local_starts, local_stops
                )

            # Optionally flag unfiltered data
            if self.shared_flags is not None and self.poly_flag_mask is not None:
                shared_flags = np.array(obs.shared[self.shared_flags])
                last_stop = None
                for stop, start in zip(local_starts, local_stops):
                    if last_stop is not None and last_stop < start:
                        shared_flags[last_stop:start] |= self.poly_flag_mask
                obs.shared[self.shared_flags].set(shared_flags, fromrank=0)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": [self.view],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov


class CommonModeFilter(Operator):
    """Operator to regress out common mode at each time stamp."""

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key apply filtering to"
    )

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are filtered.",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    focalplane_key = Unicode(
        None, allow_none=True, help="Which focalplane key to match"
    )

    redistribute = Bool(
        False,
        help="If True, redistribute data before and after filtering for "
        "optimal data locality.",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _redistribute(self, data, obs, timer, log):
        if self.redistribute:
            # Redistribute the data so each process has all detectors for some sample range
            # Duplicate just the fields of the observation we will use
            dup_shared = list()
            if self.shared_flags is not None:
                dup_shared.append(self.shared_flags)
            dup_detdata = [self.det_data]
            if self.det_flags is not None:
                dup_detdata.append(self.det_flags)
            dup_intervals = list()
            temp_ob = obs.duplicate(
                times=self.times,
                meta=list(),
                shared=dup_shared,
                detdata=dup_detdata,
                intervals=dup_intervals,
            )
            log.debug_rank(
                f"{data.comm.group:4} : Duplicated observation in",
                comm=temp_ob.comm,
                timer=timer,
            )
            # Redistribute this temporary observation to be distributed by sample sets
            temp_ob.redistribute(1, times=self.times, override_sample_sets=None)
            log.debug_rank(
                f"{data.comm.group:4} : Redistributed observation in",
                comm=temp_ob.comm,
                timer=timer,
            )
            comm = None
        else:
            comm = obs.comm_col
            temp_ob = obs
            proc_rows = None

        return comm, temp_ob

    @function_timer
    def _re_redistribute(self, data, obs, timer, log, temp_ob):
        if self.redistribute:
            # Redistribute data back
            temp_ob.redistribute(
                obs.dist.process_rows,
                times=self.times,
                override_sample_sets=obs.dist.sample_sets,
            )
            log.debug_rank(
                f"{data.comm.group:4} : Re-redistributed observation in",
                comm=temp_ob.comm,
                timer=timer,
            )
            # Copy data to original observation
            obs.detdata[self.det_data][:] = temp_ob.detdata[self.det_data][:]
            log.debug_rank(
                f"{data.comm.group:4} : Copied observation data in",
                comm=temp_ob.comm,
                timer=timer,
            )
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        """Apply the common mode filter to the signal.

        Args:
            data (toast.Data): The distributed data.

        """
        if detectors is not None:
            raise RuntimeError("CommonModeFilter cannot be run in batch mode")

        log = Logger.get()
        timer = Timer()
        timer.start()
        pat = re.compile(self.pattern)

        for obs in data.obs:
            comm, temp_ob = self._redistribute(data, obs, timer, log)

            focalplane = temp_ob.telescope.focalplane

            detectors = temp_ob.all_detectors
            if self.focalplane_key is None:
                values = [None]
            else:
                values = set()
                for det in detectors:
                    if pat.match(det) is None:
                        continue
                    values.add(focalplane[det][self.focalplane_key])
                values = sorted(values)

            nsample = temp_ob.n_local_samples
            ndet = len(temp_ob.local_detectors)

            # Loop over all values of the focalplane key
            for value in values:
                local_dets = []
                for idet, det in enumerate(temp_ob.local_detectors):
                    if pat.match(det) is None:
                        continue
                    if (
                        value is not None
                        and focalplane[det][self.focalplane_key] != value
                    ):
                        continue
                    local_dets.append(idet)
                local_dets = np.array(local_dets)

                # Average all detectors that match the key
                template = np.zeros(nsample)
                hits = np.zeros(nsample, dtype=np.int64)

                if self.shared_flags is not None:
                    shared_flags = temp_ob.shared[self.shared_flags].data
                else:
                    shared_flags = np.zeros(nsample, dtype=np.uint8)
                if self.det_flags is not None:
                    det_flags = temp_ob.detdata[self.det_flags].data
                else:
                    det_flags = np.zeros([ndet, nsample], dtype=np.uint8)

                sum_detectors(
                    local_dets,
                    shared_flags,
                    self.shared_flag_mask,
                    temp_ob.detdata[self.det_data].data,
                    det_flags,
                    self.det_flag_mask,
                    template,
                    hits,
                )

                if comm is not None:
                    comm.Barrier()
                    comm.Allreduce(MPI.IN_PLACE, template, op=MPI.SUM)
                    comm.Allreduce(MPI.IN_PLACE, hits, op=MPI.SUM)

                subtract_mean(
                    local_dets,
                    temp_ob.detdata[self.det_data].data,
                    template,
                    hits,
                )

            log.debug_rank(
                f"{data.comm.group:4} : Commonfiltered observation in",
                comm=temp_ob.comm.comm_group,
                timer=timer,
            )

            self._re_redistribute(data, obs, timer, log, temp_ob)
            if self.redistribute:
                # In this case our temp_ob holds a copied subset of the
                # observation.  Clear it.
                temp_ob.clear()
                del temp_ob

        return

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
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov

def filter_polynomial_switch(order, flags, signals, starts, stops, use_compiled=False):
    """
    Used in test to select the `filter_polynomial` implementation
    TODO: remove once tests are done
    """
    if use_compiled: filter_polynomial(order, flags, signals, starts, stops)
    else: filter_polynomial_jax(order, flags, signals, starts, stops)


def filter_polynomial_jax(order, flags, signals_list, starts, stops):
    """
    Fit and subtract a polynomial from one or more signals.

    Args:
        order (int):  The order of the polynomial.
        flags (numpy array, uint8):  The common flags to use for all signals
        signals_list (list of numpy array of double):  A list of float64 arrays containing the signals.
        starts (numpy array, int64):  The start samples of each scan.
        stops (numpyarray, int64):  The stop samples of each scan.

    Returns:
        None: The signals are updated in place.

    NOTE: port of `filter_polynomial` from compiled code to JAX
    """
    # validate order
    if (order < 0): return

    # problem size
    n = flags.size
    nsignal = len(signals_list)
    norder = order + 1
    scanlen_upperbound = np.max(stops + 1 - starts)
    #print(f"DEBUG: n:{n} nsignal:{nsignal} norder:{norder} scanlen_upperbound:{scanlen_upperbound}") # DEBUG

    # process a single interval, returns the shift that should be applied to result
    # scanlen_upperbound is an upperbound that should be known at JIT time
    def filter_polynomial_interval(flags, signals, start, stop, scanlen_upperbound):
        # validates interval
        start = jnp.maximum(0, start)
        stop = jnp.minimum(n-1, stop)
        scanlen = stop - start + 1

        # extracts the interval of signals
        # NOTE: we take an upper bound (scanlen_upperbound rather than scanlen) in order to have a size known at JIT time
        #       we will mask it later to cover only the interval of interest
        # signals_interval = signals[start:(stop+1),:] # scanlen*nsignal
        signals_interval = jax.lax.dynamic_slice(signals, (start,0), (scanlen_upperbound,nsignal)) # scanlen_upperbound*nsignal
        # extracts the interval of flags
        # NOTE: we take an upper bound (scanlen_upperbound rather than scanlen) in order to have a size known at JIT time
        #       we will mask it later to cover only the interval of interest
        # flags_interval = flags[start:(stop+1)] # scanlen
        flags_interval = jax.lax.dynamic_slice(flags, (start,), (scanlen_upperbound,)) # scanlen_upperbound

        # builds masks operate within the actual interval and where flags are set to 0
        indices = jnp.arange(start=0, stop=scanlen_upperbound)
        mask_interval = indices < scanlen
        mask_flag = flags_interval == 0
        mask = mask_interval & mask_flag

        # Build the full template matrix used to clean the signal.
        # We subtract the template value even from flagged samples to support point source masking etc.
        full_templates = jnp.zeros(shape=(scanlen_upperbound, norder)) # scanlen_upperbound*norder
        # defines x
        xstart = (1. / scanlen) - 1.
        dx = 2. / scanlen
        x = xstart + dx*indices
        # deals with order 0
        if norder > 0: 
            # full_templates[:,0] = 1
            full_templates = full_templates.at[:,0].set(1)
        # deals with order 1
        if norder > 1: 
            # full_templates[:,1] = x
            full_templates = full_templates.at[:,1].set(x)
        # deals with other orders
        # NOTE: this formulation is inherently sequential but this should be okay as `order` is likely small
        for iorder in range(2,norder):
            previous_previous_order = full_templates[:,iorder-2]
            previous_order = full_templates[:,iorder-1]
            current_order = ((2 * iorder - 1) * x * previous_order - (iorder - 1) * previous_previous_order) / iorder
            # full_templates[:,iorder] = current_order
            full_templates = full_templates.at[:,iorder].set(current_order)
        # zero out rows that fall outside the interval
        full_templates = full_templates * mask_interval[:, jnp.newaxis] # scanlen*norder

        # Assemble the flagged template matrix used in the linear regression
        # We zero out the rows that are flagged or outside the interval
        masked_templates = full_templates * mask[:, jnp.newaxis] # nb_zero_flags*norder

        # Square the template matrix for A^T.A
        invcov = jnp.dot(masked_templates.T, masked_templates) # norder*norder

        # We zero out the rows that are flagged or outside the interval
        masked_signals = signals_interval * mask[:, jnp.newaxis] # nb_zero_flags*nsignal

        # Project the signals against the templates
        proj = jnp.dot(masked_templates.T, masked_signals) # norder*nsignal

        # Fit the templates against the data
        # by minimizing the norm2 of the difference and the solution vector
        (x, _residue, _rank, _singular_values) = jnp.linalg.lstsq(invcov, proj, rcond=1e-3) # norder*nsignal

        # computes the value to be subtracted from the signals
        # signals_interval[scanlen,nsignal] -= x[norder,nsignal] * full_templates[scanlen,norder]
        # signals_interval -= jnp.einsum('ij,ki->kj', x, full_templates)
        result = jnp.zeros_like(signals)
        shift_signal = -jnp.einsum('ij,ki->kj', x, full_templates)
        return jax.lax.dynamic_update_slice(result, shift_signal, (start,0))
    
    # process a batch of intervals, returns the shift that should be applied to result
    # scanlen_upperbound is an upperbound that should be known at JIT time
    def filter_polynomial_intervals(flags, signals_list, starts, stops, scanlen_upperbound):
        #print(f"DEBUG: jit-compiling for {starts.size} intervals and an upper scanlen of {scanlen_upperbound}")
        # converts signal into a flat array to avoid having to loop over them and simplify further processing
        signals = jnp.vstack(signals_list).T # n*nsignal
        # padds signals and flags to insure that calls to dynamic_slice with scanlen_upperbound will fall inside the arrays
        flags = jnp.pad(flags, pad_width=((0,scanlen_upperbound),), mode='empty')
        signals = jnp.pad(signals, pad_width=((0,scanlen_upperbound),(0,0)), mode='empty')
        # converts the function to batch it over start and stop, the new function will return batched results
        batched_filter_polynomial = jax.vmap(filter_polynomial_interval, in_axes=(None,None,0,0,None), out_axes=0)
        # gets shifts, batched along the 0 dimenssion
        shift_signal_batched = batched_filter_polynomial(flags, signals, starts, stops, scanlen_upperbound)
        # sums along the 0 dimension to get the overall shift
        shift_signal = jnp.sum(shift_signal_batched, axis=0)
        # corrects signals with the shift
        return signals + shift_signal
    # JIT compiles the JAX function
    filter_polynomial_intervals = jax.jit(filter_polynomial_intervals, static_argnames='scanlen_upperbound')

    # updates the signals
    new_signals = filter_polynomial_intervals(flags, signals_list, starts, stops, scanlen_upperbound)

    # puts resulting signals back into list form
    for isignal in range(nsignal):
        # we give size inormation (0:n) as we added some padding to new_signals
        signals_list[isignal][:] = new_signals[0:n,isignal]

def filter_polynomial_numpy(order, flags, signals, starts, stops):
    """
    Fit and subtract a polynomial from one or more signals.

    Args:
        order (int):  The order of the polynomial.
        flags (numpy array, uint8):  The common flags to use for all signals
        signals (list of numpy array of double):  A list of float64 arrays containing the signals.
        starts (numpy array, int64):  The start samples of each scan.
        stops (numpyarray, int64):  The stop samples of each scan.

    Returns:
        None: The signals are updated in place.

    NOTE: port of `filter_polynomial` from compiled code to Numpy
    """
    # validate order
    if (order < 0): return

    # problem size
    n = flags.size
    nsignal = len(signals)
    norder = order + 1

    # converts signal into a numpy array to avoid having to loop over them
    # NOTE: this could be done by default removing the need for this step
    signals_np = np.vstack(signals).T # n*nsignal

    # NOTE: that loop is parallel in the C++ code
    for (start, stop) in zip(starts, stops):
        # validates interval
        if (start < 0): start = 0
        if (stop > n - 1): stop = n - 1
        if (stop < start): continue
        scanlen = stop - start + 1

        # extracts the signals that will be impacted by this interval
        signals_interval = signals_np[start:(stop+1),:] # scanlen*nsignal

        # set aside the indexes of the zero flags to be used as a mask
        flags_interval = flags[start:(stop+1)] # scanlen
        zero_flags = np.where(flags_interval == 0)
        nb_zero_flags = zero_flags[0].size
        if (nb_zero_flags == 0): continue

        # Build the full template matrix used to clean the signal.
        # We subtract the template value even from flagged samples to
        # support point source masking etc.
        full_templates = np.zeros(shape=(scanlen, norder)) # scanlen*norder
        xstart = (1. / scanlen) - 1.
        xstop = (1. / scanlen) + 1.
        dx = 2. / scanlen
        x = np.arange(start=xstart, stop=xstop, step=dx)
        # deals with order 0
        if norder > 0: full_templates[:,0] = 1
        # deals with order 1
        if norder > 1: full_templates[:,1] = x
        # deals with other orders
        # NOTE: this formulation is inherently sequential but this should be okay as `order` is likely small
        for iorder in range(2,norder):
            previous_previous_order = full_templates[:,iorder-2]
            previous_order = full_templates[:,iorder-1]
            full_templates[:,iorder] = ((2 * iorder - 1) * x * previous_order - (iorder - 1) * previous_previous_order) / iorder
        
        # Assemble the flagged template matrix used in the linear regression
        masked_templates = full_templates[zero_flags] # nb_zero_flags*norder

        # Square the template matrix for A^T.A
        invcov = np.dot(masked_templates.T, masked_templates) # norder*norder

        # Project the signals against the templates
        masked_signals = signals_interval[zero_flags] # nb_zero_flags*nsignal
        proj = np.dot(masked_templates.T, masked_signals) # norder*nsignal

        # Fit the templates against the data
        # by minimizing the norm2 of the difference and the solution vector
        (x, _residue, _rank, _singular_values) = np.linalg.lstsq(invcov, proj, rcond=1e-3) # norder*nsignal
        # signals_interval[scanlen,nsignal] -= x[norder,nsignal] * full_templates[scanlen,norder]
        signals_interval -= np.einsum('ij,ki->kj', x, full_templates)
    
    # puts resulting signals back into list form
    for isignal in range(nsignal):
        signals[isignal][:] = signals_np[:,isignal]

"""
void toast::filter_polynomial(int64_t order, size_t n, uint8_t * flags,
                              std::vector <double *> const & signals, size_t nscan,
                              int64_t const * starts, int64_t const * stops) {
    if (order < 0) return;

    int nsignal = signals.size();
    int norder = order + 1;

    char upper = 'U';
    char lower = 'L';
    char notrans = 'N';
    char trans = 'T';
    double fzero = 0.0;
    double fone = 1.0;

    for (size_t iscan = 0; iscan < nscan; ++iscan) 
    {
        int64_t start = starts[iscan];
        int64_t stop = stops[iscan];
        if (start < 0) start = 0;
        if (stop > n - 1) stop = n - 1;
        if (stop < start) continue;
        int scanlen = stop - start + 1;

        int ngood = 0;
        for (size_t i = 0; i < scanlen; ++i) 
        {
            if (flags[start + i] == 0) ngood++;
        }
        if (ngood == 0) continue;

        // Build the full template matrix used to clean the signal.
        // We subtract the template value even from flagged samples to
        // support point source masking etc.
        toast::AlignedVector <double> full_templates(scanlen * norder);

        double dx = 2. / scanlen;
        double xstart = 0.5 * dx - 1;
        double * current, * last, * lastlast;

        for (size_t iorder = 0; iorder < norder; ++iorder) 
        {
            current = &full_templates[iorder * scanlen];
            if (iorder == 0) 
            {
                for (size_t i = 0; i < scanlen; ++i) 
                {
                    current[i] = 1;
                }
            } 
            else if (iorder == 1) 
            {
                for (size_t i = 0; i < scanlen; ++i) 
                {
                    const double x = xstart + i * dx;
                    current[i] = x;
                }
            } 
            else 
            {
                last = &full_templates[(iorder - 1) * scanlen];
                lastlast = &full_templates[(iorder - 2) * scanlen];
                double orderinv = 1. / iorder;

                for (size_t i = 0; i < scanlen; ++i) 
                {
                    const double x = xstart + i * dx;
                    current[i] = ((2 * iorder - 1) * x * last[i] - (iorder - 1) * lastlast[i]) * orderinv;
                }
            }
        }

        // Assemble the flagged template matrix used in the linear
        // regression

        toast::AlignedVector <double> masked_templates(ngood * norder);

        for (size_t iorder = 0; iorder < norder; ++iorder) 
        {
            size_t offset = iorder * ngood;
            current = &full_templates[iorder * scanlen];
            for (size_t i = 0; i < scanlen; ++i) 
            {
                if (flags[start + i] == 0) 
                {
                    masked_templates[offset++] = current[i];
                }
            }
        }

        // Square the template matrix for A^T.A
        toast::AlignedVector <double> invcov(norder * norder);
        toast::LinearAlgebra::syrk(upper, trans, norder, ngood, fone,
                                   masked_templates.data(), ngood, fzero, invcov.data(),
                                   norder);

        // Project the signals against the templates

        toast::AlignedVector <double> masked_signals(ngood * nsignal);

        for (size_t isignal = 0; isignal < nsignal; ++isignal) 
        {
            size_t offset = isignal * ngood;
            double * signal = signals[isignal] + start;
            for (int64_t i = 0; i < scanlen; ++i) 
            {
                if (flags[start + i] == 0) 
                {
                    masked_signals[offset++] = signal[i];
                }
            }
        }

        toast::AlignedVector <double> proj(norder * nsignal);

        toast::LinearAlgebra::gemm(trans, notrans, norder, nsignal, ngood,
                                   fone, masked_templates.data(), ngood,
                                   masked_signals.data(), ngood,
                                   fzero, proj.data(), norder);

        // Symmetrize the covariance matrix, dgells is written for
        // generic matrices

        for (size_t row = 0; row < norder; ++row) 
        {
            for (size_t col = row + 1; col < norder; ++col) 
            {
                invcov[col + row * norder] = invcov[row + col * norder];
            }
        }

        // Fit the templates against the data.
        // DGELSS minimizes the norm of the difference and the solution vector
        // and overwrites proj with the fitting coefficients.
        int rank, info;
        double rcond_limit = 1e-3;
        int LWORK = toast::LinearAlgebra::gelss_buffersize(norder, norder, nsignal,
                                                           norder, norder, rcond_limit);
        toast::AlignedVector <double> WORK(LWORK);
        toast::AlignedVector <double> singular_values(norder);
        toast::LinearAlgebra::gelss(
            norder, norder, nsignal, invcov.data(), norder,
            proj.data(), norder, singular_values.data(), rcond_limit,
            &rank, WORK.data(), LWORK, &info);

        for (int iorder = 0; iorder < norder; ++iorder) 
        {
            double * temp = &full_templates[iorder * scanlen];
            for (int isignal = 0; isignal < nsignal; ++isignal) 
            {
                double * signal = &signals[isignal][start];
                double amp = proj[iorder + isignal * norder];
                if (toast::is_aligned(signal) && toast::is_aligned(temp)) 
                {
                    for (size_t i = 0; i < scanlen; ++i) 
                    {
                        signal[i] -= amp * temp[i];
                    }
                } 
                else 
                {
                    for (size_t i = 0; i < scanlen; ++i) 
                    {
                        signal[i] -= amp * temp[i];
                    }
                }
            }
        }
    }
}
"""