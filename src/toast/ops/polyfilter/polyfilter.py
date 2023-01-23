# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
import warnings
from time import time

import numpy as np
import traitlets
from astropy import units as u

from ... import qarray as qa
from ..._libtoast import subtract_mean, sum_detectors
from ...accelerator import ImplementationType
from ...mpi import MPI, Comm, MPI_Comm, use_mpi
from ...observation import default_values as defaults
from ...timing import function_timer
from ...traits import Bool, Dict, Instance, Int, Quantity, Unicode, UseEnum, trait_docs
from ...utils import (
    AlignedF64,
    AlignedU8,
    Environment,
    GlobalTimers,
    Logger,
    Timer,
    dtype_to_aligned,
)
from ..operator import Operator
from .kernels import filter_poly2D, filter_polynomial

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
        defaults.det_mask_invalid | defaults.det_mask_processing,
        help="Bit mask value for optional detector flagging",
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
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        gt = GlobalTimers.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

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
                x, y, z = qa.rotate(temp_ob.telescope.focalplane[det]["quat"], ZAXIS)
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

            for det in detectors:
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

            views = temp_ob.intervals[self.view]
            for iview, view in enumerate(views):
                nsample = view.last - view.first + 1
                vslice = slice(view.first, view.last + 1)

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
                    shared_flags = temp_ob.shared[self.shared_flags][vslice]
                    shared_mask = (shared_flags & self.shared_flag_mask) == 0
                else:
                    shared_mask = np.ones(nsample, dtype=bool)

                for idet, det in enumerate(temp_ob.local_detectors):
                    if det not in detector_index:
                        continue
                    ind_det = detector_index[det]
                    ind_group = group_index[det]
                    det_groups[ind_det] = ind_group

                    signal = temp_ob.detdata[self.det_data][idet, vslice]
                    if self.det_flags is not None:
                        det_flags = temp_ob.detdata[self.det_flags][idet, vslice]
                        det_mask = (det_flags & self.det_flag_mask) == 0
                        mask = np.logical_and(shared_mask, det_mask)
                    else:
                        mask = shared_mask

                    templates[ind_det, :] = detector_templates[ind_det]
                    masks[:, ind_det] = mask
                    signals[:, ind_det] = signal * mask

                gt.stop("Poly2D:  Accumulate templates")

                gt.start("Poly2D:  Solve templates")
                filter_poly2D(
                    det_groups,
                    templates,
                    signals,
                    masks,
                    coeff,
                    impl=implementation,
                    use_accel=use_accel,
                )
                gt.stop("Poly2D:  Solve templates")

                gt.start("Poly2D:  Update detector flags")

                for igroup in range(ngroup):
                    local_dets = temp_ob.local_detectors
                    dets_in_group = np.zeros(len(local_dets), dtype=bool)
                    for idet, det in enumerate(local_dets):
                        if group_index[det] == igroup:
                            dets_in_group[idet] = True
                    if not np.any(dets_in_group):
                        continue
                    if self.det_flags is not None:
                        sample_flags = np.ones(
                            len(local_dets),
                            dtype=temp_ob.detdata[self.det_flags].dtype,
                        )
                        sample_flags *= self.poly_flag_mask
                        sample_flags *= dets_in_group
                        for isample in range(nsample):
                            if np.all(coeff[isample, igroup] == 0):
                                temp_ob.detdata[self.det_flags][
                                    :, view.first + isample
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
                    signal = temp_ob.detdata[self.det_data][idet, vslice]
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
        defaults.det_mask_invalid | defaults.det_mask_processing,
        help="Bit mask value for optional detector flagging",
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
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

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
                    filter_polynomial(
                        self.order,
                        last_flags,
                        signals,
                        local_starts,
                        local_stops,
                        impl=implementation,
                        use_accel=use_accel,
                    )
                    signals = [signal]
                last_flags = flags.copy()

            if len(signals) > 0:
                filter_polynomial(
                    self.order,
                    last_flags,
                    signals,
                    local_starts,
                    local_stops,
                    impl=implementation,
                    use_accel=use_accel,
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
        defaults.det_mask_invalid | defaults.det_mask_processing,
        help="Bit mask value for optional detector flagging",
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
                comm=temp_ob.comm.comm_group,
                timer=timer,
            )
            # Redistribute this temporary observation to be distributed by sample sets
            temp_ob.redistribute(1, times=self.times, override_sample_sets=None)
            log.debug_rank(
                f"{data.comm.group:4} : Redistributed observation in",
                comm=temp_ob.comm.comm_group,
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
                comm=temp_ob.comm.comm_group,
                timer=timer,
            )
            # Copy data to original observation
            obs.detdata[self.det_data][:] = temp_ob.detdata[self.det_data][:]
            log.debug_rank(
                f"{data.comm.group:4} : Copied observation data in",
                comm=temp_ob.comm.comm_group,
                timer=timer,
            )
        return

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
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

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

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
