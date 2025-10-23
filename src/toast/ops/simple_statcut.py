# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
from collections import namedtuple

import numpy as np
import traitlets
from astropy import units as u
from scipy.signal import medfilt
import scipy.stats as ss

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
class SimpleStatCut(Operator):
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

    medfilt_kernel_size = Int(
        101,
        help="Median filter kernel width.  Either 0 (full interval) "
        "or a positive odd number",
    )

    limit = Float(
        3.0,
        help="Distance from median to be considered pathological",
    )

    out = Unicode(
        "stats",
        help="Observation key for statistics",
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

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm.comm_group
        timer = Timer()
        timer.start()

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)

            views = ob.intervals[self.view]
            focalplane = ob.telescope.focalplane

            all_local_dets = ob.select_local_detectors(flagmask=self.det_mask)

            if len(all_local_dets) > 0 and all_local_dets[0].startswith("demod"):
                demod = True
            else:
                demod = False
            if comm is not None:
                demod = comm.allreduce(demod, op=MPI.LOR)
            if demod:
                # Demodulated case. Process I/Q/U detectors separately
                prefixes = ["demod0", "demod4r", "demod4i"]
            else:
                # Standad processing
                prefixes = [""]

            shared_flags = ob.shared[self.shared_flags].data & self.shared_flag_mask

            cut_obs = set()
            ndet_obs = len(all_local_dets)
            for prefix in prefixes:
                local_dets = []
                for det in all_local_dets:
                    if det.startswith(prefix):
                        local_dets.append(det)
                local_dets = np.array(local_dets)
                ndet = len(local_dets)
                bad_dets = np.zeros(ndet, dtype=bool)

                local_rms = np.zeros(ndet)
                local_skew = np.zeros(ndet)
                local_kurtosis = np.zeros(ndet)
                for idet, det in enumerate(local_dets):
                    sig = ob.detdata[self.det_data][det].copy()
                    det_flags = ob.detdata[self.det_flags][det] & self.det_flag_mask
                    good = np.logical_and(shared_flags == 0, det_flags == 0)
                    nsample = sig.size
                    w = self.medfilt_kernel_size
                    if w > 0 and nsample > 2 * w:
                        # Remove the running median
                        sig[w:-w] -= medfilt(sig, kernel_size=w)[w:-w]
                        # Special treatment for the ends
                        sig[:w] -= np.median(sig[:w])
                        sig[-w:] -= np.median(sig[-w:])
                    else:
                        sig -= np.median(sig)
                    local_rms[idet] = np.std(sig[good])
                    local_skew[idet] = ss.skew(sig[good])
                    local_kurtosis[idet] = ss.kurtosis(sig[good])

                if comm is not None:
                    all_dets = np.hstack(comm.allgather(local_dets))
                    all_rms = np.hstack(comm.allgather(local_rms))
                    all_skew = np.hstack(comm.allgather(local_skew))
                    all_kurtosis = np.hstack(comm.allgather(local_kurtosis))
                else:
                    all_dets = local_dets
                    all_rms = local_rms
                    all_skew = local_skew
                    all_kurtosis = local_kurtosis

                if len(local_dets) == 0:
                    continue

                stat_dict = {}
                Stats = namedtuple("Stats", ["rms", "skew", "kurtosis"])
                for det, rms, skew, kurtosis in zip(
                        all_dets, all_rms, all_skew, all_kurtosis
                ):
                    stat_dict[det] = Stats(rms, skew, kurtosis)
                if self.out not in ob:
                    ob[self.out] = {}
                ob["stats"].update(stat_dict)

                good = np.ones(len(all_dets), dtype=bool)
                for all_stat, local_stat in zip(
                        [all_rms, all_skew, all_kurtosis],
                        [local_rms, local_skew, local_kurtosis],
                ):
                    while True:
                        med = np.median(all_stat[good])
                        rms = np.std(all_stat[good])
                        bad = np.abs(all_stat - med) > rms * self.limit
                        if np.any(bad[good]):
                            good[bad] = False
                        else:
                            break
                    local_bad = np.abs(local_stat - med) > rms * self.limit
                    for det in local_dets[local_bad]:
                        ob.local_detector_flags[det] |= defaults.det_mask_invalid
                        cut_obs.add(det)
                        if prefix != "":
                            # Demodulated case, flag the associated pseudo detectors
                            for alt_prefix in ["demod0", "demod4r", "demod4i"]:
                                if prefix == alt_prefix:
                                    continue
                                alt_det = det.replace(prefix, alt_prefix)
                                if alt_det in ob.local_detector_flags:
                                    ob.local_detector_flags[alt_det] \
                                        |= defaults.det_mask_invalid
                                    cut_obs.add(alt_det)

            nbad_obs = len(cut_obs)
            if comm is not None:
                ndet_obs = comm.reduce(ndet_obs)
                nbad_obs = comm.reduce(nbad_obs)
            log.debug_rank(
                f"Flagged {nbad_obs} / {ndet_obs} additional detectors in "
                f"{ob.name} due to statistics in",
                comm=comm,
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
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return prov
