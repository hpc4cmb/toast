# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..hwp_utils import (
    hwpss_sincos_buffer,
    hwpss_compute_coeff_covariance,
    hwpss_compute_coeff,
    hwpss_build_model,
)
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Int, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


@trait_docs
class HWPSynchronousFilter(Operator):
    """Operator that filters HWP synchronous signal.

    This fits and subtracts a Maxipol / EBEX style model for the HWPSS.
    The 2f component is optionally used to build a dictionary of relative calibration
    factors.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key",
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

    hwp_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to use when adding flags based on HWP filter failures.",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle, allow_none=True, help="Observation shared key for HWP angle"
    )

    harmonics = Int(9, help="Number of harmonics to consider in the expansion")

    relcal = Unicode(
        None,
        allow_none=True,
        help="Build a relative calibration dictionary in this observation key",
    )

    fill_gaps = Bool(False, help="If True, fit a simple line across gaps")

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

    @traitlets.validate("harmonics")
    def _check_harmonics(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Harmonics should be a non-negative integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        for ob in data.obs:
            timer = Timer()
            timer.start()

            # Get the timestamps relative to the observation start
            reltime = np.array(ob.shared[self.times].data, copy=True)
            time_offset = reltime[0]
            reltime -= time_offset

            if self.hwp_angle not in ob.shared:
                # Nothing to do, but if a relative calibration dictionary
                # was requested, make a fake one.
                if self.relcal is not None:
                    ob[self.relcal] = {x: 1.0 for x in ob.local_detectors}
                continue

            # The shared flags
            if self.shared_flags is None:
                flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
            else:
                flags = ob.shared[self.shared_flags].data & self.shared_flag_mask

            # Compute flags for samples where the hwp is stopped
            stopped = self._stopped_flags(ob)
            flags |= stopped

            # Build the products common to all detectors
            sincos = hwpss_sincos_buffer(
                ob.shared[self.hwp_angle].data,
                flags,
                self.harmonics,
                comm=ob.comm.comm_group,
            )
            msg = f"HWPSS Filter {ob.name}: built sincos buffer in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            obs_cov = hwpss_compute_coeff_covariance(
                reltime,
                flags,
                sincos,
                comm=ob.comm.comm_group,
            )
            if obs_cov is None:
                msg = f"HWPSS Filter {ob.name} failed to compute coefficient"
                msg += " covariance.  Skipping observation."
                log.warning(msg)
                continue

            msg = f"HWPSS Filter {ob.name}: built coefficient covariance in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            # Local detectors we are considering
            local_dets = ob.select_local_detectors(flagmask=self.det_mask)

            # The 2f magnitude
            mag = np.zeros(len(local_dets))
            h_index = 1

            # Fit coefficients
            coeff = dict()

            # Save the detector sample flags for later use
            det_flags = dict()
            for idet, det in enumerate(local_dets):
                if self.det_flags is None:
                    det_flags[det] = flags
                else:
                    det_flags[det] = np.copy(ob.detdata[self.det_flags][det])
                    det_flags[det] &= self.det_flag_mask
                    det_flags[det] |= flags
                good_samp = det_flags[det] == 0
                sig = ob.detdata[self.det_data][det]
                dc = np.mean(sig[good_samp])
                sig -= dc

                coeff[det] = hwpss_compute_coeff(
                    sig,
                    det_flags[det],
                    reltime,
                    sincos,
                    obs_cov[0],
                    obs_cov[1],
                )
                mag[idet] = np.sqrt(
                    coeff[det][4 * h_index + 0] ** 2 + coeff[det][4 * h_index + 2] ** 2
                )
                print(f"----- {ob.name} -----")
                for h in range(self.harmonics):
                    print(
                        f"{det}:  {coeff[det][4 * h + 0]} {coeff[det][4 * h + 1]} {coeff[det][4 * h + 2]} {coeff[det][4 * h + 3]}"
                    )
            msg = f"HWPSS Filter {ob.name}: compute detector coefficients in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            # Cut detectors with outlier values
            new_flags = dict()
            cur_flags = ob.local_detector_flags
            det_indx = {y: x for x, y in enumerate(local_dets)}
            good_dets = [True for x in mag]
            n_cut = 1
            while n_cut > 0:
                n_cut = 0
                mn = np.mean(mag[good_dets])
                std = np.std(mag[good_dets])
                for idet, det in enumerate(local_dets):
                    if not good_dets[idet]:
                        continue
                    if np.absolute(mag[idet] - mn) > 5 * std:
                        good_dets[idet] = False
                        new_flags[det] = cur_flags[det] | self.hwp_flag_mask
                        n_cut += 1
            ob.update_local_detector_flags(new_flags)
            n_bad = len(good_dets) - np.count_nonzero(good_dets)
            msg = f"HWPSS Filter {ob.name}: cut {n_bad} 2f magnitude outliers in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            # Build relative calibration factors
            if self.relcal is not None:
                relcal = dict()
                unity = np.mean(mag[good_dets])
                for idet, det in enumerate(local_dets):
                    if good_dets[idet]:
                        relcal[det] = unity / mag[idet]
                ob[self.relcal] = relcal
                msg = f"HWPSS Filter {ob.name}: built relcal in"
                log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            # Subtract the model.  Use the shared flags for this.  Note that we
            # already subtracted the DC level above.  Also update detector flags.
            for idet, det in enumerate(local_dets):
                if not good_dets[idet]:
                    continue
                model = hwpss_build_model(reltime, flags, sincos, coeff[det])
                ob.detdata[self.det_data][det] -= model
                # Update flags
                ob.detdata[self.det_flags][det] |= det_flags[det]
                # Note: we could do something fancier and fit a trend, but just
                # trying to ensure that the flagged samples are not crazy relative to
                # the template subtracted data.
                if self.fill_gaps:
                    self._fill_gaps(ob, det, det_flags[det])

    def _stopped_flags(self, obs):
        hdata = obs.shared[self.hwp_angle].data
        hdiff = np.diff(hdata)
        hdiffmed = np.median(hdiff)
        stopped = np.zeros(len(hdata), dtype=np.uint8)
        stopped[:-1] = np.absolute(hdiff) < 1e-6 * hdiffmed
        stopped[-1] = stopped[-2]
        stopped *= self.hwp_flag_mask
        return stopped

    def _fill_gaps(self, obs, det, flags):
        # Fill gaps with a line, just to kill large artifacts in flagged
        # regions after removal of the HWPSS.  This is mostly just for visualization.
        # Downstream codes should ignore these flagged samples anyway.
        sig = obs.detdata[self.det_data][det]
        flag_indx = np.arange(len(flags), dtype=np.int64)[np.nonzero(flags)]
        flag_groups = np.split(flag_indx, np.where(np.diff(flag_indx) != 1)[0] + 1)
        for grp in flag_groups:
            if len(grp) == 0:
                continue
            bad_first = grp[0]
            bad_last = grp[-1]
            if bad_first == 0:
                # Starting bad samples, do nothing
                continue
            if bad_last == len(flags):
                # Ending bad samples, do nothing
                continue
            int_first = bad_first - 1
            int_last = bad_last + 1
            sig[bad_first:bad_last] = np.interp(
                np.arange(bad_first, bad_last, dtype=np.int32),
                [int_first, int_last],
                [sig[int_first], sig[int_last]],
            )

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # Note that the hwp_angle is not strictly required- this
        # is just a no-op.
        req = {
            "shared": [self.times],
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {
            "detdata": [self.det_data],
        }
        return prov
