# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
import os

import numpy as np
import scipy.interpolate
import traitlets
from astropy import units as u

from ..hwp_utils import (
    hwpss_build_model,
    hwpss_build_interpolated_model,
    hwpss_compute_coeff,
    hwpss_compute_coeff_step2f,
    hwpss_sincos_buffer,
)
from ..intervals import regular_intervals
from ..mpi import MPI, flatten
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, flagged_noise_fill
from .operator import Operator


@trait_docs
class HWPSynchronousModel(Operator):
    """Operator that models and removes HWP synchronous signal.

    This fits and optionally subtracts a Maxipol / EBEX style model for the HWPSS.
    See the details in `toast.hwp_utils.hwpss_compute_coeff()`.

    If estimating the HWPSS model in chunks over the observation, the 2F amplitude
    can be used as a proxy for the time-varying calibration drift within a detector.

    If a chunkwise HWPSS model is calculated, the template coefficients are smoothly
    interpolated between chunks to build a dynamic model of the HWPSS at each sample.

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

    subtract_model = Bool(False, help="Subtract the model from the input data")

    save_model = Unicode(
        None, allow_none=True, help="Save the model to this observation key"
    )

    chunk_view = Unicode(
        None,
        allow_none=True,
        help="The intervals over which to independently compute the HWPSS template",
    )

    chunk_time = Quantity(
        None,
        allow_none=True,
        help="The overlapping time chunks over which to compute the HWPSS template",
    )

    calibrate = Bool(
        False,
        help="If True, estimate the calibration drift from 2F variations and remove",
    )

    fill_gaps = Bool(False, help="If True, fill gaps with a simple noise model")

    debug = Unicode(
        None,
        allow_none=True,
        help="Path to directory for generating debug plots",
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

        if self.chunk_view is not None and self.chunk_time is not None:
            msg = "Only one of chunk_view and chunk_time can be used"
            raise RuntimeError(msg)

        if not self.subtract_model and (self.save_model is None) and not self.calibrate:
            msg = "Nothing to do.  You should enable at least one of the options"
            msg += " to subtract or save the model or to remove calibration drift."
            raise RuntimeError(msg)

        if self.calibrate and (self.chunk_time is None and self.chunk_view is None):
            msg = "Cannot compute calibration drift without using chunks"
            log.warning_rank(msg, comm=data.comm.comm_world)

        if self.debug is not None:
            if data.comm.world_rank == 0:
                os.makedirs(self.debug)
            if data.comm.comm_world is not None:
                data.comm.comm_world.barrier()

        for ob in data.obs:
            timer = Timer()
            timer.start()

            if not ob.is_distributed_by_detector:
                msg = f"{ob.name} is not distributed by detector"
                raise RuntimeError(msg)

            if self.hwp_angle not in ob.shared:
                # Nothing to do
                msg = f"{ob.name} has no '{self.hwp_angle}' field, skipping"
                log.warning_rank(msg, comm=data.comm.comm_group)
                continue

            # Compute quantities we need for all detectors and which we
            # might re-use for overlapping chunks.

            # Local detectors we are considering
            local_dets = ob.select_local_detectors(flagmask=self.det_mask)
            n_dets = len(local_dets)

            # Get the timestamps relative to the observation start
            reltime = ob.shared[self.times].data.copy()
            time_offset = reltime[0]
            reltime -= time_offset

            # Compute the properties of the chunks we are using
            chunks = self._compute_chunking(ob, reltime)
            n_chunk = len(chunks)

            # Compute shared and per-detector flags.  These already have
            # masks applied and have values of either zero or one.
            sh_flags, det_flags = self._compute_flags(ob, local_dets)

            if np.count_nonzero(sh_flags) == ob.n_local_samples:
                msg = f"{ob.name} has all samples cut by shared flags and "
                msg += "unstable HWP motion"
                log.warning_rank(msg, comm=data.comm.comm_group)
                continue

            msg = f"HWPSS Model {ob.name}: compute flags and chunking in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            # Trig quantities of the HWP angle
            sincos = hwpss_sincos_buffer(
                ob.shared[self.hwp_angle].data,
                sh_flags,
                self.harmonics,
                comm=ob.comm.comm_group,
            )
            msg = f"HWPSS Model {ob.name}: built sincos buffer in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            # Calibrate if needed
            self._calibrate(ob, reltime, local_dets, det_flags, chunks, sincos)
            msg = f"HWPSS Model {ob.name}: finished gain drift calibration in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            # Solve for our chunkwise HWPSS model
            n_coeff = 2 * self.harmonics
            coeff = np.zeros((n_dets, n_coeff, n_chunk), dtype=np.float64)
            coeff_flags = np.zeros((n_dets, n_chunk), dtype=bool)

            self._fit_chunks(
                ob,
                local_dets,
                chunks,
                sincos,
                det_flags,
                coeff,
                coeff_flags,
            )

            # Optionally save the model coefficients
            if self.save_model is not None:
                self._store_model(ob, local_dets, chunks, coeff, coeff_flags)

            msg = f"HWPSS Model {ob.name}: compute detector coefficients in"
            log.debug_rank(msg, comm=ob.comm.comm_group, timer=timer)

            if not self.subtract_model:
                # We are done
                continue

            # For all detectors, interpolate the model to the sample rate and subtract.
            self._apply_model(
                ob,
                reltime,
                local_dets,
                sh_flags,
                det_flags,
                chunks,
                sincos,
                coeff,
                coeff_flags,
            )

            msg = f"HWPSS Model {ob.name}: interpolate and subtract model in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

    def _calibrate(self, obs, reltime, local_dets, det_flags, chunks, sincos):
        if not self.calibrate:
            # We are not computing calibration drift
            return
        if self.chunk_time is None and self.chunk_view is None:
            # We have no chunks / steps over which to estimate the drift
            return

        n_dets = len(local_dets)
        n_chunk = len(chunks)
        n_coeff = 2 * self.harmonics
        coeff = np.zeros((n_dets, n_coeff, n_chunk), dtype=np.float64)
        coeff_flags = np.zeros((n_dets, n_chunk), dtype=bool)

        self._fit_chunks(
            obs,
            local_dets,
            chunks,
            sincos,
            det_flags,
            coeff,
            coeff_flags,
        )

        # Get the times and values for all chunks / steps.  Extract the 2F
        # magnitudes and interpolate.
        for idet, det in enumerate(local_dets):
            cal_time = list()
            cal_val = list()
            for ch in range(n_chunk):
                if coeff_flags[idet, ch]:
                    continue
                val = np.sqrt(coeff[idet, 2, ch] ** 2 + coeff[idet, 3, ch] ** 2)
                cal_time.append(chunks[ch]["time"])
                cal_val.append(val)
            cal_time = np.array(cal_time)
            cal_val = np.array(cal_val)
            cal = np.interp(reltime, cal_time, cal_val)
            cal_mean = np.mean(cal)
            cal = 1.0 + (cal - cal_mean)
            obs.detdata[self.det_data][det] /= cal

    def _fit_chunks(
        self,
        obs,
        dets,
        chunks,
        sincos,
        det_flags,
        coeff,
        coeff_flags,
        step_size=None,
    ):
        log = Logger.get()
        for idet, det in enumerate(dets):
            guess = None
            for ichunk, chunk in enumerate(chunks):
                start = chunk["start"]
                end = chunk["end"]
                slc = slice(start, end, 1)

                good_samp = det_flags[det][slc] == 0
                if np.count_nonzero(good_samp) < coeff.shape[1]:
                    # Not very many good samples, set coefficients to zero
                    msg = f"HWPSS Model {obs.name}[{ichunk}] {det}: insufficient good "
                    msg += "samples, setting coefficients to zero"
                    log.verbose(msg)
                    coeff[idet, :, ichunk] = 0
                    coeff_flags[idet, ichunk] = 1
                    continue
                sig = obs.detdata[self.det_data][det, slc].copy()
                dc = np.mean(sig[good_samp])
                sig -= dc
                if step_size is None:
                    cf = hwpss_compute_coeff(
                        sincos[slc],
                        sig,
                        det_flags[det][slc],
                        guess=guess,
                    )
                else:
                    cf = hwpss_compute_coeff_step2f(
                        sincos[slc],
                        sig,
                        det_flags[det][slc],
                        step_size=step_size,
                        guess=guess,
                    )
                coeff[idet, :, ichunk] = cf
                guess = cf

    def _apply_model(
        self,
        obs,
        reltime,
        local_dets,
        sh_flags,
        det_flags,
        chunks,
        sincos,
        coeff,
        coeff_flags,
    ):
        log = Logger.get()
        n_samp = len(reltime)

        # Sample rate
        rate = obs.telescope.focalplane.sample_rate.to_value(u.Hz)

        for idet, det in enumerate(local_dets):
            if len(chunks) == 1:
                if coeff_flags[0] != 0:
                    msg = f"{obs.name}[{det}]: only one chunk, which is flagged"
                    log.warning(msg)
                    # Flag this detector
                    current = obs.local_detector_flags[det]
                    obs.update_local_detector_flags({det: current | self.hwp_flag_mask})
                    continue
                det_coeff = coeff[idet, :, 0]
                model = hwpss_build_model(sincos, sh_flags, det_coeff)
                plt_coeff = coeff[idet]
            else:
                # Compute the chunks we are using
                ch_wts = list()
                ch_indx = list()
                ch_coeff = list()
                avg_size = np.mean([(x["end"] - x["start"]) for x in chunks])
                for ich, chk in enumerate(chunks):
                    if coeff_flags[idet, ich]:
                        continue
                    sz = chk["end"] - chk["start"]
                    ch_indx.append(chk["start"] + sz // 2)
                    invwt = 1.0 + avg_size - sz
                    ch_wts.append(1.0 / invwt)
                    ch_coeff.append(coeff[idet][:, ich].reshape((-1, 1)))
                if len(ch_coeff) == 0:
                    # All chunks are flagged
                    msg = f"{obs.name}[{det}]: all chunks are flagged"
                    log.warning(msg)
                    continue
                ch_indx = np.array(ch_indx, dtype=np.int64)
                ch_wts = np.array(ch_wts, dtype=np.float64)
                ch_coeff = np.hstack(ch_coeff).astype(np.float64)
                print(f"DBG {det} ch_coeff = {ch_coeff}", flush=True)
                plt_coeff = ch_coeff

                model = hwpss_build_interpolated_model(
                    sincos, sh_flags, ch_coeff, ch_indx, coeff_wts=ch_wts
                )

            self._plot_model(
                obs,
                det,
                reltime,
                sincos,
                sh_flags,
                det_flags,
                model,
                chunks,
                plt_coeff,
                0,
                n_samp,
            )
            self._plot_model(
                obs,
                det,
                reltime,
                sincos,
                sh_flags,
                det_flags,
                model,
                chunks,
                plt_coeff,
                n_samp // 2 - 500,
                n_samp // 2 + 500,
            )

            # Subtract model from good samples
            good = det_flags[det] == 0
            obs.detdata[self.det_data][det][good] -= model[good]
            dc = np.mean(obs.detdata[self.det_data][det][good])
            obs.detdata[self.det_data][det][good] -= dc

            if self.fill_gaps:
                # 1 second buffer
                buffer = int(rate)
                flagged_noise_fill(
                    obs.detdata[self.det_data][det],
                    det_flags[det],
                    buffer,
                    poly_order=1,
                )

            # Update sample flags
            obs.detdata[self.det_flags][det] |= det_flags[det] * self.hwp_flag_mask

    def _plot_model(
        self,
        obs,
        det_name,
        reltime,
        sincos,
        sh_flags,
        det_flags,
        model,
        chunks,
        chunk_coeff,
        first,
        last,
    ):
        if self.debug is None:
            return
        import matplotlib.pyplot as plt

        slc = slice(first, last, 1)
        # If we are plotting per-chunk quantities, find the overlap of every chunk
        # with our plot range
        chunk_slc = None
        if len(chunks) > 1:
            chunk_slc = list()
            for ich, chk in enumerate(chunks):
                ch_start = chk["start"]
                ch_end = chk["end"]
                ch_size = ch_end - ch_start
                prp = dict()
                prp["abs_slc"] = slice(ch_start, ch_end, 1)
                if ch_start < last and ch_end > first:
                    # some overlap
                    if ch_start < first:
                        ch_first = first - ch_start
                        plt_first = first
                    else:
                        ch_first = 0
                        plt_first = ch_start
                    if ch_end > last:
                        ch_last = ch_size - (ch_end - last)
                        plt_last = last
                    else:
                        ch_last = ch_size
                        plt_last = ch_end

                    prp["ch_overlap"] = slice(int(ch_first), int(ch_last), 1)
                    prp["plt_overlap"] = slice(int(plt_first), int(plt_last), 1)
                else:
                    prp["ch_overlap"] = None
                    prp["plt_overlap"] = None
                chunk_slc.append(prp)
        cmap = plt.get_cmap("tab10")
        plt_file = os.path.join(
            self.debug,
            f"{obs.name}_model_{det_name}_{first}-{last}.png",
        )
        fig = plt.figure(figsize=(12, 18), dpi=100)
        ax = fig.add_subplot(3, 1, 1, aspect="auto")
        # Plot original signal
        sig = obs.detdata[self.det_data][det_name, slc].copy()
        sig -= np.mean(sig)
        ax.plot(
            reltime[slc],
            sig,
            color="black",
            label=f"Signal {det_name}",
        )
        # Plot per chunk models
        if len(chunks) > 1:
            for ich, chk in enumerate(chunks):
                if chunk_slc[ich]["ch_overlap"] is None:
                    # No overlap
                    continue
                ch_coeff = chunk_coeff[:, ich]
                if np.count_nonzero(ch_coeff) == 0:
                    continue
                ch_model = hwpss_build_model(
                    sincos[chunk_slc[ich]["abs_slc"]],
                    sh_flags[chunk_slc[ich]["abs_slc"]],
                    ch_coeff,
                    times=reltime[chunk_slc[ich]["abs_slc"]],
                    time_drift=self.time_drift,
                )
                ax.plot(
                    reltime[chunk_slc[ich]["plt_overlap"]],
                    ch_model[chunk_slc[ich]["ch_overlap"]],
                    color=cmap(ich),
                    label=f"Model {det_name}",
                )
        # Plot full model
        ax.plot(
            reltime[slc],
            model[slc],
            color="red",
            label=f"Model {det_name}",
        )
        ax.legend(loc="best")

        # Plot residual
        ax = fig.add_subplot(3, 1, 2, aspect="auto")
        ax.plot(
            reltime[slc],
            model[slc] - sig,
            color="blue",
            label=f"Residual {det_name}",
        )
        ax.legend(loc="best")

        cmap = plt.get_cmap("tab10")
        ax = fig.add_subplot(3, 1, 3, aspect="auto")
        # Plot flags
        ax.plot(
            reltime[slc],
            det_flags[det_name][slc],
            color="black",
            label=f"Flags {det_name}",
        )
        # Plot chunk boundaries
        if len(chunks) > 1:
            incr = 1 / (len(chunks) + 1)
            for ich, chk in enumerate(chunks):
                if chunk_slc[ich]["ch_overlap"] is None:
                    # No overlap
                    continue
                ax.plot(
                    reltime[chunk_slc[ich]["plt_overlap"]],
                    incr * ich * np.ones_like(reltime[chunk_slc[ich]["plt_overlap"]]),
                    color=cmap(ich),
                    linewidth=3,
                    label=f"Chunk {ich}",
                )
        ax.legend(loc="best")
        fig.suptitle(f"Obs {obs.name} Samples {first} - {last}")
        fig.savefig(plt_file)
        plt.close(fig)

    def _store_model(self, obs, dets, chunks, coeff, coeff_flags):
        log = Logger.get()
        if self.save_model in obs:
            msg = f"observation {obs.name} already has something at "
            msg += f"key {self.save_model}.  Overwriting."
            log.warning(msg)
        # Repackage the coefficients and chunk information
        ob_start = obs.shared[self.times].data[0]
        model = list()
        for ichk, chk in enumerate(chunks):
            props = {
                "start": chk["start"],
                "end": chk["end"],
                "time": ob_start + chk["time"],
                "flag": coeff_flags[ichk],
            }
            props["dets"] = dict()
            for idet, det in enumerate(dets):
                props["dets"][det] = np.array(coeff[idet, :, ichk])
            model.append(props)
        obs[self.save_model] = model

    def _compute_chunking(self, obs, reltime):
        chunks = list()
        if self.chunk_view is None:
            if self.chunk_time is None:
                # One chunk for the whole observation
                chunks.append(
                    {
                        "start": 0,
                        "end": obs.n_local_samples,
                        "time": reltime[obs.n_local_samples // 2],
                    }
                )
            else:
                # Overlapping chunks
                duration = reltime[-1] - reltime[0]
                non_overlap = int(duration / self.chunk_time.to_value(u.second))
                # Adjust the chunk size to evenly divide into the obs range
                adjusted_time = duration / non_overlap
                # Convert to samples.  Round up so that the final chunk has
                # a few less samples rather than having a short chunk at the
                # end.
                rate = obs.telescope.focalplane.sample_rate.to_value(u.Hz)
                chunk_samples = int(adjusted_time * rate + 0.5)
                half_chunk = chunk_samples // 2
                ch_start = 0
                for ch in range(non_overlap):
                    ch_mid = ch_start + half_chunk
                    chunks.append(
                        {
                            "start": ch_start,
                            "end": ch_start + chunk_samples,
                            "time": reltime[ch_mid],
                        }
                    )
                    if ch != non_overlap - 1:
                        # Add the overlapping chunk
                        chunks.append(
                            {
                                "start": ch_start + half_chunk,
                                "end": ch_start + half_chunk + chunk_samples,
                                "time": reltime[ch_start + chunk_samples],
                            }
                        )
                    ch_start += chunk_samples
        else:
            # Use the specified interval list for the chunks.  Cut any
            # chunks that are tiny.
            for intr in obs.intervals[self.chunk_view]:
                ch_size = intr.last - intr.first
                ch_mid = intr.first + ch_size // 2
                if ch_size > 10 * self.harmonics:
                    chunks.append(
                        {
                            "start": intr.first,
                            "end": intr.last,
                            "time": reltime[ch_mid],
                        }
                    )
        return chunks

    def _compute_flags(self, obs, dets):
        # The shared flags
        if self.shared_flags is None:
            shared_flags = np.zeros(obs.n_local_samples, dtype=np.uint8)
        else:
            shared_flags = obs.shared[self.shared_flags].data.copy()
            shared_flags &= self.shared_flag_mask

        # Compute flags for samples where the hwp is stopped
        stopped = self._stopped_flags(obs)
        shared_flags |= stopped

        # If we are chunking based on intervals, flag the regions between valid
        # intervals.
        if self.chunk_view is not None:
            not_modelled = np.ones_like(shared_flags)
            for intr in obs.intervals[self.chunk_view]:
                not_modelled[intr.first : intr.last] = 0
            shared_flags |= not_modelled

        # Per-detector flags.  We merge in the shared flags to these since the
        # detector flags will be written out at the end if the model is subtracted.
        det_flags = dict()
        for idet, det in enumerate(dets):
            if self.det_flags is None:
                det_flags[det] = shared_flags
            else:
                det_flags[det] = obs.detdata[self.det_flags][det].copy()
                det_flags[det] &= self.det_flag_mask
                det_flags[det] |= shared_flags
        return (shared_flags, det_flags)

    def _stopped_flags(self, obs):
        hdata = np.unwrap(obs.shared[self.hwp_angle].data, period=2 * np.pi)
        hvel = np.gradient(hdata)
        moving = np.absolute(hvel) > 1.0e-6
        nominal = np.median(hvel[moving])
        abs_nominal = np.absolute(nominal)
        unstable = np.absolute(hvel) - abs_nominal > 1.0e-3 * abs_nominal
        stopped = np.array(unstable, dtype=np.uint8)
        return stopped

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
            "meta": [],
            "detdata": [self.det_data],
        }
        if self.relcal_continuous is not None:
            prov["detdata"].append(self.relcal_continuous)
        if self.save_model is not None:
            prov["meta"].append(self.save_model)
        if self.relcal_fixed is not None:
            prov["meta"].append(self.relcal_fixed)
        return prov
