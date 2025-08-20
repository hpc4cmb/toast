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
    hwpss_compute_coeff,
    hwpss_compute_coeff_covariance,
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
    The time dependent drift term is optional.  See the details in
    `toast.hwp_utils.hwpss_compute_coeff_covariance()`.

    The 2f component of the model is optionally used to build a relative calibration
    between detectors, either as a fixed table per observation or as continuously
    varying factors.

    The HWPSS model can be constructed either with one set of template coefficients
    for the entire observation, or one set per time interval smoothly interpolated
    across the observation.

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

    relcal_fixed = Unicode(
        None,
        allow_none=True,
        help="Build a relative calibration dictionary in this observation key",
    )

    relcal_continuous = Unicode(
        None,
        allow_none=True,
        help="Build interpolated relative calibration timestreams",
    )

    relcal_cut_sigma = Float(
        5.0, help="Sigma cut for outlier rejection based on relative calibration"
    )

    time_drift = Bool(False, help="If True, include time drift terms in the model")

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

        if self.relcal_continuous is not None and self.relcal_fixed is not None:
            msg = "Only one of continuous and fixed relative calibration can be used"
            raise RuntimeError(msg)

        if self.chunk_view is not None and self.chunk_time is not None:
            msg = "Only one of chunk_view and chunk_time can be used"
            raise RuntimeError(msg)

        do_cal = self.relcal_continuous or self.relcal_fixed
        if not self.subtract_model and (self.save_model is None) and not do_cal:
            msg = "Nothing to do.  You should enable at least one of the options"
            msg += " to subtract or save the model or to generate calibrations."
            raise RuntimeError(msg)

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
                # Nothing to do, but if a relative calibration
                # was requested, make a fake one.
                if self.relcal_fixed is not None:
                    ob[self.relcal_fixed] = {x: 1.0 for x in ob.local_detectors}
                if self.relcal_continuous is not None:
                    ob.detdata.ensure(
                        self.relcal_continuous,
                        dtype=np.float32,
                        create_units=ob.detdata[self.det_data].units,
                    )
                    ob.detdata[self.relcal_continuous][:, :] = 1.0
                msg = f"{ob.name} has no '{self.hwp_angle}' field, skipping"
                log.warning_rank(msg, comm=data.comm.comm_group)
                continue

            # Compute quantities we need for all detectors and which we
            # might re-use for overlapping chunks.

            # Local detectors we are considering
            local_dets = ob.select_local_detectors(flagmask=self.det_mask)
            n_dets = len(local_dets)

            # Get the timestamps relative to the observation start
            reltime = np.array(ob.shared[self.times].data, copy=True)
            time_offset = reltime[0]
            reltime -= time_offset

            # Compute the properties of the chunks we are using
            chunks = self._compute_chunking(ob, reltime)
            n_chunk = len(chunks)

            # Compute shared and per-detector flags.  These already have
            # masks applied and have values of either zero or one.
            sh_flags, det_flags = self._compute_flags(ob, local_dets)

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

            # The coefficients for all detectors and chunks
            if self.time_drift:
                n_coeff = 4 * self.harmonics
            else:
                n_coeff = 2 * self.harmonics
            coeff = np.zeros((n_dets, n_coeff, n_chunk), dtype=np.float64)
            coeff_flags = np.zeros(n_chunk, dtype=np.uint8)

            for ichunk, chunk in enumerate(chunks):
                self._fit_chunk(
                    ob,
                    local_dets,
                    ichunk,
                    chunk["start"],
                    chunk["end"],
                    sincos,
                    sh_flags,
                    det_flags,
                    reltime,
                    coeff,
                    coeff_flags,
                )

            msg = f"HWPSS Model {ob.name}: fit model to all chunks in"
            log.debug_rank(msg, comm=data.comm.comm_group, timer=timer)

            if self.save_model is not None:
                self._store_model(ob, local_dets, chunks, coeff, coeff_flags)

            # Even if we are not saving a fixed relative calibration table, compute
            # the mean 2f magnitude in order to cut outlier detectors.  The
            # calibration factors are relative to the mean of the distribution
            # of good detectors values.
            mag_table = self._average_magnitude(local_dets, coeff, coeff_flags)
            good_dets, cal_center = self._cut_outliers(ob, mag_table)
            relcal_table = dict()
            for det in good_dets:
                relcal_table[det] = cal_center / mag_table[det]
            if self.relcal_fixed is not None:
                ob[self.relcal_fixed] = relcal_table

            # If we are generating relative calibration timestreams create that now.
            if self.relcal_continuous is not None:
                ob.detdata.ensure(
                    self.relcal_continuous,
                    dtype=np.float32,
                    create_units=ob.detdata[self.det_data].units,
                )
                ob.detdata[self.relcal_continuous][:, :] = 1.0

            # For each detector, compute the model and subtract from the data.  Also
            # compute the interpolated calibration timestream if requested.  We
            # assume that the model coefficients are slowly varying and just do a
            # linear interpolation.
            if not self.subtract_model and not self.relcal_continuous:
                # No need to compute the full time-domain templates
                continue

            good_check = set(good_dets)
            for idet, det in enumerate(local_dets):
                if det not in good_check:
                    continue
                model, det_mag = self._build_model(
                    ob,
                    reltime,
                    sincos,
                    sh_flags,
                    det_flags,
                    det,
                    mag_table[det],
                    chunks,
                    coeff[idet],
                    coeff_flags,
                )
                # Update flags
                ob.detdata[self.det_flags][det] |= det_flags[det] * self.hwp_flag_mask
                if model is None:
                    # The model construction failed due to flagged samples.  Nothing to
                    # subtract, since the detector has been flagged.
                    continue
                # Subtract model from good samples
                if self.subtract_model:
                    good = det_flags[det] == 0
                    ob.detdata[self.det_data][det][good] -= model[good]
                    dc = np.mean(ob.detdata[self.det_data][det][good])
                    ob.detdata[self.det_data][det][good] -= dc
                if self.fill_gaps:
                    rate = ob.telescope.focalplane.sample_rate.to_value(u.Hz)
                    # 1 second buffer
                    buffer = int(rate)
                    flagged_noise_fill(
                        ob.detdata[self.det_data][det],
                        det_flags[det],
                        buffer,
                        poly_order=1,
                    )
                if self.relcal_continuous is not None:
                    ob.detdata[self.relcal_continuous][det, :] = cal_center / det_mag

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
        fig = plt.figure(figsize=(12, 12), dpi=100)
        ax = fig.add_subplot(2, 1, 1, aspect="auto")
        # Plot original signal
        ax.plot(
            reltime[slc],
            obs.detdata[self.det_data][det_name, slc],
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

        cmap = plt.get_cmap("tab10")
        ax = fig.add_subplot(2, 1, 2, aspect="auto")
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

    def _build_model(
        self,
        obs,
        reltime,
        sincos,
        sh_flags,
        det_flags,
        det_name,
        det_mag,
        chunks,
        ch_coeff,
        coeff_flags,
        min_smooth=4,
    ):
        log = Logger.get()
        nsamp = len(reltime)
        if len(chunks) == 1:
            if coeff_flags[0] != 0:
                msg = f"{obs.name}[{det_name}]: only one chunk, which is flagged"
                log.warning(msg)
                # Flag this detector
                current = obs.local_detector_flags[det_name]
                obs.update_local_detector_flags(
                    {det_name: current | self.hwp_flag_mask}
                )
                return (None, None)
            det_coeff = ch_coeff[:, 0]
            model = hwpss_build_model(
                sincos,
                sh_flags,
                det_coeff,
                times=reltime,
                time_drift=self.time_drift,
            )
            self._plot_model(
                obs,
                det_name,
                reltime,
                sincos,
                sh_flags,
                det_flags,
                model,
                chunks,
                ch_coeff,
                0,
                nsamp,
            )
            self._plot_model(
                obs,
                det_name,
                reltime,
                sincos,
                sh_flags,
                det_flags,
                model,
                chunks,
                ch_coeff,
                nsamp // 2 - 500,
                nsamp // 2 + 500,
            )
        else:
            n_coeff = ch_coeff.shape[0]
            n_chunk = ch_coeff.shape[1]
            good_chunk = [
                np.count_nonzero(ch_coeff[:, x]) > 0 and coeff_flags[x] == 0
                for x in range(n_chunk)
            ]
            if np.count_nonzero(good_chunk) == 0:
                msg = f"{obs.name}[{det_name}]: All {len(good_chunk)} chunks"
                msg += f" are flagged."
                log.warning(msg)
                # Flag this detector
                current = obs.local_detector_flags[det_name]
                obs.update_local_detector_flags(
                    {det_name: current | self.hwp_flag_mask}
                )
                return (None, None)
            ch_times = np.array(
                [x["time"] for y, x in enumerate(chunks) if good_chunk[y]]
            )
            smoothing = max(n_chunk // 16, min_smooth)
            if smoothing >= n_chunk:
                msg = f"Only {n_chunk} chunks for interpolation. "
                msg += f"Reduce the split time or use different intervals"
                raise RuntimeError(msg)
            det_coeff = np.zeros((len(reltime), n_coeff), dtype=np.float64)
            for icoeff in range(n_coeff):
                coeff_spl = scipy.interpolate.splrep(
                    ch_times, ch_coeff[icoeff, good_chunk], s=smoothing
                )
                det_coeff[:, icoeff] = scipy.interpolate.splev(
                    reltime, coeff_spl, ext=0
                )
            model = hwpss_build_model(
                sincos,
                sh_flags,
                det_coeff,
                times=reltime,
                time_drift=self.time_drift,
            )
            if self.relcal_continuous is not None:
                if self.time_drift:
                    det_mag = np.sqrt(det_coeff[:, 4] ** 2 + det_coeff[:, 6] ** 2)
                else:
                    det_mag = np.sqrt(det_coeff[:, 2] ** 2 + det_coeff[:, 3] ** 2)
                det_mag[det_flags[det_name] != 0] = 1.0
                if self.debug is not None:
                    import matplotlib.pyplot as plt

                    def plot_2f(first, last):
                        slc = slice(first, last, 1)
                        plt_file = os.path.join(
                            self.debug,
                            f"{obs.name}_model_{det_name}_2f_{first}-{last}.png",
                        )
                        fig = plt.figure(figsize=(12, 12), dpi=100)
                        ax = fig.add_subplot(2, 1, 1, aspect="auto")
                        ax.plot(
                            reltime[slc],
                            det_mag[slc],
                            color="red",
                            label=f"Interpolated 2f Magnitude {det_name}",
                        )
                        if self.time_drift:
                            ch_mag = np.sqrt(
                                ch_coeff[4, good_chunk] ** 2
                                + ch_coeff[6, good_chunk] ** 2
                            )
                        else:
                            ch_mag = np.sqrt(
                                ch_coeff[2, good_chunk] ** 2
                                + ch_coeff[3, good_chunk] ** 2
                            )
                        ax.scatter(
                            ch_times,
                            ch_mag,
                            marker="*",
                            color="blue",
                            label="Estimated Chunk 2f Magnitude",
                        )
                        ax.legend(loc="best")
                        ax.set_xlim(left=reltime[first], right=reltime[last - 1])
                        ax = fig.add_subplot(2, 1, 2, aspect="auto")
                        ax.plot(
                            reltime[slc],
                            det_flags[det_name][slc],
                            color="black",
                            label=f"Flags {det_name}",
                        )
                        fig.suptitle(f"Obs {obs.name} Samples {first} - {last}")
                        fig.savefig(plt_file)
                        plt.close(fig)

                    plot_2f(0, nsamp)
                    plot_2f(nsamp // 2 - 500, nsamp // 2 + 500)
            self._plot_model(
                obs,
                det_name,
                reltime,
                sincos,
                sh_flags,
                det_flags,
                model,
                chunks,
                ch_coeff,
                0,
                nsamp,
            )
            self._plot_model(
                obs,
                det_name,
                reltime,
                sincos,
                sh_flags,
                det_flags,
                model,
                chunks,
                ch_coeff,
                nsamp // 2 - 500,
                nsamp // 2 + 500,
            )
        return model, det_mag

    def _store_model(self, obs, dets, chunks, coeff, coeff_flags):
        log = Logger.get()
        if self.save_model in obs:
            msg = "observation {obs.name} already has something at "
            msg += "key {self.save_model}.  Overwriting."
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

    def _cut_outliers(self, obs, det_mag):
        log = Logger.get()
        cut_timer = Timer()
        cut_timer.start()

        dets = list(det_mag.keys())
        mag = np.array([det_mag[x] for x in dets])

        # Communicate magnitudes
        all_dets = None
        all_mag = None
        if obs.comm_col is None:
            all_dets = dets
            all_mag = mag
        else:
            all_dets = obs.comm_col.gather(dets, root=0)
            all_mag = obs.comm_col.gather(mag, root=0)
            if obs.comm_col.rank == 0:
                all_dets = list(flatten(all_dets))
                all_mag = np.array(list(flatten(all_mag)))

        # One process does the trivial calculation
        all_flags = None
        central_mag = None
        if obs.comm_col_rank == 0:
            all_good = [True for x in all_dets]
            n_cut = 1
            while n_cut > 0:
                n_cut = 0
                mn = np.mean(all_mag[all_good])
                std = np.std(all_mag[all_good])
                for idet, det in enumerate(all_dets):
                    if not all_good[idet]:
                        continue
                    if np.absolute(all_mag[idet] - mn) > self.relcal_cut_sigma * std:
                        all_good[idet] = False
                        n_cut += 1
            central_mag = np.mean(all_mag[all_good])
            all_flags = {
                x: self.hwp_flag_mask for i, x in enumerate(all_dets) if not all_good[i]
            }
        if obs.comm_col is not None:
            all_flags = obs.comm_col.bcast(all_flags, root=0)
            central_mag = obs.comm_col.bcast(central_mag, root=0)

        # Every process flags its local detectors
        det_check = set(dets)
        local_flags = dict(obs.local_detector_flags)
        for det, val in all_flags.items():
            if det in det_check:
                local_flags[det] |= val
        obs.update_local_detector_flags(local_flags)
        local_good = [x for x in dets if x not in all_flags]

        return local_good, central_mag

    def _average_magnitude(self, dets, coeff, coeff_flags):
        mag = dict()
        if self.time_drift:
            # 4 values per harmonic, 2f is index 1
            re_comp = 4 * 1 + 0
            im_comp = 4 * 1 + 2
        else:
            # 2 values per harmonic, 2f is index 1
            re_comp = 2 * 1 + 0
            im_comp = 2 * 1 + 1
        n_chunk = coeff.shape[2]
        for idet, det in enumerate(dets):
            ch_mag = list()
            for ch in range(n_chunk):
                if coeff_flags[ch] != 0:
                    # All detectors in this chunk were flagged
                    continue
                if coeff[idet, re_comp, ch] == 0 and coeff[idet, im_comp, ch] == 0:
                    # This detector data was flagged
                    continue
                ch_mag.append(
                    np.sqrt(
                        coeff[idet, re_comp, ch] ** 2 + coeff[idet, im_comp, ch] ** 2
                    )
                )
            mag[det] = np.mean(ch_mag)
        return mag

    def _fit_chunk(
        self,
        obs,
        dets,
        indx,
        start,
        end,
        sincos,
        sh_flags,
        det_flags,
        reltime,
        coeff,
        coeff_flags,
    ):
        log = Logger.get()
        ch_timer = Timer()
        ch_timer.start()

        # The sample slice
        slc = slice(start, end, 1)
        slc_samps = end - start

        if reltime is None:
            ch_reltime = None
        else:
            ch_reltime = reltime[slc]

        obs_cov = hwpss_compute_coeff_covariance(
            sincos[slc],
            sh_flags[slc],
            comm=obs.comm.comm_group,
            times=ch_reltime,
            time_drift=self.time_drift,
        )
        if obs_cov is None:
            msg = f"HWPSS Model {obs.name}[{indx}] ({slc_samps} samples)"
            msg += " failed to compute coefficient"
            msg += " covariance.  Flagging this chunk when building model."
            log.verbose_rank(msg, comm=obs.comm.comm_group)
            coeff_flags[indx] = 1
            return

        msg = f"HWPSS Model {obs.name}[{indx}]: built coefficient covariance in"
        log.verbose_rank(msg, comm=obs.comm.comm_group, timer=ch_timer)

        for idet, det in enumerate(dets):
            good_samp = det_flags[det][slc] == 0
            if np.count_nonzero(good_samp) < coeff.shape[1]:
                # Not very many good samples, set coefficients to zero
                msg = f"HWPSS Model {obs.name}[{indx}] {det}: insufficient good "
                msg += "samples, setting coefficients to zero"
                log.verbose(msg)
                coeff[idet, :, indx] = 0
                continue
            sig = np.array(obs.detdata[self.det_data][det, slc])
            dc = np.mean(sig[good_samp])
            sig -= dc

            cf = hwpss_compute_coeff(
                sincos[slc],
                sig,
                det_flags[det][slc],
                obs_cov[0],
                obs_cov[1],
                times=ch_reltime,
                time_drift=self.time_drift,
            )
            if idet == 0:
                cfstr = ""
                for ic in cf:
                    cfstr += f"{ic} "
            coeff[idet, :, indx] = cf

        msg = f"HWPSS Model {obs.name}[{indx}]: compute detector coefficients in"
        log.verbose_rank(msg, comm=obs.comm.comm_group, timer=ch_timer)

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
            shared_flags = np.array(obs.shared[self.shared_flags].data)
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
                det_flags[det] = np.copy(obs.detdata[self.det_flags][det])
                det_flags[det] &= self.det_flag_mask
                det_flags[det] |= shared_flags
        return (shared_flags, det_flags)

    def _stopped_flags(self, obs):
        hdata = np.unwrap(obs.shared[self.hwp_angle].data, period=2 * np.pi)
        hvel = np.gradient(hdata)
        moving = np.absolute(hvel) > 1.0e-6
        nominal = np.median(hvel[moving])
        unstable = np.absolute(hvel - nominal) > 1.0e-3 * nominal
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
