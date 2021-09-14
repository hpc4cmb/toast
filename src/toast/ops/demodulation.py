# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
from time import time
import warnings

from astropy import units as u
from astropy.table import QTable

import numpy as np
from scipy.signal import firwin, fftconvolve
import traitlets

from ..mpi import MPI, MPI_Comm, use_mpi, Comm

from .operator import Operator
from .. import qarray as qa
from ..timing import function_timer
from ..traits import trait_docs, Int, Unicode, Bool, Dict, Quantity, Instance, Float
from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned, name_UID
from ..observation import Observation
from ..observation import default_names as obs_names
from ..noise import Noise
from ..instrument import Telescope, Focalplane
from ..data import Data


class Lowpass:
    """ A callable class that applies the low pass filter """

    def __init__(self, wkernel, fmax, fsample, offset, nskip, window="hamming"):
        """ Arguments:
        wkernel(int) : width of the filter kernel
        fmax(float) : maximum frequency of the filter
        fsample(float) : signal sampling frequency
        offset(int) : signal index offset for downsampling
        nskip(int) : downsampling factor
        """
        self.lpf = firwin(
            wkernel,
            fmax.to_value(u.Hz),
            window=window,
            pass_zero=True,
            fs=fsample.to_value(u.Hz),
        )
        self._offset = offset
        self._nskip = nskip

    def __call__(self, signal):
        lowpassed = fftconvolve(signal, self.lpf, mode="same").real
        downsampled = lowpassed[self._offset % self._nskip :: self._nskip]
        return downsampled


@trait_docs
class Demodulate(Operator):
    """ Demodulate and downsample HWP-modulated data
    """

    API = Int(0, help="Internal interface version for this operator")

    pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator.  "
        "Used exclusively for pointing weights, not pixel numbers.",
    )

    times = Unicode(
        obs_names.times,
        help="Observation shared key for timestamps",
    )

    hwp_angle = Unicode(
        obs_names.hwp_angle, help="Observation shared key for HWP angle"
    )

    boresight = Unicode(
        obs_names.boresight_radec, help="Observation shared key for boresight"
    )

    det_data = Unicode(
        obs_names.det_data,
        help="Observation detdata key apply filtering to",
    )

    det_flags = Unicode(
        obs_names.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(1, help="Bit mask value for optional detector flagging")

    demod_flag_mask = Int(1, help="Bit mask value for demod & downsample flagging")

    shared_flags = Unicode(
        obs_names.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(1, help="Bit mask value for optional shared flagging")

    noise_model = Unicode(
        "noise_model",
        allow_none=True,
        help="Observation key containing the noise model",
    )

    wkernel = Int(None, allow_none=True, help="kernel size of filter")

    fmax = Quantity(None, allow_none=True, help="Maximum frequency for lowpass")

    nskip = Int(3, help="Downsampling factor")

    window = Unicode("hamming", help="Window function name recognized by scipy.signal.firwin")

    purge = Bool(False, help="Remove inputs after demodulation")

    do_2f = Bool(False, help="also cache the 2f-demodulated signal")

    # Intervals?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.demod_data = Data()
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "noise_model", "pointing":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        # Demodulation only applies to observations with HWP.  Verify
        # that there are such observations in `data`

        demodulate_obs = []
        for obs in data.obs:
            if self.hwp_angle not in obs.shared:
                continue
            hwp_angle = obs.shared[self.hwp_angle]
            if np.abs(np.median(np.diff(hwp_angle))) < 1e-6:
                # Stepped or stationary HWP
                continue
            demodulate_obs.append(obs)
        n_obs = len(demodulate_obs)
        if data.comm.comm_world is not None:
            n_obs = data.comm.comm_world.allreduce(n_obs)
        if n_obs == 0:
            raise RuntimeError(
                "None of the observations have a spinning HWP.  Nothing to demodulate."
            )

        # Each modulated detector demodulates into 3 or 5 pseudo detectors

        self.prefixes = ["demod0", "demod4r", "demod4i"]
        if self.do_2f:
            self.prefixes.extend(["demod2r", "demod2i"])

        for obs in demodulate_obs:
            dets = obs.select_local_detectors(detectors)

            offset = obs.local_index_offset
            nsample = obs.n_local_samples

            fsample = obs.telescope.focalplane.sample_rate
            fmax, hwp_rate = self._get_fmax(obs)
            wkernel = self._get_wkernel(fmax, fsample)
            lowpass = Lowpass(wkernel, fmax, fsample, offset, self.nskip, self.window)

            # Create a new observation to hold the demodulated and downsampled data

            demod_telescope = self._demodulate_telescope(obs)
            demod_times = self._demodulate_times(obs)
            demod_detsets = self._demodulate_detsets(obs)
            demod_sample_sets = self._demodulate_sample_sets(obs)
            demod_process_rows = obs.dist.process_rows

            demod_name = f"demod_{obs.name}"
            demod_obs = Observation(
                demod_telescope,
                demod_times.size,
                name=demod_name,
                uid=name_UID(demod_name),
                comm=obs.comm,
                detector_sets=demod_detsets,
                process_rows=demod_process_rows,
                sample_sets=demod_sample_sets,
            )

            # Allocate storage

            demod_dets = []
            for det in dets:
                for prefix in self.prefixes:
                    demod_dets.append(f"{prefix}_{det}")
            n_local = demod_obs.n_local_samples

            demod_obs.shared.create(self.times, (n_local,), comm=demod_obs.comm_col)
            demod_obs.shared[self.times].set(demod_times, offset=(0,), fromrank=0)
            demod_obs.shared.create(
                self.boresight, (n_local, 4), comm=demod_obs.comm_col
            )
            demod_obs.shared.create(
                self.shared_flags, (n_local,), dtype=np.uint8, comm=demod_obs.comm_col
            )

            demod_obs.detdata.ensure(self.det_data, detectors=demod_dets)
            demod_obs.detdata.ensure(self.det_flags, detectors=demod_dets)

            self._demodulate_flags(obs, demod_obs, dets, wkernel, offset)
            self._demodulate_signal(data, obs, demod_obs, dets, lowpass)
            self._demodulate_pointing(data, obs, demod_obs, dets, lowpass, offset)
            self._demodulate_noise(obs, demod_obs, dets, fsample, hwp_rate, lowpass)

            #self._demodulate_offsets(obs, tod)

            self.demod_data.obs.append(demod_obs)

        return

    @function_timer
    def _get_fmax(self, obs):
        times = obs.shared[self.times].data
        hwp_angle = np.unwrap(obs.shared[self.hwp_angle].data)
        hwp_rate = np.mean(np.diff(hwp_angle) / np.diff(times)) / (2 * np.pi) * u.Hz
        if self.fmax is not None:
            fmax = self.fmax
        else:
            # set low-pass filter cut-off frequency as same as HWP 1f
            fmax = hwp_rate
        return fmax, hwp_rate

    @function_timer
    def _get_wkernel(self, fmax, fsample):
        if self.wkernel is not None:
            wkernel = self.wkernel
        else:
            # set kernel size longer than low-pass filter time scale
            wkernel = (1 << int(np.ceil(np.log(fsample / fmax * 10) / np.log(2)))) - 1
        return wkernel

    @function_timer
    def _demodulate_telescope(self, obs):
        focalplane = obs.telescope.focalplane
        det_data = focalplane.detector_data
        field_names = det_data.colnames
        # Initialize files to empty lists
        fields = dict([(name, []) for name in field_names])
        for idet, det in enumerate(det_data["name"]):
            for field_name in field_names:
                # Each detector translates into 3 or 5 new entries
                for prefix in self.prefixes:
                    if field_name == "name":
                        fields[field_name].append(f"{prefix}_{det}")
                    else:
                        fields[field_name].append(det_data[field_name][idet])
        fields = [fields[field_name] for field_name in field_names]
        demod_det_data = QTable(fields, names=field_names)
        demod_focalplane = Focalplane(
            detector_data=demod_det_data,
            field_of_view=focalplane.field_of_view,
            sample_rate=focalplane.sample_rate / self.nskip,
        )
        demod_name = f"demod_{obs.telescope.name}"
        demod_telescope = Telescope(
            name=demod_name,
            uid=name_UID(demod_name),
            focalplane=demod_focalplane,
            site=obs.telescope.site,
        )
        return demod_telescope

    @function_timer
    def _demodulate_times(self, obs):
        """ Downsample timestamps """
        times = obs.shared[self.times].data.copy()
        if self.nskip != 1:
            offset = obs.local_index_offset
            times = times[offset % self.nskip :: self.nskip]
        return times

    @function_timer
    def _demodulate_detsets(self, obs):
        """ Lump all derived detectors into detector sets """
        detsets = obs.all_detector_sets
        demod_detsets = []
        if detsets is None:
            for det in obs.all_detectors:
                demod_detset = []
                for prefix in self.prefixes:
                    demod_detset.append(f"{prefix}_{det}")
                demod_detsets.append(demod_detset)
        else:
            for detset in detsets:
                demod_detset = []
                for det in detset:
                    for prefix in self.prefixes:
                        demod_detset.append(f"{prefix}_{det}")
                demod_detsets.append(demod_detset)
        return demod_detsets

    @function_timer
    def _demodulate_sample_sets(self, obs):
        sample_sets = obs.all_sample_sets
        if sample_sets is None:
            return None
        demod_sample_sets = []
        offset = 0
        for sample_set in sample_sets:
            demod_sample_set = []
            for chunksize in sample_set:
                first_sample = offset
                last_sample = offset + chunksize
                demod_first_sample = int(np.ceil(first_sample / self.nskip))
                demod_last_sample = int(np.ceil(last_sample / self.nskip))
                demod_chunksize = demod_last_sample - demod_first_sample
                demod_sample_set.append(demod_chunksize)
                offset += chunksize
            demod_sample_sets.append(demod_sample_set)
        return demod_sample_sets

    @function_timer
    def _demodulate_offsets(self, obs, tod):
        if self.nskip == 1:
            return
        # Modulate sample distribution
        old_offsets, old_nsamples = np.vstack(tod._dist_samples).T
        new_offsets = old_offsets // self.nskip
        new_nsamples = np.roll(np.cumsum(new_offsets), -1)
        new_total = (tod.total_samples - 1) // self.nskip + 1
        new_nsamples[-1] = new_total - new_offsets[-1]
        tod._dist_samples = list(np.vstack([new_offsets, new_nsamples]).T)
        tod._nsamp = new_total
        offset, nsample = tod.local_samples
        times = tod.local_times()
        for ival in obs[self._intervals]:
            ival.first //= self.nskip
            ival.last //= self.nskip
            local_first = ival.first - offset
            if local_first >= 0 and local_first < nsample:
                ival.start = times[local_first]
            local_last = ival.last - offset
            if local_last >= 0 and local_last < nsample:
                ival.stop = times[local_last]
        return

    @function_timer
    def _demodulate_flag(self, flags, wkernel, offset):
        """ Collapse flags inside the filter window and downsample """
        """
        # FIXME: this is horribly inefficient but optimization may require
        # FIXME: a compiled kernel
        n = flags.size
        new_flags = []
        width = wkernel // 2 + 1
        for i in range(0, n, self.nskip):
            ind = slice(max(0, i - width), min(n, i + width + 1))
            buf = flags[ind]
            flag = buf[0]
            for flag2 in buf[1:]:
                flag |= flag2
            new_flags.append(flag)
        new_flags = np.array(new_flags)
        """
        # FIXME: for now, just downsample the flags.  Real data will require
        # FIXME:    measuring the total flag within the filter window
        flags = flags.copy()
        # flag invalid samples in both ends
        flags[: wkernel // 2] |= self.demod_flag_mask
        flags[-(wkernel // 2) :] |= self.demod_flag_mask
        new_flags = flags[offset % self.nskip::self.nskip]
        return new_flags

    @function_timer
    def _demodulate_common_flags(self, tod, wkernel, offset):
        """ Combine and downsample flags in the filter window """
        common_flags = tod.local_common_flags()
        new_flags = self._demodulate_flag(common_flags, wkernel, offset)
        tod.cache.put(tod.COMMON_FLAG_NAME, new_flags, replace=True)
        return

    @function_timer
    def _demodulate_signal(self, data, obs, demod_obs, dets, lowpass):
        """ demodulate signal TOD """

        for det in dets:
            signal = obs.detdata[self.det_data][det]
            # Get weights
            obs_data = data.select(obs_uid=obs.uid)
            self.pointing.apply(obs_data, dets=[det])
            weights = obs.detdata[self.pointing.weights][det]
            # iweights = 1
            # qweights = eta * cos(2 * psi_det + 4 * psi_hwp)
            # uweights = eta * sin(2 * psi_det + 4 * psi_hwp)
            iweights, qweights, uweights = weights.T
            etainv = 1 / np.sqrt(qweights ** 2 + uweights ** 2)
            signal_demod0 = lowpass(signal)
            signal_demod4r = lowpass(signal * 2 * qweights * etainv)
            signal_demod4i = lowpass(signal * 2 * uweights * etainv)

            det_data = demod_obs.detdata[self.det_data]
            det_data[f"demod0_{det}"] = signal_demod0
            det_data[f"demod4r_{det}"] = signal_demod4r
            det_data[f"demod4i_{det}"] = signal_demod4i

            if self.do_2f:
                # Start by evaluating the 2f demodulation factors from the
                # pointing matrix.  We use the half-angle formulas and some
                # extra logic to identify the right branch
                #
                # |cos(psi/2)| and |sin(psi/2)|:
                signal_demod2r = np.sqrt(0.5 * (1 + qweights * etainv))
                signal_demod2i = np.sqrt(0.5 * (1 - qweights * etainv))
                # inverse the sign for every second mode
                for sig in signal_demod2r, signal_demod2i:
                    dsig = np.diff(sig)
                    dsig[sig[1:] > 0.5] = 0
                    starts = np.where(dsig[:-1] * dsig[1:] < 0)[0]
                    for start, stop in zip(starts[::2], starts[1::2]):
                        sig[start + 1:stop + 2] *= -1
                    # handle some corner cases
                    dsig = np.diff(sig)
                    dstep = np.median(np.abs(dsig[sig[1:] < 0.5]))
                    bad = np.abs(dsig) > 2 * dstep
                    bad = np.hstack([bad, False])
                    sig[bad] *= -1
                # Demodulate and lowpass for 2f
                signal_demod2r = lowpass(signal * signal_demod2r)
                signal_demod2i = lowpass(signal * signal_demod2i)
                det_data[f"demod2r_{det}"] = signal_demod2r
                det_data[f"demod2i_{det}"] = signal_demod2i

        if self.purge:
            del obs.det_data[self.det_data]

        return

    @function_timer
    def _demodulate_flags(self, obs, demod_obs, dets, wkernel, offset):
        """ Demodulate and downsample flags """

        shared_flags = obs.shared[self.shared_flags].data
        demod_shared_flags = self._demodulate_flag(shared_flags, wkernel, offset)
        demod_obs.shared[self.shared_flags].set(
            demod_shared_flags, offset=(0,), fromrank=0
        )
        if self.purge:
            del obs.shared[self.shared_flags]

        for det in dets:
            flags = obs.detdata[self.det_flags][det]
            # Downsample flags
            demod_flags = self._demodulate_flag(flags, wkernel, offset)
            for prefix in self.prefixes:
                demod_det = f"{prefix}_{det}"
                demod_obs.detdata[self.det_flags][demod_det] = demod_flags
        if self.purge:
            del obs.detdata[self.det_flags]
        return

    @function_timer
    def _demodulate_pointing(self, data, obs, demod_obs, dets, lowpass, offset):
        """ demodulate pointing matrix """

        # Pointing matrix is now computed on the fly.  We only need to
        # demodulate the boresight quaternions

        quats = obs.shared[self.boresight].data
        demod_obs.shared[self.boresight].set(
            quats[offset % self.nskip :: self.nskip],
            offset=(0, 0),
            fromrank=0,
        )

        # This code is kept for reference
        """
        weights = tod.cache.reference("weights_{}".format(det))
        iweights, qweights, uweights = weights.T
        # We lowpass even constant-valued vectors to match the
        # normalization and downsampling
        iweights = lowpass(iweights)
        eta = np.sqrt(qweights ** 2 + uweights ** 2)
        eta = lowpass(eta)
        zeros = np.zeros_like(iweights)

        weights_demod0 = np.column_stack([iweights, zeros, zeros])
        weights_name = "weights_demod0_{}".format(det)
        tod.cache.put(weights_name, weights_demod0, replace=True)

        weights_demod4r = np.column_stack([zeros, eta, zeros])
        weights_name_4r = "weights_demod4r_{}".format(det)
        tod.cache.put(weights_name_4r, weights_demod4r, replace=True)

        weights_demod4i = np.column_stack([zeros, zeros, eta])
        weights_name_4i = "weights_demod4i_{}".format(det)
        tod.cache.put(weights_name_4i, weights_demod4i, replace=True)

        # Downsample and copy pixel numbers
        local_pixels = tod.cache.reference("pixels_{}".format(det))
        pixels = local_pixels[offset % self.nskip :: self.nskip]
        for demodkey in ["demod0", "demod4r", "demod4i"]:
            demod_name = "pixels_{}_{}".format(demodkey, det)
            tod.cache.put(demod_name, pixels, replace=True)

        if self.purge:
            tod.cache.destroy("{}_{}".format("weights", det))
            tod.cache.destroy("{}_{}".format("pixels", det))
        """
        return

    @function_timer
    def _demodulate_noise(
            self,
            obs,
            demod_obs,
            dets,
            fsample,
            hwp_rate,
            lowpass,
    ):
        """ Add Noise objects for the new detectors """
        noise = obs[self.noise_model]

        demod_detectors = []
        demod_freqs = {}
        demod_psds = {}
        demod_indices = {}

        lpf = lowpass.lpf
        lpf_freq = np.fft.rfftfreq(lpf.size, 1 / fsample.to_value(u.Hz))
        lpf_value = np.abs(np.fft.rfft(lpf)) ** 2
        for det in dets:
            # weight -- ignored
            # index  - ignored
            # rate
            rate_in = noise.rate(det)
            # freq
            freq_in = noise.freq(det)
            # psd
            psd_in = noise.psd(det)
            n_mode = len(self.prefixes)
            for indexoff, prefix in enumerate(self.prefixes):
                demod_det = "{}_{}".format(prefix, det)
                # Lowpass
                if prefix == "demod0":
                    # lowpass psd
                    psd_out = psd_in * np.interp(freq_in.to_value(u.Hz), lpf_freq, lpf_value)
                elif prefix.startswith("demod2"):
                    # get noise at 2f
                    psd_out = np.zeros_like(psd_in)
                    psd_out[:] = np.interp(2 * hwp_rate, freq_in, psd_in)
                else:
                    # get noise at 4f
                    psd_out = np.zeros_like(psd_in)
                    psd_out[:] = np.interp(4 * hwp_rate, freq_in, psd_in)
                # Downsample
                rate_out = rate_in / self.nskip
                ind = freq_in <= rate_out / 2
                freq_out = freq_in[ind]
                # Last bin must equal the new Nyquist frequency
                freq_out[-1] = rate_out / 2
                psd_out = psd_out[ind] / self.nskip
                # Insert
                demod_detectors.append(demod_det)
                demod_freqs[demod_det] = freq_out
                demod_psds[demod_det] = psd_out
                demod_indices[demod_det] = noise.index(det) * n_mode + indexoff
        demod_obs[self.noise_model] = Noise(
            detectors=demod_detectors,
            freqs=demod_freqs,
            psds=demod_psds,
            indices=demod_indices,
        )
        if self.purge:
            del obs[self._noise]
        return

    def _finalize(self, data, **kwargs):
        return self.demod_data

    def _requires(self):
        req = {
            "shared": [self.times, self.boresight],
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.boresight_azel is not None:
            req["shared"].append(self.boresight_azel)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()

    def _accelerators(self):
        return list()
