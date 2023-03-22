# Copyright (c) 2021-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
import warnings
from time import time

import numpy as np
import traitlets
from astropy import units as u
from astropy.table import QTable
from scipy.signal import fftconvolve, firwin

from ..data import Data
from ..instrument import Focalplane, Telescope
from ..intervals import IntervalList
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..noise import Noise
from ..observation import Observation
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import GlobalTimers, Logger, Timer, dtype_to_aligned, name_UID
from .operator import Operator


class Lowpass:
    """A callable class that applies the low pass filter"""

    def __init__(self, wkernel, fmax, fsample, offset, nskip, window="hamming"):
        """
        Args:
            wkernel (int) : width of the filter kernel
            fmax (float) : maximum frequency of the filter
            fsample (float) : signal sampling frequency
            offset (int) : signal index offset for downsampling
            nskip (int) : downsampling factor
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
    """Demodulate and downsample HWP-modulated data"""

    API = Int(0, help="Internal interface version for this operator")

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a Stokes weights operator",
    )

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    hwp_angle = Unicode(defaults.hwp_angle, help="Observation shared key for HWP angle")

    azimuth = Unicode(defaults.azimuth, help="Observation shared key for Azimuth")

    elevation = Unicode(defaults.elevation, help="Observation shared key for Elevation")

    boresight = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    demod_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for demod & downsample flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    noise_model = Unicode(
        "noise_model",
        allow_none=True,
        help="Observation key containing the noise model",
    )

    wkernel = Int(None, allow_none=True, help="Override automatic filter kernel size")

    fmax = Quantity(
        None, allow_none=True, help="Override automatic lowpass cut-off frequency"
    )

    nskip = Int(3, help="Downsampling factor")

    window = Unicode(
        "hamming", help="Window function name recognized by scipy.signal.firwin"
    )

    purge = Bool(False, help="Remove inputs after demodulation")

    do_2f = Bool(False, help="also cache the 2f-demodulated signal")

    # Intervals?

    @traitlets.validate("stokes_weights")
    def _check_stokes_weights(self, proposal):
        weights = proposal["value"]
        if weights is not None:
            if not isinstance(weights, Operator):
                raise traitlets.TraitError(
                    "stokes_weights should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["weights", "view"]:
                if not weights.has_trait(trt):
                    msg = f"stokes_weights operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return weights

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.demod_data = Data()
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "noise_model", "stokes_weights":
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

        timer = Timer()
        timer.start()
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
                obs.comm,
                demod_telescope,
                demod_times.size,
                name=demod_name,
                uid=name_UID(demod_name),
                detector_sets=demod_detsets,
                process_rows=demod_process_rows,
                sample_sets=demod_sample_sets,
            )
            for key, value in obs.items():
                if key == "noise_model":
                    # Will be generated later
                    continue
                demod_obs[key] = value

            # Allocate storage

            demod_dets = []
            for det in dets:
                for prefix in self.prefixes:
                    demod_dets.append(f"{prefix}_{det}")
            n_local = demod_obs.n_local_samples

            demod_obs.shared.create_column(self.times, (n_local,))
            demod_obs.shared[self.times].set(demod_times, offset=(0,), fromrank=0)
            demod_obs.shared.create_column(self.boresight, (n_local, 4))
            demod_obs.shared.create_column(
                self.shared_flags, (n_local,), dtype=np.uint8
            )

            self._demodulate_shared_data(obs, demod_obs)

            exists_data = demod_obs.detdata.ensure(
                self.det_data,
                detectors=demod_dets,
                dtype=np.float64,
                create_units=obs.detdata[self.det_data].units,
            )
            exists_flags = demod_obs.detdata.ensure(
                self.det_flags, detectors=demod_dets, dtype=np.uint8
            )

            self._demodulate_flags(obs, demod_obs, dets, wkernel, offset)
            self._demodulate_signal(data, obs, demod_obs, dets, lowpass)
            self._demodulate_pointing(data, obs, demod_obs, dets, lowpass, offset)
            self._demodulate_noise(obs, demod_obs, dets, fsample, hwp_rate, lowpass)

            self._demodulate_intervals(obs, demod_obs)

            self.demod_data.obs.append(demod_obs)

            if self.purge:
                if self.shared_flags is not None:
                    del obs.shared[self.shared_flags]
                del obs.detdata[self.det_data]
                if self.det_flags is not None:
                    del obs.detdata[self.det_flags]
                if self.noise_model is not None:
                    del obs[self.noise_model]

            log.debug_rank(
                "Demodulated observation in", comm=data.comm.comm_group, timer=timer
            )

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
        # Initialize fields to empty lists
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
        """Downsample timestamps"""
        times = obs.shared[self.times].data.copy()
        if self.nskip != 1:
            offset = obs.local_index_offset
            times = np.array(times[offset % self.nskip :: self.nskip])
        return times

    @function_timer
    def _demodulate_shared_data(self, obs, demod_obs):
        """Downsample shared data"""
        n_local = demod_obs.n_local_samples
        for key in self.azimuth, self.elevation:
            if key is None:
                continue
            values = obs.shared[key].data.copy()
            if self.nskip != 1:
                offset = obs.local_index_offset
                values = np.array(values[offset % self.nskip :: self.nskip])
            demod_obs.shared.create_column(key, (n_local,))
            demod_obs.shared[key].set(
                values,
                offset=(0,),
                fromrank=0,
            )
        return

    @function_timer
    def _demodulate_detsets(self, obs):
        """Lump all derived detectors into detector sets"""
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
    def _demodulate_intervals(self, obs, demod_obs):
        if self.nskip == 1:
            demod_obs.intervals = obs.intervals
            return
        times = demod_obs.shared[self.times]
        for name, ivals in obs.intervals.items():
            timespans = [[ival.start, ival.stop] for ival in ivals]
            demod_obs.intervals[name] = IntervalList(times, timespans=timespans)
        # Force the creation of new "all" interval
        del demod_obs.intervals[None]
        return

    @function_timer
    def _demodulate_flag(self, flags, wkernel, offset):
        """Collapse flags inside the filter window and downsample"""
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
        new_flags = np.array(flags[offset % self.nskip :: self.nskip])
        return new_flags

    @function_timer
    def _demodulate_signal(self, data, obs, demod_obs, dets, lowpass):
        """demodulate signal TOD"""

        for det in dets:
            signal = obs.detdata[self.det_data][det]
            # Get weights
            obs_data = data.select(obs_uid=obs.uid)
            self.stokes_weights.apply(obs_data, dets=[det])
            weights = obs.detdata[self.stokes_weights.weights][det]
            # iweights = 1
            # qweights = eta * cos(2 * psi_det + 4 * psi_hwp)
            # uweights = eta * sin(2 * psi_det + 4 * psi_hwp)
            iweights, qweights, uweights = weights.T
            etainv = 1 / np.sqrt(qweights**2 + uweights**2)

            det_data = demod_obs.detdata[self.det_data]
            det_data[f"demod0_{det}"] = lowpass(signal)
            det_data[f"demod4r_{det}"] = lowpass(signal * 2 * qweights * etainv)
            det_data[f"demod4i_{det}"] = lowpass(signal * 2 * uweights * etainv)

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
                        sig[start + 1 : stop + 2] *= -1
                    # handle some corner cases
                    dsig = np.diff(sig)
                    dstep = np.median(np.abs(dsig[sig[1:] < 0.5]))
                    bad = np.abs(dsig) > 2 * dstep
                    bad = np.hstack([bad, False])
                    sig[bad] *= -1
                # Demodulate and lowpass for 2f
                det_data[f"demod2r_{det}"] = lowpass(signal * signal_demod2r)
                det_data[f"demod2i_{det}"] = lowpass(signal * signal_demod2i)

        return

    @function_timer
    def _demodulate_flags(self, obs, demod_obs, dets, wkernel, offset):
        """Demodulate and downsample flags"""

        shared_flags = obs.shared[self.shared_flags].data
        demod_shared_flags = self._demodulate_flag(shared_flags, wkernel, offset)
        demod_obs.shared[self.shared_flags].set(
            demod_shared_flags, offset=(0,), fromrank=0
        )

        for det in dets:
            flags = obs.detdata[self.det_flags][det]
            # Downsample flags
            demod_flags = self._demodulate_flag(flags, wkernel, offset)
            for prefix in self.prefixes:
                demod_det = f"{prefix}_{det}"
                demod_obs.detdata[self.det_flags][demod_det] = demod_flags
        return

    @function_timer
    def _demodulate_pointing(self, data, obs, demod_obs, dets, lowpass, offset):
        """demodulate pointing matrix"""

        # Pointing matrix is now computed on the fly.  We only need to
        # demodulate the boresight quaternions

        quats = obs.shared[self.boresight].data
        demod_obs.shared[self.boresight].set(
            np.array(quats[offset % self.nskip :: self.nskip]),
            offset=(0, 0),
            fromrank=0,
        )

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
        """Add Noise objects for the new detectors"""
        noise = obs[self.noise_model]

        demod_detectors = []
        demod_freqs = {}
        demod_psds = {}
        demod_indices = {}
        demod_weights = {}

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
            # Lowpass transfer function
            tf = np.interp(freq_in.to_value(u.Hz), lpf_freq, lpf_value)
            # Find the highest frequency without significant suppression
            # to measure noise weights at
            iweight = tf.size - 1
            while iweight > 0 and tf[iweight] < 0.99:
                iweight -= 1
            # psd
            psd_in = noise.psd(det)
            n_mode = len(self.prefixes)
            for indexoff, prefix in enumerate(self.prefixes):
                demod_det = f"{prefix}_{det}"
                # Get the demodulated PSD
                if prefix == "demod0":
                    # this PSD does not change
                    psd_out = psd_in.copy()
                elif prefix.startswith("demod2"):
                    # get noise at 2f
                    psd_out = np.zeros_like(psd_in)
                    psd_out[:] = np.interp(2 * hwp_rate, freq_in, psd_in)
                else:
                    # get noise at 4f
                    psd_out = np.zeros_like(psd_in)
                    psd_out[:] = np.interp(4 * hwp_rate, freq_in, psd_in)
                # Lowpass
                psd_out *= tf
                # Downsample
                rate_out = rate_in / self.nskip
                ind = freq_in <= rate_out / 2
                freq_out = freq_in[ind]
                # Last bin must equal the new Nyquist frequency
                freq_out[-1] = rate_out / 2
                psd_out = psd_out[ind] / self.nskip
                # Calculate noise weight
                noisevar = psd_out[iweight].to_value(u.K**2 * u.second)
                invvar = 1.0 / noisevar / rate_out.to_value(u.Hz)
                # Insert
                demod_detectors.append(demod_det)
                demod_freqs[demod_det] = freq_out
                demod_psds[demod_det] = psd_out
                demod_indices[demod_det] = noise.index(det) * n_mode + indexoff
                demod_weights[demod_det] = invvar / u.K**2
        demod_obs[self.noise_model] = Noise(
            detectors=demod_detectors,
            freqs=demod_freqs,
            psds=demod_psds,
            indices=demod_indices,
            detweights=demod_weights,
        )
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


@trait_docs
class StokesWeightsDemod(Operator):
    """Compute the Stokes pointing weights for demodulated data"""

    API = Int(0, help="Internal interface version for this operator")

    mode = Unicode("IQU", help="The Stokes weights to generate")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    weights = Unicode(
        defaults.weights, help="Observation detdata key for output weights"
    )

    single_precision = Bool(False, help="If True, use 32bit float in output")

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        mode = proposal["value"]
        if mode not in ["IQU"]:
            raise traitlets.TraitError("Invalid mode (must be 'IQU')")
        return mode

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        nnz = len(self.mode)

        if self.single_precision:
            dtype = np.float32
        else:
            dtype = np.float64

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)

            exists_weights = obs.detdata.ensure(
                self.weights,
                sample_shape=(nnz,),
                dtype=dtype,
                detectors=dets,
            )
            nsample = obs.n_local_samples
            ones = np.ones(nsample, dtype=dtype)
            zeros = np.zeros(nsample, dtype=dtype)
            weights = obs.detdata[self.weights]
            for det in dets:
                props = obs.telescope.focalplane[det]
                if "pol_efficiency" in props.colnames:
                    eta = props["pol_efficiency"]
                else:
                    eta = 1.0
                if det.startswith("demod0"):
                    # Stokes I only
                    weights[det] = np.column_stack([ones, zeros, zeros])
                elif det.startswith("demod4r"):
                    # Stokes Q only
                    weights[det] = np.column_stack([zeros, eta * ones, zeros])
                elif det.startswith("demod4i"):
                    # Stokes U only
                    weights[det] = np.column_stack([zeros, zeros, eta * ones])
                else:
                    # 2f, systematics only
                    weights[det] = np.column_stack([zeros, zeros, zeros])
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": list(),
            "detdata": list(),
        }
        return req

    def _provides(self):
        return {"detdata": self.weights}
