# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
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

from .. import qarray as qa
from ..data import Data
from ..instrument import Focalplane, Telescope
from ..intervals import IntervalList
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..noise import Noise
from ..observation import Observation
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Logger, dtype_to_aligned, name_UID
from .operator import Operator


class Lowpass:
    """A callable class that applies the low pass filter"""

    def __init__(self, fmax, fsample, wkernel=None, offset=0, nskip=1, window="hamming"):
        """
        Args:
            wkernel (int) : width of the filter kernel
            fmax (float) : maximum frequency of the filter
            fsample (float) : signal sampling frequency
            offset (int) : signal index offset for downsampling
            nskip (int) : downsampling factor
        """
        if wkernel is None:
            # set kernel size longer than low-pass filter time scale
            wkernel = (1 << int(np.ceil(np.log(fsample / fmax * 10) / np.log(2)))) - 1
        self.wkernel = wkernel
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


class Bandpass:
    """A callable class that applies the bandpass filter"""

    def __init__(self, fmin, fmax, fsample, wkernel=None, window="hamming"):
        """
        Args:
            wkernel (int) : width of the filter kernel
            fmin (float) : minimum frequency of the passband
            fmax (float) : maximum frequency of the passband
            fsample (float) : signal sampling frequency
        """
        if wkernel is None:
            # set kernel size longer than low-pass filter time scale
            wkernel = (1 << int(np.ceil(np.log(fsample / fmin * 10) / np.log(2)))) - 1
        self.wkernel = wkernel
        self.bpf = firwin(
            wkernel,
            [fmin.to_value(u.Hz), fmax.to_value(u.Hz)],
            window=window,
            pass_zero=False,
            fs=fsample.to_value(u.Hz),
        )

    def __call__(self, signal, downsample=True):
        bandpassed = fftconvolve(signal, self.bpf, mode="same").real
        return bandpassed


@trait_docs
class Demodulate(Operator):
    """Demodulate and downsample HWP-modulated data"""

    allowed_modes = ("", "I", "QU", "IQU")

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

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to.  Use ';' if multiple "
        "signal flavors should be demodulated.",
    )

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for detector sample flagging"
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

    fcut = Float(
        0.95, help="Low pass cut-off frequency in units if HWP frequency"
    )

    fmin_2f = Float(
        1.05, help="Low frequency end of the 2f-bandpass filter in units of HWP frequency"
    )

    fmax_2f = Float(
        2.95, help="High frequency end of the 2f-bandpass filter in units of HWP frequency"
    )

    fmin_4f = Float(
        3.05, help="Low frequency end of the 4f-bandpass filter in units of HWP frequency"
    )

    fmax_4f = Float(
        4.95, help="High frequency end of the 4fbandpass filter in units of HWP frequency"
    )

    nskip = Int(3, help="Downsampling factor")

    window = Unicode(
        "hamming", help="Window function name recognized by scipy.signal.firwin"
    )

    keep_dets_frac = Float(
        0.1,
        help="If less than this fraction of detectors are good, cut the observation",
    )

    purge = Bool(False, help="Remove inputs after demodulation")

    in_place = Bool(False, help="Modify the data object in-place.  Implies purge=True.")

    do_2f = Bool(False, help="also cache the 2f-demodulated signal")

    mode = Unicode("IQU", help="Return I, QU or IQU timestreams.")

    # Intervals?

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
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("stokes_weights")
    def _check_stokes_weights(self, proposal):
        weights = proposal["value"]
        if weights is not None:
            if not isinstance(weights, Operator):
                raise traitlets.TraitError(
                    "stokes_weights should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["weights", "view", "mode"]:
                if not weights.has_trait(trt):
                    msg = f"stokes_weights operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
            # Check that weights are supported
            supported = ("I", "QU", "IQU")
            if weights.mode not in supported:
                msg = f"Stokes weights mode not in {supported}"
                raise traitlets.TraitError(msg)
        return weights

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        mode = proposal["value"]
        if mode not in self.allowed_modes:
            msg = f"mode must be one of {self.allowed_modes}"
            raise traitlets.TraitError(msg)
        return mode

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "noise_model", "stokes_weights":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if "QU" in self.mode and "QU" not in self.stokes_weights.mode:
            msg = "Cannot produce demodulated QU without QU Stokes weights"
            raise RuntimeError(msg)

        if self.stokes_weights.hwp_angle is None:
            msg = f"The Stokes weights operator (self.stokes_weights) "
            msg += "does not have HWP angle"
            raise RuntimeError(msg)

        if self.in_place:
            self.demod_data = None
        else:
            self.demod_data = Data()

        # Demodulation only applies to observations with HWP.  Verify
        # that there are such observations in `data`.  We also cut all
        # observations where more than self.keep_frac fraction of optical
        # detectors are cut.

        demodulate_input_obs = []
        for obs in data.obs:
            if self.hwp_angle not in obs.shared:
                msg = f"Obs {obs.name} has no HWP angle, skipping"
                log.debug_rank(msg, obs.comm.comm_group)
                if self.in_place or self.purge:
                    # Un-demodulated observations will be deleted
                    obs.clear()
                continue
            hwp_angle = obs.shared[self.hwp_angle]
            if np.abs(np.median(np.diff(hwp_angle))) < 1e-6:
                # Stepped or stationary HWP
                msg = f"Obs {obs.name} has a stepped / stationary HWP, skipping"
                log.debug_rank(msg, obs.comm.comm_group)
                if self.in_place:
                    # Un-demodulated observations will be deleted
                    obs.clear()
                continue
            n_local = len(obs.local_detectors)
            n_local_good = np.sum(
                [1 for x, y in obs.local_detector_flags.items() if y == 0]
            )
            if obs.comm.comm_group is None:
                n_dets = n_local
                n_good = n_local_good
            else:
                n_dets = obs.comm.comm_group.allreduce(n_local, op=MPI.SUM)
                n_good = obs.comm.comm_group.allreduce(n_local_good, op=MPI.SUM)
            if n_good / n_dets < self.keep_dets_frac:
                msg = f"Obs {obs.name} has only {n_good} / {n_dets} good dets, cutting"
                log.debug_rank(msg, obs.comm.comm_group)
                if self.in_place:
                    # Un-demodulated observations will be deleted
                    obs.clear()
                continue
            demodulate_input_obs.append(obs)
        n_obs = len(demodulate_input_obs)
        if data.comm.comm_world is not None:
            n_obs = data.comm.comm_world.allreduce(n_obs)
        if n_obs == 0:
            raise RuntimeError(
                "None of the observations have a spinning HWP and/or enough detectors. "
                "Nothing to demodulate."
            )

        # Each modulated detector demodulates into one or more pseudo detectors

        self.prefixes = []
        if "I" in self.mode:
            self.prefixes.append("demod0")
        if "QU" in self.mode:
            self.prefixes.extend(["demod4r", "demod4i"])
        if self.do_2f:
            self.prefixes.extend(["demod2r", "demod2i"])
        if len(self.prefixes) == 0:
            raise RuntimeError("There are no pseudo detectors to modulate to")

        timer = Timer()
        timer.start()

        # The list of demodulated observations.  This will either be placed in a new
        # data object or the list will be swapped into the existing data object.
        demodulate_obs = []

        for obs in demodulate_input_obs:
            # Get the detectors which are not cut with per-detector flags
            local_dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            if obs.comm.comm_group is None:
                all_dets = local_dets
            else:
                proc_dets = obs.comm.comm_group.gather(local_dets, root=0)
                all_dets = None
                if obs.comm.comm_group.rank == 0:
                    all_dets = set()
                    for pdets in proc_dets:
                        for d in pdets:
                            all_dets.add(d)
                    all_dets = list(sorted(all_dets))
                all_dets = obs.comm.comm_group.bcast(all_dets, root=0)

            offset = obs.local_index_offset
            nsample = obs.n_local_samples

            fsample = obs.telescope.focalplane.sample_rate
            # fmod is the HWP spin frequency.  Polarization signal is at 4 x fmod
            fmod = self._get_fmod(obs)

            lowpass = Lowpass(
                self.fcut * fmod,
                fsample,
                wkernel=self.wkernel,
                offset=offset,
                nskip=self.nskip,
                window=self.window,
            )
            bandpass2f = Bandpass(
                self.fmin_2f * fmod,
                self.fmax_2f * fmod,
                fsample,
                wkernel=self.wkernel,
                window=self.window,
            )
            bandpass4f = Bandpass(
                self.fmin_4f * fmod,
                self.fmax_4f * fmod,
                fsample,
                wkernel=self.wkernel,
                window=self.window,
            )

            # Create a new observation to hold the demodulated and downsampled data

            demod_telescope = self._demodulate_telescope(obs, all_dets)
            demod_all_samples = self._demodulated_samples(obs)
            demod_detsets = self._demodulate_detsets(obs, all_dets)
            demod_sample_sets = self._demodulate_sample_sets(obs)
            demod_process_rows = obs.dist.process_rows

            demod_name = f"demod_{obs.name}"
            demod_obs = Observation(
                obs.comm,
                demod_telescope,
                demod_all_samples,
                name=demod_name,
                uid=name_UID(demod_name),
                session=obs.session,
                detector_sets=demod_detsets,
                process_rows=demod_process_rows,
                sample_sets=demod_sample_sets,
            )

            # Allocate storage

            demod_dets = []
            for det in local_dets:
                for prefix in self.prefixes:
                    demod_dets.append(f"{prefix}_{det}")

            self._demodulate_shared_data(obs, demod_obs)

            for det_data in self.det_data.split(";"):
                exists_data = demod_obs.detdata.ensure(
                    det_data,
                    detectors=demod_dets,
                    dtype=np.float64,
                    create_units=obs.detdata[det_data].units,
                )
            exists_flags = demod_obs.detdata.ensure(
                self.det_flags, detectors=demod_dets, dtype=np.uint8
            )

            self._demodulate_flags(obs, demod_obs, local_dets, lowpass.wkernel, offset)
            self._demodulate_signal(
                data, obs, demod_obs, local_dets, lowpass, bandpass2f, bandpass4f
            )
            self._demodulate_noise(
                obs,
                demod_obs,
                local_dets,
                fsample,
                fmod,
                lowpass,
                bandpass2f,
                bandpass4f,
            )

            self._demodulate_intervals(obs, demod_obs)

            self._demodulate_metadata(obs, demod_obs)

            demodulate_obs.append(demod_obs)

            if self.in_place or self.purge:
                # Input observations are not saved
                obs.clear()

            log.debug_rank(
                f"Demodulated observation {obs.name} in",
                comm=data.comm.comm_group,
                timer=timer,
            )
        if self.in_place:
            data.obs.clear()
            data.obs = demodulate_obs
        else:
            self.demod_data.obs = demodulate_obs

    @function_timer
    def _get_fmod(self, obs):
        """Return the modulation frequency"""
        times = obs.shared[self.times].data
        hwp_angle = np.unwrap(obs.shared[self.hwp_angle].data)
        hwp_rate = np.absolute(
            np.mean(np.diff(hwp_angle) / np.diff(times)) / (2 * np.pi) * u.Hz
        )
        return hwp_rate

    @function_timer
    def _demodulate_telescope(self, obs, all_dets):
        focalplane = obs.telescope.focalplane
        det_data = focalplane.detector_data
        field_names = det_data.colnames
        # Initialize fields to empty lists
        fields = {name: list() for name in field_names}
        all_set = set(all_dets)
        for row, det in enumerate(det_data["name"]):
            if det not in all_set:
                continue
            for field_name in field_names:
                # Each detector translates into one or more
                for prefix in self.prefixes:
                    if field_name == "name":
                        fields[field_name].append(f"{prefix}_{det}")
                    else:
                        fields[field_name].append(det_data[field_name][row])
        demod_det_data = QTable(
            [fields[field_name] for field_name in field_names], names=field_names
        )
        my_all = list()
        for name in demod_det_data["name"]:
            my_all.append(name)

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
    def _demodulated_samples(self, obs):
        """Compute number of samples in the demodulated observation."""
        n_all = None
        if obs.comm_row_rank == 0:
            # First process row gathers the size of sample slices to the first process
            off = obs.local_index_offset % self.nskip
            slc = slice(off, None, self.nskip)
            n_local = len(obs.shared[self.times].data[slc])
            if obs.comm_row is None:
                n_all = n_local
            else:
                n_all = obs.comm_row.reduce(n_local, op=MPI.SUM, root=0)
        if obs.comm.comm_group is not None:
            # Broadcast result to the whole group
            n_all = obs.comm.comm_group.bcast(n_all, root=0)
        return n_all

    @function_timer
    def _demodulate_shared_data(self, obs, demod_obs):
        """Downsample shared data"""
        for field in obs.shared.keys():
            shobj = obs.shared[field]
            commtype = obs.shared.comm_type(field)
            if commtype == "group":
                # Using full group communicator, just copy to new obs.
                demod_obs.shared.assign_mpishared(field, shobj, commtype)
            elif commtype == "row":
                # Shared in the sample direction (per-detector object like a beam,
                # bandpass, etc).  This means that downsampling does not effect the
                # shared object.  Just copy to the new obs.
                demod_obs.shared.assign_mpishared(field, shobj, commtype)
            elif commtype == "column":
                # Shared in the detector direction.
                # Set the data on one process
                if obs.comm_col_rank == 0:
                    off = obs.local_index_offset % self.nskip
                    slc = slice(off, None, self.nskip)
                    values = np.ascontiguousarray(obs.shared[field].data[slc])
                    n_samp = len(values)
                else:
                    n_samp = None
                    values = None

                # Data type
                dtype = shobj.dtype

                # Downsampled shape
                if obs.comm_col is not None:
                    n_samp = obs.comm_col.bcast(n_samp, root=0)
                shp = [n_samp]
                shp.extend(shobj.shape[1:])
                shp = tuple(shp)

                # Create the object and set
                demod_obs.shared.create_column(
                    field,
                    shape=shp,
                    dtype=dtype,
                )
                demod_obs.shared[field].set(values, fromrank=0)
            else:
                msg = "Only shared objects using the group, row, and column "
                msg += "communicators can be demodulated"
                raise RuntimeError(msg)
        return

    @function_timer
    def _demodulate_metadata(self, obs, demod_obs):
        """Copy over and optionally downsample metadata"""

        demod_times = demod_obs.shared[self.times].data

        # Metadata dictionary

        for key, value in obs.items():
            if key in demod_obs:
                # Already demodulated
                continue
            if hasattr(key, "downsample"):
                demod_obs[key] = value.downsample(demod_times)
            else:
                demod_obs[key] = value

        # Other observation attributes

        for key, value in vars(obs).items():
            if key.startswith("_"):
                continue
            if hasattr(demod_obs, key):
                # Already demodulated
                continue
            if hasattr(value, "downsample"):
                setattr(demod_obs, key, value.downsample(demod_times))
            else:
                setattr(demod_obs, key, value)

        return

    @function_timer
    def _demodulate_detsets(self, obs, all_dets):
        """In order to force local detectors to remain on their original
        process, we create a detector set for each row of the process
        grid.
        """
        log = Logger.get()
        if obs.comm_col_size == 1:
            # One process row
            detsets = [all_dets]
        else:
            local_proc_dets = obs.comm_col.gather(obs.local_detectors, root=0)
            detsets = None
            if obs.comm_col_rank == 0:
                all_set = set(all_dets)
                detsets = list()
                for iprow, pdets in enumerate(local_proc_dets):
                    plocal = list()
                    for d in pdets:
                        if d in all_set:
                            plocal.append(d)
                    if len(plocal) == 0:
                        msg = f"obs {obs.name}, process row {iprow} has no"
                        msg += " good detectors.  This is inefficient..."
                        log.debug(msg)
                    detsets.append(plocal)
            detsets = obs.comm_col.bcast(detsets, root=0)

        demod_detsets = list()
        for dset in detsets:
            demod_detset = list()
            for det in dset:
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
        flags[:wkernel] |= self.demod_flag_mask
        flags[-wkernel:] |= self.demod_flag_mask
        new_flags = np.array(flags[offset % self.nskip :: self.nskip])
        return new_flags

    @function_timer
    def _demodulate_signal(
        self, data, obs, demod_obs, dets, lowpass, bandpass2f, bandpass4f
    ):
        """demodulate signal TOD"""

        for det in dets:
            # Get weights
            obs_data = data.select(obs_uid=obs.uid)
            self.stokes_weights.apply(obs_data, detectors=[det])
            weights = obs.detdata[self.stokes_weights.weights][det]
            # iweights = 1
            # qweights = eta * cos(2 * psi_det + 4 * psi_hwp)
            # uweights = eta * sin(2 * psi_det + 4 * psi_hwp)
            if self.stokes_weights.mode == "IQU":
                iweights, qweights, uweights = weights.T
            elif self.stokes_weights.mode == "QU":
                qweights, uweights = weights.T
                iweights = np.ones_like(qweights)
            if "QU" in self.mode:
                # remove polarization efficiency from the Q/U weights
                etainv = 1 / np.sqrt(qweights**2 + uweights**2)
                qweights = qweights * etainv
                uweights = uweights * etainv

            for flavor in self.det_data.split(";"):
                signal = obs.detdata[flavor][det]
                det_data = demod_obs.detdata[flavor]
                if "I" in self.mode:
                    det_data[f"demod0_{det}"] = lowpass(signal)
                if "QU" in self.mode:
                    bandpassed = bandpass4f(signal)
                    det_data[f"demod4r_{det}"] = lowpass(bandpassed * 2 * qweights)
                    det_data[f"demod4i_{det}"] = lowpass(bandpassed * 2 * uweights)
                if self.do_2f:
                    # Start by evaluating the 2f demodulation factors from the
                    # pointing matrix.  We use the half-angle formulas and some
                    # extra logic to identify the right branch
                    #
                    # |cos(psi/2)| and |sin(psi/2)|:
                    signal_demod2r = np.sqrt(0.5 * (1 + qweights))
                    signal_demod2i = np.sqrt(0.5 * (1 - qweights))
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
                    highpassed = bandpass2f(signal)
                    det_data[f"demod2r_{det}"] = lowpass(highpassed * signal_demod2r)
                    det_data[f"demod2i_{det}"] = lowpass(highpassed * signal_demod2i)

        return

    @function_timer
    def _demodulate_flags(self, obs, demod_obs, dets, wkernel, offset):
        """Demodulate and downsample flags"""

        shared_flags = obs.shared[self.shared_flags].data
        demod_shared_flags = self._demodulate_flag(shared_flags, wkernel, offset)
        demod_obs.shared[self.shared_flags].set(
            demod_shared_flags, offset=(0,), fromrank=0
        )

        input_det_flags = obs.local_detector_flags
        output_det_flags = dict()

        for det in dets:
            flags = obs.detdata[self.det_flags][det]
            # Downsample flags
            demod_flags = self._demodulate_flag(flags, wkernel, offset)
            for prefix in self.prefixes:
                demod_det = f"{prefix}_{det}"
                demod_obs.detdata[self.det_flags][demod_det] = demod_flags
                output_det_flags[demod_det] = input_det_flags[det]
        demod_obs.update_local_detector_flags(output_det_flags)
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
        bandpass2f,
        bandpass4f,
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
            "shared": [self.times],
            "detdata": [self.det_data],
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()


@trait_docs
class StokesWeightsDemod(Operator):
    """Compute the Stokes pointing weights for demodulated data"""

    allowed_modes = ("I", "QU", "IQU")

    API = Int(0, help="Internal interface version for this operator")

    mode = Unicode("IQU", help="The Stokes weights to generate")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    weights = Unicode(
        defaults.weights, help="Observation detdata key for output weights"
    )

    single_precision = Bool(False, help="If True, use 32bit float in output")

    detector_pointing_in = Instance(
        klass=Operator,
        allow_none=True,
        help="Pointing operator in the native Q/U frame, typically az/el.  "
        "Must be set if `detector_pointing_out` is set.  Has no effect if "
        " `detector_pointing_out` is not set.",
    )

    detector_pointing_out = Instance(
        klass=Operator,
        allow_none=True,
        help="Pointing operator for the desired frame, typically RA/Dec.  "
        "Requires `detector_pointing_in` to be set.",
    )

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        mode = proposal["value"]
        if mode not in self.allowed_modes:
            msg = f"Invalid mode (must be one of {self.allowed_modes})"
            raise traitlets.TraitError(msg)
        return mode

    @traitlets.validate("detector_pointing_in")
    def _check_detector_pointing_in(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "det_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("detector_pointing_out")
    def _check_detector_pointing_out(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "det_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @function_timer
    def _get_delta(self, data, ob, det):
        """Get the polarization angle in the input and output
        frames to rotate Q and U accordingly

        """
        if self.detector_pointing_out is None:
            return None

        if det.startswith("demod4r") or det.startswith("demod4i"):
            # Get input and output detector pointing
            ob_data = data.select(obs_name=ob.name)
            # detector pointing will short-circuit if detdata already has the required key
            reset = self.detector_pointing_in.quats == self.detector_pointing_out.quats
            if reset and self.detector_pointing_in.quats in ob_data.obs[0].detdata:
                del ob_data.obs[0].detdata[self.detector_pointing_in.quats]
            # Get input pointing
            self.detector_pointing_in.apply(ob_data, detectors=[det])
            quats_in = ob_data.obs[0].detdata[self.detector_pointing_in.quats][det]
            psi_in = qa.to_iso_angles(quats_in)[2]
            if reset and self.detector_pointing_out.quats in ob_data.obs[0].detdata:
                del ob_data.obs[0].detdata[self.detector_pointing_out.quats]
            # Get output pointing
            self.detector_pointing_out.apply(ob_data, detectors=[det])
            quats_out = ob_data.obs[0].detdata[self.detector_pointing_out.quats][det]
            psi_out = qa.to_iso_angles(quats_out)[2]
            if reset:
                # Purge the quaternions to avoid confusion later
                del ob_data.obs[0].detdata[self.detector_pointing_out.quats]
            # Get the difference in position angle
            delta = psi_out - psi_in
            delta = delta[:, np.newaxis]
        else:
            delta = None

        return delta

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        nnz = len(self.mode)

        if self.detector_pointing_in is None and self.detector_pointing_out is not None:
            raise RuntimeError(
                "You must set the input detector pointing with output pointing"
            )

        if self.single_precision:
            dtype = np.float32
        else:
            dtype = np.float64

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            if len(dets) == 0:
                continue

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
            if self.mode == "I":
                i_weights = ones
                q_weights = zeros
                u_weights = zeros
                no_weights = zeros
            elif self.mode == "QU":
                i_weights = np.column_stack([zeros, zeros])
                q_weights = np.column_stack([ones, zeros])
                u_weights = np.column_stack([zeros, ones])
                no_weights = np.column_stack([zeros, zeros])
            elif self.mode == "IQU":
                i_weights = np.column_stack([ones, zeros, zeros])
                q_weights = np.column_stack([zeros, ones, zeros])
                u_weights = np.column_stack([zeros, zeros, ones])
                no_weights = np.column_stack([zeros, zeros, zeros])

            for det in dets:
                props = obs.telescope.focalplane[det]
                if "pol_efficiency" in props.colnames:
                    eta = props["pol_efficiency"]
                else:
                    eta = 1.0

                # Check if we need to rotate Q/U weights between reference frames
                delta = self._get_delta(data, obs, det)

                if det.startswith("demod0"):
                    # Stokes I only
                    weights[det] = i_weights
                elif det.startswith("demod4r"):
                    # Stokes Q only
                    if delta is None:
                        weights[det] = q_weights * eta
                    else:
                        # Q' = Qcos(2psi) + Usin(2psi)
                        weights[det] = (
                            np.cos(2 * delta) * q_weights
                            + np.sin(2 * delta) * u_weights
                        ) * eta
                elif det.startswith("demod4i"):
                    # Stokes U only
                    if delta is None:
                        weights[det] = u_weights * eta
                    else:
                        # U' = Ucos(2psi) - Qsin(2psi)
                        weights[det] = (
                            np.cos(2 * delta) * u_weights
                            - np.sin(2 * delta) * q_weights
                        ) * eta
                else:
                    # Not an I/Q/U pseudo detector
                    weights[det] = no_weights

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
