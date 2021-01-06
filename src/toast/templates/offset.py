# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import numpy as np

import scipy

from ..utils import Logger, rate_from_times, AlignedF32

from ..timing import function_timer

from ..mpi import MPI

from ..traits import trait_docs, Int, Unicode, Bool, Instance, Float

from ..data import Data

from .template import Template, Amplitudes

from .._libtoast import template_offset_add_to_signal, template_offset_project_signal


@trait_docs
class Offset(Template):
    """This class represents noise fluctuations as a step function.

    Every process stores the offsets for its local data.  Although our data is arranged
    in observations and then in terms of detectors, we will often be projecting our
    template for a single detector at a time.  Because of this, we will arrange our
    template amplitudes in "detector major" order and store offsets into this for each
    observation.

    """

    # Notes:  The TraitConfig base class defines a "name" attribute.  The Template
    # class (derived from TraitConfig) defines the following traits already:
    #    data             : The Data instance we are working with
    #    view             : The timestream view we are using
    #    det_data         : The detector data key with the timestreams
    #    flags            : Optional detector solver flags
    #    flag_mask        : Bit mask for detector solver flags
    #

    step_time = Float(10000.0, help="Seconds per baseline step")

    times = Unicode("times", help="Observation shared key for timestamps")

    noise_model = Unicode(
        None,
        allow_none=True,
        help="Observation key containing the optional noise model",
    )

    good_fraction = Float(
        0.5,
        help="Fraction of unflagged samples needed to keep a given offset amplitude",
    )

    use_noise_prior = Bool(
        False,
        help="Construct the offset noise covariance and use it for a noise prior and as a preconditioner",
    )

    precond_width = Int(20, help="Preconditioner width in terms of offsets / baselines")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _initialize(self, new_data):
        # Compute the step boundaries for every observation and the number of
        # amplitude values on this process.  Every process only stores amplitudes
        # for its locally assigned data.

        # Use this as an "Ordered Set".  We want the unique detectors on this process,
        # but sorted in order of occurrence.
        all_dets = OrderedDict()

        # Amplitude lengths of all views for each obs
        self._obs_views = dict()

        # Sample rate for each obs.
        self._obs_rate = dict()

        # Frequency bins for the noise prior for each obs.
        self._freq = dict()

        for iob, ob in enumerate(new_data.obs):
            # Compute sample rate from timestamps
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(ob.shared[self.times])
            self._obs_rate[iob] = rate

            # The step length for this observation
            step_length = int(self.step_time * self._obs_rate[iob])

            # Track number of offset amplitudes per view.
            self._obs_views[iob] = list()
            for view_slice in ob.view[self.view]:
                slice_len = None
                if view_slice.start is None:
                    # This is a view of the whole obs
                    slice_len = ob.n_local_samples
                else:
                    slice_len = view_slice.stop - view_slice.start
                view_n_amp = slice_len // step_length
                self._obs_views[iob].append(view_n_amp)

            # The noise model.
            if self.noise_model is not None:
                if self.noise_model not in ob:
                    msg = "Observation {}:  noise model {} does not exist".format(
                        ob.name, self.noise_model
                    )
                    log.error(msg)
                    raise RuntimeError(msg)

                # Determine the binning for the noise prior
                if self.use_noise_prior:
                    obstime = ob.shared[self.times][-1] - ob.shared[self.times][0]
                    tbase = step_length
                    fbase = 1.0 / tbase
                    powmin = np.floor(np.log10(1 / obstime)) - 1
                    powmax = min(np.ceil(np.log10(1 / tbase)) + 2, self._obs_rate[iob])
                    self._freq[iob] = np.logspace(powmin, powmax, 1000)

            # Build up detector list
            for d in ob.local_detectors:
                if d not in all_dets:
                    all_dets[d] = None

        self._all_dets = list(all_dets.keys())

        # Go through the data one local detector at a time and compute the offsets into
        # the amplitudes.

        self._det_start = dict()

        offset = 0
        for det in self._all_dets:
            self._det_start[det] = offset
            for iob, ob in enumerate(new_data.obs):
                if det not in ob.local_detectors:
                    continue
                offset += np.sum(self._obs_views[iob])

        # Now we know the total number of amplitudes.

        self._n_local = offset
        if new_data.comm.comm_world is None:
            self._n_global = self._n_local
        else:
            self._n_global = new_data.comm.comm_world.allreduce(
                self._n_local, op=MPI.SUM
            )

        # Now that we know the number of amplitudes, we go through the solver flags
        # and determine what amplitudes, if any, are poorly constrained.  These are
        # stored internally as a bool array, and used when constructing a new
        # Amplitudes object.  We also compute and store the variance of each amplitude,
        # based on the noise weight of the detector and the number of flagged samples.

        # Boolean flags
        self._amp_flags = np.zeros(self._n_local, dtype=np.bool)

        # For the sigmasq values (offset / baseline variance), we have one per
        # amplitude, which can approach the size of the time ordered data.  Store these
        # in C-allocated memory as 32bit float.
        self._sigmasq_raw = AlignedF32.zeros(self._n_local)
        self._sigmasq = self._sigmasq_raw.array()

        offset = 0
        for det in self._all_dets:
            for iob, ob in enumerate(new_data.obs):
                if det not in ob.local_detectors:
                    continue

                # Noise weight
                detnoise = 1.0
                if self.noise_model is not None:
                    detnoise = ob[self.noise_model].detector_weight(det)

                # The step length for this observation
                step_length = int(self.step_time * self._obs_rate[iob])

                # Loop over views
                views = ob.view[self.view]
                for ivw, vw in enumerate(views):
                    view_samples = None
                    if vw.start is None:
                        # This is a view of the whole obs
                        view_samples = ob.n_local_samples
                    else:
                        view_samples = vw.stop - vw.start
                    n_amp_view = slice_len // step_length

                    # Move this loop to compiled code if it is slow
                    if self.flags is None:
                        voff = 0
                        for amp in range(n_amp_view):
                            amplen = step_length
                            if amp == n_amp_view - 1:
                                amplen = view_samples - voff
                            self._sigmasq[offset + amp] = 1.0 / (detnoise * amplen)
                    else:
                        flags = views.detdata[self.flags][ivw]
                        voff = 0
                        for amp in range(n_amp_view):
                            amplen = step_length
                            if amp == n_amp_view - 1:
                                amplen = view_samples - voff
                            n_good = amplen - np.count_nonzero(
                                flags[det][voff : voff + amplen] & self.flag_mask
                            )
                            if (n_good / amplen) > self.good_fraction:
                                # Keep this
                                self._sigmasq[offset + amp] = 1.0 / (detnoise * n_good)
                            else:
                                # Flag it
                                self._sigmasq[offset + amp] = 0.0
                                self._amp_flags[offset + amp] = True
                            voff += step_length
                    offset += n_amp_view

        # Compute the amplitude noise filter and preconditioner for each detector
        # and each view.

        self._filters = dict()
        self._precond = dict()

        if self.use_noise_prior:
            offset = 0
            for det in self._all_dets:
                for iob, ob in enumerate(new_data.obs):
                    if det not in ob.local_detectors:
                        continue
                    if iob not in self._filters:
                        self._filters[iob] = dict()
                        self._precond[iob] = dict()
                    if self.noise_model is not None:
                        # We have noise information.  Get the PSD describing noise
                        # correlations between offset amplitudes for this observation.
                        offset_psd = self._get_offset_psd(
                            ob[self.noise_model], self._freq[iob], self.step_time, det
                        )

                        # Log version of offset PSD for interpolation
                        logfreq = np.log(freq)
                        logpsd = np.log(offset_psd)
                        logfilter = np.log(1 / offset_psd)

                        # Helper functions
                        def _interpolate(x, psd):
                            result = np.zeros(x.size)
                            good = np.abs(x) > 1e-10
                            logx = np.log(np.abs(x[good]))
                            logresult = np.interp(logx, logfreq, psd)
                            result[good] = np.exp(logresult)
                            return result

                        def _truncate(noisefilter, lim=1e-4):
                            icenter = noisefilter.size // 2
                            ind = (
                                np.abs(noisefilter[:icenter])
                                > np.abs(noisefilter[0]) * lim
                            )
                            icut = np.argwhere(ind)[-1][0]
                            if icut % 2 == 0:
                                icut += 1
                            noisefilter = np.roll(noisefilter, icenter)
                            noisefilter = noisefilter[
                                icenter - icut : icenter + icut + 1
                            ]
                            return noisefilter

                        # Compute the list of filters and preconditioners (one per view)
                        # For this detector.
                        self._filters[iob][det] = list()
                        self._precond[iob][det] = list()

                        # Loop over views
                        views = ob.view[self.view]
                        for ivw, vw in enumerate(views):
                            view_samples = None
                            if vw.start is None:
                                # This is a view of the whole obs
                                view_samples = ob.n_local_samples
                            else:
                                view_samples = vw.stop - vw.start
                            n_amp_view = self._obs_view[iob][ivw]
                            sigmasq_slice = self._sigmasq[offset : offset + n_amp_view]

                            # nstep = offset_slice.stop - offset_slice.start

                            filterlen = n_amp_view * 2 + 1
                            filterfreq = np.fft.rfftfreq(filterlen, self.step_time)
                            noisefilter = _truncate(
                                np.fft.irfft(_interpolate(filterfreq, logfilter))
                            )
                            self._filters[iob][det].append(noisefilter)

                            # Build the band-diagonal preconditioner
                            lower = None
                            if self.precond_width <= 1:
                                # Compute C_a prior
                                preconditioner = _truncate(
                                    np.fft.irfft(_interpolate(filterfreq, logpsd))
                                )
                            else:
                                # Compute Cholesky decomposition prior
                                wband = min(self.precond_width, noisefilter.size // 2)
                                precond_width = max(
                                    wband, min(self.precond_width, n_amp_view)
                                )
                                icenter = noisefilter.size // 2
                                preconditioner = np.zeros(
                                    [precond_width, nstep], dtype=np.float64
                                )
                                preconditioner[0] = sigmasq_slice
                                preconditioner[:wband, :] += np.repeat(
                                    noisefilter[icenter : icenter + wband, np.newaxis],
                                    n_amp_view,
                                    1,
                                )
                                lower = True
                                scipy.linalg.cholesky_banded(
                                    preconditioner,
                                    overwrite_ab=True,
                                    lower=lower,
                                    check_finite=True,
                                )
                            self._precond[iob][det].append((preconditioner, lower))
                            offset += n_amp_view
        return

    def __del__(self):
        if hasattr(self, "_sigmasq"):
            del self._sigmasq
        if hasattr(self, "_sigmasq_raw"):
            self._sigmasq_raw.clear()
            del self._sigmasq_raw

    @staticmethod
    def _get_offset_psd(noise, freq, step_time, det):
        """Compute the PSD of the baseline offsets."""
        psdfreq = noise.freq(det)
        psd = noise.psd(det)
        rate = noise.rate(det)
        # Remove the white noise component from the PSD
        psd = psd.copy() * np.sqrt(rate)
        psd -= np.amin(psd[psdfreq > 1.0])
        psd[psd < 1e-30] = 1e-30

        # The calculation of `offset_psd` is from Keihänen, E. et al:
        # "Making CMB temperature and polarization maps with Madam",
        # A&A 510:A57, 2010
        logfreq = np.log(psdfreq)
        logpsd = np.log(psd)

        def interpolate_psd(x):
            result = np.zeros(x.size)
            good = np.abs(x) > 1e-10
            logx = np.log(np.abs(x[good]))
            logresult = np.interp(logx, logfreq, logpsd)
            result[good] = np.exp(logresult)
            return result

        def g(x):
            bad = np.abs(x) < 1e-10
            good = np.logical_not(bad)
            arg = np.pi * x[good]
            result = bad.astype(np.float64)
            result[good] = (np.sin(arg) / arg) ** 2
            return result

        tbase = step_time
        fbase = 1.0 / tbase
        offset_psd = interpolate_psd(freq) * g(freq * tbase)
        for m in range(1, 2):
            offset_psd += interpolate_psd(freq + m * fbase) * g(freq * tbase + m)
            offset_psd += interpolate_psd(freq - m * fbase) * g(freq * tbase - m)
        offset_psd *= fbase
        return offset_psd

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(self.data.comm.comm_world, self._n_global, self._n_local)
        z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    @function_timer
    def _add_to_signal(self, detector, amplitudes):
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            # The step length for this observation
            step_length = int(self.step_time * self._obs_rate[iob])
            for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                n_amp_view = self._obs_views[iob][ivw]
                template_offset_add_to_signal(
                    step_length,
                    amplitudes.local[offset : offset + n_amp_view],
                    vw[detector],
                )
                offset += n_amp_view

    @function_timer
    def _project_signal(self, detector, amplitudes):
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            # The step length for this observation
            step_length = int(self.step_time * self._obs_rate[iob])
            for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                n_amp_view = self._obs_views[iob][ivw]
                template_offset_project_signal(
                    step_length,
                    vw[detector],
                    amplitudes.local[offset : offset + n_amp_view],
                )
                offset += n_amp_view

    @function_timer
    def _add_prior(self, amplitudes_in, amplitudes_out):
        if self.noise_model is None:
            # No noise model is specified, so no prior is used.
            return
        for det in self._all_dets:
            offset = self._det_start[detector]
            for iob, ob in enumerate(self.data.obs):
                if det not in ob.local_detectors:
                    continue
                for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                    n_amp_view = self._obs_views[iob][ivw]
                    amps_in = amplitudes_in[offset : offset + n_amp_view]
                    amps_out = amplitudes_out[offset : offset + n_amp_view]
                    amps_out[:] += scipy.signal.convolve(
                        amps_in, self._filters[iob][det][ivw], mode="same"
                    )
                    offset += n_amp_view

    @function_timer
    def _apply_precond(self, amplitudes_in, amplitudes_out):
        if self.use_noise_prior:
            # C_a preconditioner
            for det in self._all_dets:
                offset = self._det_start[det]
                for iob, ob in enumerate(new_data.obs):
                    if det not in ob.local_detectors:
                        continue
                    # Loop over views
                    views = ob.view[self.view]
                    for ivw, vw in enumerate(views):
                        view_samples = None
                        if vw.start is None:
                            # This is a view of the whole obs
                            view_samples = ob.n_local_samples
                        else:
                            view_samples = vw.stop - vw.start

                        n_amp_view = self._obs_view[iob][ivw]
                        amp_slice = slice(offset, offset + n_amp_view, 1)

                        amps_in = amplitudes_in[amp_slice]
                        amps_out = None
                        if self.precond_width <= 1:
                            # Use C_a prior
                            # scipy.signal.convolve will use either `convolve` or
                            # `fftconvolve` depending on the size of the inputs
                            amps_out = scipy.signal.convolve(
                                amps_in, self._precond[iob][det][ivw], mode="same"
                            )
                        else:
                            # Use pre-computed Cholesky decomposition
                            amps_out = scipy.linalg.cho_solve_banded(
                                self._precond[iob][det][ivw],
                                amps_in,
                                overwrite_b=False,
                                check_finite=True,
                            )
                        amplitudes_out[amp_slice] = amps_out
        else:
            # Diagonal preconditioner
            amplitudes_out.local[:] = amplitudes_in.local
            amplitudes_out.local *= self._sigmasq
        return
