# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import scipy

from ..utils import Logger, rate_from_times

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

    precond_width = Int(20, help="Preconditioner width in terms of offsets / baselines")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # Offset covariance and preconditioner for each obs and detector.
        self._freq = dict()
        self._filters = dict()
        self._precond = dict()

        for iob, ob in enumerate(new_data.obs):
            # Compute sample rate from timestamps
            self._obs_rate[iob] = rate_from_times(ob.shared[self.times])

            # The step length for this observation
            step_length = self.step_time * self._obs_rate[iob]

            # Track number of offset amplitudes per view.
            self._obs_views[iob] = list()
            for view_slice in ob.view[self.view]:
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
                self._filters[iob] = dict()
                self._precond[iob] = dict()

                # Determine the binning for the noise prior
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
        # the amplitudes.  Also compute the amplitude noise filter and preconditioner
        # for each detector and each interval / view.

        self._det_start = dict()

        offset = 0
        for det in self._all_dets:
            self._det_start[det] = offset
            for iob, ob in enumerate(new_data.obs):
                if det not in ob.local_detectors:
                    continue
                if self.noise_model is not None:
                    offset_psd = self._get_offset_psd(
                        ob[self.noise_model], self._freq[iob], det
                    )
                    (
                        self._filters[iob][det],
                        self._precond[iob][det],
                    ) = self._get_filter_and_precond(
                        self._freq[iob], offset_psd, ob.view[self.view]
                    )
                offset += np.sum(self.obs_views[iob])

        self._n_local = offset
        if new_data.comm.comm_world is None:
            self._n_global = self._n_local
        else:
            self._n_global = new_data.comm.comm_world.allreduce(
                self._n_local, op=MPI.SUM
            )

        return

    @function_timer
    def _get_offset_psd(self, noise, freq, det):
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

        tbase = self.step_length
        fbase = 1 / tbase
        offset_psd = interpolate_psd(freq) * g(freq * tbase)
        for m in range(1, 2):
            offset_psd += interpolate_psd(freq + m * fbase) * g(freq * tbase + m)
            offset_psd += interpolate_psd(freq - m * fbase) * g(freq * tbase - m)
        offset_psd *= fbase
        return offset_psd

    @function_timer
    def _get_filter_and_precond(self, freq, offset_psd, view_slices):
        logfreq = np.log(freq)
        logpsd = np.log(offset_psd)
        logfilter = np.log(1 / offset_psd)

        def interpolate(x, psd):
            result = np.zeros(x.size)
            good = np.abs(x) > 1e-10
            logx = np.log(np.abs(x[good]))
            logresult = np.interp(logx, logfreq, psd)
            result[good] = np.exp(logresult)
            return result

        def truncate(noisefilter, lim=1e-4):
            icenter = noisefilter.size // 2
            ind = np.abs(noisefilter[:icenter]) > np.abs(noisefilter[0]) * lim
            icut = np.argwhere(ind)[-1][0]
            if icut % 2 == 0:
                icut += 1
            noisefilter = np.roll(noisefilter, icenter)
            noisefilter = noisefilter[icenter - icut : icenter + icut + 1]
            return noisefilter

        vw_filters = list()
        vw_precond = list()
        for offset_slice, sigmasqs in offset_slices:
            nstep = offset_slice.stop - offset_slice.start
            filterlen = nstep * 2 + 1
            filterfreq = np.fft.rfftfreq(filterlen, self.step_length)
            noisefilter = truncate(np.fft.irfft(interpolate(filterfreq, logfilter)))
            noisefilters.append(noisefilter)
            # Build the band-diagonal preconditioner
            if self.precond_width <= 1:
                # Compute C_a prior
                preconditioner = truncate(np.fft.irfft(interpolate(filterfreq, logpsd)))
            else:
                # Compute Cholesky decomposition prior
                wband = min(self.precond_width, noisefilter.size // 2)
                precond_width = max(wband, min(self.precond_width, nstep))
                icenter = noisefilter.size // 2
                preconditioner = np.zeros([precond_width, nstep], dtype=np.float64)
                preconditioner[0] = sigmasqs
                preconditioner[:wband, :] += np.repeat(
                    noisefilter[icenter : icenter + wband, np.newaxis], nstep, 1
                )
                lower = True
                scipy.linalg.cholesky_banded(
                    preconditioner, overwrite_ab=True, lower=lower, check_finite=True
                )
            preconditioners.append((preconditioner, lower))
        return noisefilters, preconditioners

    def _zeros(self):
        return Amplitudes(self.data.comm.comm_world, self._n_global, self._n_local)

    @function_timer
    def _add_to_signal(self, detector, amplitudes):
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if det not in ob.local_detectors:
                continue
            # The step length for this observation
            step_length = self.step_time * self._obs_rate[iob]
            for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                n_amp_view = self._obs_views[iob][ivw]
                template_offset_add_to_signal(
                    step_length, amplitudes.local[offset : offset + n_amp_view], vw
                )
                offset += n_amp_view

    @function_timer
    def _project_signal(self, detector, amplitudes):
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if det not in ob.local_detectors:
                continue
            # The step length for this observation
            step_length = self.step_time * self._obs_rate[iob]
            for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                n_amp_view = self._obs_views[iob][ivw]
                template_offset_project_signal(
                    step_length, vw, amplitudes.local[offset : offset + n_amp_view]
                )
                offset += n_amp_view

    @function_timer
    def _project_flags(self, detector, amplitudes):
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if det not in ob.local_detectors:
                continue
            # The step length for this observation
            step_length = self.step_time * self._obs_rate[iob]
            obview = ob.view[self.view]
            for ivw, vw_ in enumerate(ob.view[self.view].detdata[self.det_data]):
                n_amp_view = self._obs_views[iob][ivw]
                flags = np.array()
                template_offset_project_flags(
                    step_length,
                    flags,
                    amplitudes.local_flags[offset : offset + n_amp_view],
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
                        amps_in, self._filters[iob][det], mode="same"
                    )
                    offset += n_amp_view

    @function_timer
    def _apply_precond(self, amplitudes_in, amplitudes_out):
        # offset_amplitudes_in = amplitudes_in[self.name]
        # offset_amplitudes_out = amplitudes_out[self.name]
        # if self.use_noise_prior:
        #     # C_a preconditioner
        #     for iobs, obs in enumerate(self.data.obs):
        #         tod = obs["tod"]
        #         for det in tod.local_dets:
        #             slices = self.offset_slices[iobs][det]
        #             preconditioners = self.preconditioners[iobs][det]
        #             for (offsetslice, sigmasqs), preconditioner in zip(
        #                 slices, preconditioners
        #             ):
        #                 amps_in = offset_amplitudes_in[offsetslice]
        #                 if self.precond_width <= 1:
        #                     # Use C_a prior
        #                     # scipy.signal.convolve will use either `convolve` or `fftconvolve`
        #                     # depending on the size of the inputs
        #                     amps_out = scipy.signal.convolve(
        #                         amps_in, preconditioner, mode="same"
        #                     )
        #                 else:
        #                     # Use pre-computed Cholesky decomposition
        #                     amps_out = scipy.linalg.cho_solve_banded(
        #                         preconditioner,
        #                         amps_in,
        #                         overwrite_b=False,
        #                         check_finite=True,
        #                     )
        #                 offset_amplitudes_out[offsetslice] = amps_out
        # else:
        #     # Diagonal preconditioner
        #     offset_amplitudes_out[:] = offset_amplitudes_in
        #     for itemplate, iobs, det, todslice, sigmasq in self.offset_templates:
        #         offset_amplitudes_out[itemplate] *= sigmasq
        #
        return
