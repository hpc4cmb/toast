# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import numpy as np
import scipy
import scipy.signal
import traitlets
from astropy import units as u

from .._libtoast import (
    template_offset_add_to_signal,
    template_offset_apply_diag_precond,
    template_offset_project_signal,
)
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import AlignedF64, Logger, rate_from_times
from .amplitudes import Amplitudes
from .template import Template


@trait_docs
class Offset(Template):
    """This class represents noise fluctuations as a step function.

    Every process stores the amplitudes for its local data, which is disjoint from the
    amplitudes on other processes.  We project amplitudes one detector at a time, and
    so we arrange our template amplitudes in "detector major" order and store offsets
    into this for each observation.

    """

    # Notes:  The TraitConfig base class defines a "name" attribute.  The Template
    # class (derived from TraitConfig) defines the following traits already:
    #    data             : The Data instance we are working with
    #    view             : The timestream view we are using
    #    det_data         : The detector data key with the timestreams
    #    det_flags        : Optional detector solver flags
    #    det_flag_mask    : Bit mask for detector solver flags
    #

    step_time = Quantity(10000.0 * u.second, help="Time per baseline step")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

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
        help="Use detector PSDs to build the noise prior and preconditioner",
    )

    precond_width = Int(20, help="Preconditioner width in terms of offsets / baselines")

    use_python = Bool(False, help="If True, use python implementation")

    @traitlets.validate("precond_width")
    def _check_precond_width(self, proposal):
        w = proposal["value"]
        if w < 1:
            raise traitlets.TraitError("Preconditioner width should be >= 1")
        return w

    @traitlets.validate("good_fraction")
    def _check_good_fraction(self, proposal):
        f = proposal["value"]
        if f < 0.0 or f > 1.0:
            raise traitlets.TraitError("good_fraction should be a value from 0 to 1")
        return f

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def clear(self):
        """Delete the underlying C-allocated memory."""
        if hasattr(self, "_offsetvar"):
            del self._offsetvar
        if hasattr(self, "_offsetvar_raw"):
            self._offsetvar_raw.clear()
            del self._offsetvar_raw

    def __del__(self):
        self.clear()

    @function_timer
    def _initialize(self, new_data):
        log = Logger.get()
        # This function is called whenever a new data trait is assigned to the template.
        # Clear any C-allocated buffers from previous uses.
        self.clear()

        if self.use_python and self.use_accel:
            raise RuntimeError("Cannot use accelerator with pure python implementation")

        # Compute the step boundaries for every observation and the number of
        # amplitude values on this process.  Every process only stores amplitudes
        # for its locally assigned data.

        if self.use_noise_prior and self.noise_model is None:
            raise RuntimeError("cannot use noise prior without specifying noise_model")

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
            step_length = self._step_length(
                self.step_time.to_value(u.second), self._obs_rate[iob]
            )

            # Track number of offset amplitudes per view, per det.
            ob_views = list()
            for view_slice in ob.view[self.view]:
                view_len = None
                if view_slice.start < 0:
                    # This is a view of the whole obs
                    view_len = ob.n_local_samples
                else:
                    view_len = view_slice.stop - view_slice.start
                view_n_amp = view_len // step_length
                if view_n_amp * step_length < view_len:
                    view_n_amp += 1
                ob_views.append(view_n_amp)
            self._obs_views[iob] = np.array(ob_views, dtype=np.int64)

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
                    tbase = self.step_time.to_value(u.second)
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
        self._amp_flags = np.zeros(self._n_local, dtype=bool)

        # Here we track the variance of the offsets based on the detector noise weights
        # and the number of unflagged / good samples per offset.
        self._offsetvar_raw = AlignedF64.zeros(self._n_local)
        self._offsetvar = self._offsetvar_raw.array()

        offset = 0
        for det in self._all_dets:
            for iob, ob in enumerate(new_data.obs):
                if det not in ob.local_detectors:
                    continue

                # "Noise weight" (time-domain inverse variance)
                detnoise = 1.0
                if self.noise_model is not None:
                    detnoise = ob[self.noise_model].detector_weight(det)

                # The step length for this observation
                step_length = self._step_length(
                    self.step_time.to_value(u.second), self._obs_rate[iob]
                )

                # Loop over views
                views = ob.view[self.view]
                for ivw, vw in enumerate(views):
                    view_samples = None
                    if vw.start < 0:
                        # This is a view of the whole obs
                        view_samples = ob.n_local_samples
                    else:
                        view_samples = vw.stop - vw.start
                    n_amp_view = self._obs_views[iob][ivw]

                    # Move this loop to compiled code if it is slow...
                    # Note:  we are building the offset amplitude *variance*, which is
                    # why the "noise weight" (inverse variance) is in the denominator.
                    if self.det_flags is None:
                        voff = 0
                        for amp in range(n_amp_view):
                            amplen = step_length
                            if amp == n_amp_view - 1:
                                amplen = view_samples - voff
                            self._offsetvar[offset + amp] = 1.0 / (detnoise * amplen)
                            voff += step_length
                    else:
                        flags = views.detdata[self.det_flags][ivw]
                        voff = 0
                        for amp in range(n_amp_view):
                            amplen = step_length
                            if amp == n_amp_view - 1:
                                amplen = view_samples - voff
                            n_good = amplen - np.count_nonzero(
                                flags[det][voff : voff + amplen] & self.det_flag_mask
                            )
                            if (n_good / amplen) > self.good_fraction:
                                # Keep this
                                self._offsetvar[offset + amp] = 1.0 / (
                                    detnoise * n_good
                                )
                            else:
                                # Flag it
                                self._offsetvar[offset + amp] = 0.0
                                self._amp_flags[offset + amp] = True
                            voff += step_length
                    offset += n_amp_view

        # Compute the amplitude noise filter and preconditioner for each detector
        # and each view.  The "noise filter" is the real-space inverse amplitude
        # covariance, which is constructed from the Fourier domain amplitude PSD.
        #
        # The preconditioner is either a diagonal one using the amplitude variance,
        # or is a banded one using the amplitude covariance plus the diagonal term.

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

                    offset_psd = self._get_offset_psd(
                        ob[self.noise_model],
                        self._freq[iob],
                        self.step_time.to_value(u.second),
                        det,
                    )

                    # "Noise weight" (time-domain inverse variance)
                    detnoise = ob[self.noise_model].detector_weight(det)

                    # Log version of offset PSD and its inverse for interpolation
                    logfreq = np.log(self._freq[iob])
                    logpsd = np.log(offset_psd)
                    logfilter = np.log(1.0 / offset_psd)

                    # Compute the list of filters and preconditioners (one per view)
                    # For this detector.

                    self._filters[iob][det] = list()
                    self._precond[iob][det] = list()

                    # Loop over views
                    views = ob.view[self.view]
                    for ivw, vw in enumerate(views):
                        view_samples = None
                        if vw.start < 0:
                            # This is a view of the whole obs
                            view_samples = ob.n_local_samples
                        else:
                            view_samples = vw.stop - vw.start
                        n_amp_view = self._obs_views[iob][ivw]
                        offsetvar_slice = self._offsetvar[offset : offset + n_amp_view]

                        filterlen = 2
                        while filterlen < 2 * n_amp_view:
                            filterlen *= 2
                        filterfreq = np.fft.rfftfreq(
                            filterlen, self.step_time.to_value(u.second)
                        )

                        # Recall that the "noise filter" is the inverse amplitude
                        # covariance, which is why we are using 1/PSD.  Also note that
                        # the truncate function shifts the filter to be symmetric about
                        # the center, which is needed for use with scipy.signal.convolve
                        # If we move this application back to compiled FFT based
                        # methods, we should instead keep this filter in the fourier
                        # domain.

                        noisefilter = self._truncate(
                            np.fft.irfft(
                                self._interpolate_psd(filterfreq, logfreq, logfilter)
                            )
                        )

                        self._filters[iob][det].append(noisefilter)

                        # Build the preconditioner
                        lower = None
                        preconditioner = None

                        if self.precond_width == 1:
                            # We are using a Toeplitz preconditioner.  The first row
                            # of the matrix is the inverse FFT of the offset PSD,
                            # with an added zero-lag component from the detector
                            # weight.  NOTE:  the truncate function shifts the real
                            # space filter to the center of the vector.
                            preconditioner = self._truncate(
                                np.fft.irfft(
                                    self._interpolate_psd(filterfreq, logfreq, logpsd)
                                )
                            )
                            icenter = preconditioner.size // 2
                            preconditioner[icenter] += 1.0 / detnoise
                        else:
                            # We are using a banded matrix for the preconditioner.
                            # This contains a Toeplitz component from the inverse
                            # offset variance in the LHS, and another diagonal term
                            # from the individual offset variance.
                            #
                            # NOTE:  Instead of directly solving x = M^{-1} b, we do
                            # not invert "M" and solve M x = b using the Cholesky
                            # decomposition of M (*not* M^{-1}).
                            icenter = noisefilter.size // 2
                            wband = min(self.precond_width, icenter)
                            precond_width = max(
                                wband, min(self.precond_width, n_amp_view)
                            )
                            preconditioner = np.zeros(
                                [precond_width, n_amp_view], dtype=np.float64
                            )
                            preconditioner[0, :] = 1.0 / offsetvar_slice
                            preconditioner[:wband, :] += np.repeat(
                                noisefilter[icenter : icenter + wband, np.newaxis],
                                n_amp_view,
                                1,
                            )
                            lower = True
                            preconditioner = scipy.linalg.cholesky_banded(
                                preconditioner,
                                overwrite_ab=True,
                                lower=lower,
                                check_finite=True,
                            )
                        self._precond[iob][det].append((preconditioner, lower))
                        offset += n_amp_view
        log.verbose(f"Offset variance = {self._offsetvar}")
        return

    def __del__(self):
        if hasattr(self, "_offsetvar"):
            del self._offsetvar
        if hasattr(self, "_offsetvar_raw"):
            self._offsetvar_raw.clear()
            del self._offsetvar_raw

    # Helper functions for noise / preconditioner calculations

    def _interpolate_psd(self, x, lfreq, lpsd):
        result = np.zeros(x.size)
        good = np.abs(x) > 1e-10
        logx = np.log(np.abs(x[good]))
        logresult = np.interp(logx, lfreq, lpsd)
        result[good] = np.exp(logresult)
        return result

    def _truncate(self, noisefilter, lim=1e-4):
        icenter = noisefilter.size // 2
        ind = np.abs(noisefilter[:icenter]) > np.abs(noisefilter[0]) * lim
        icut = np.argwhere(ind)[-1][0]
        if icut % 2 == 0:
            icut += 1
        noisefilter = np.roll(noisefilter, icenter)
        noisefilter = noisefilter[icenter - icut : icenter + icut + 1]
        return noisefilter

    def _get_offset_psd(self, noise, freq, step_time, det):
        """Compute the PSD of the baseline offsets."""
        psdfreq = noise.freq(det).to_value(u.Hz)
        psd = noise.psd(det).to_value(u.K**2 * u.second)
        rate = noise.rate(det).to_value(u.Hz)

        # Remove the white noise component from the PSD
        psd = psd.copy()
        psd -= np.amin(psd[psdfreq > 1.0])
        psd[psd < 1e-30] = 1e-30

        # The calculation of `offset_psd` is from Keihänen, E. et al:
        # "Making CMB temperature and polarization maps with Madam",
        # A&A 510:A57, 2010
        logfreq = np.log(psdfreq)
        logpsd = np.log(psd)

        def g(x):
            bad = np.abs(x) < 1e-10
            good = np.logical_not(bad)
            arg = np.pi * x[good]
            result = bad.astype(np.float64)
            result[good] = (np.sin(arg) / arg) ** 2
            return result

        tbase = step_time
        fbase = 1.0 / tbase

        offset_psd = np.zeros_like(freq)
        offset_psd[:] += self._interpolate_psd(freq, logfreq, logpsd) * g(freq * tbase)

        for m in range(1, 2):
            offset_psd[:] += self._interpolate_psd(
                freq + m * fbase, logfreq, logpsd
            ) * g(freq * tbase + m)
            offset_psd[:] += self._interpolate_psd(
                freq - m * fbase, logfreq, logpsd
            ) * g(freq * tbase - m)

        offset_psd *= fbase
        return offset_psd

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(self.data.comm, self._n_global, self._n_local)
        z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _step_length(self, stime, rate):
        return int(stime * rate + 0.5)

    @function_timer
    def _add_to_signal(self, detector, amplitudes):
        log = Logger.get()
        amp_offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            det_indx = ob.detdata[self.det_data].indices([detector])

            # The step length for this observation
            step_length = self._step_length(
                self.step_time.to_value(u.second), self._obs_rate[iob]
            )

            # The number of amplitudes in each view
            n_amp_views = self._obs_views[iob]

            if self.use_python:
                self._py_add_to_signal(
                    step_length,
                    amp_offset,
                    n_amp_views,
                    amplitudes.local,
                    det_indx,
                    ob.detdata[self.det_data].data,
                    ob.intervals[self.view].data,
                )
            else:
                template_offset_add_to_signal(
                    step_length,
                    amp_offset,
                    n_amp_views,
                    amplitudes.local,
                    det_indx[0],
                    ob.detdata[self.det_data].data,
                    ob.intervals[self.view].data,
                    self.use_accel,
                )
            amp_offset += np.sum(n_amp_views)

    @function_timer
    def _project_signal(self, detector, amplitudes):
        log = Logger.get()
        amp_offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            det_indx = ob.detdata[self.det_data].indices([detector])
            if self.det_flags is not None:
                flag_indx = ob.detdata[self.det_flags].indices([detector])
                flag_data = ob.detdata[self.det_flags].data
            else:
                flag_indx = np.array([-1], dtype=np.int32)
                flag_data = np.zeros(1, dtype=np.uint8)
            # The step length for this observation
            step_length = self._step_length(
                self.step_time.to_value(u.second), self._obs_rate[iob]
            )

            # The number of amplitudes in each view
            n_amp_views = self._obs_views[iob]

            if self.use_python:
                self._py_project_signal(
                    det_indx,
                    ob.detdata[self.det_data].data,
                    flag_indx,
                    flag_data,
                    self.det_flag_mask,
                    step_length,
                    amp_offset,
                    n_amp_views,
                    amplitudes.local,
                    ob.intervals[self.view].data,
                )
            else:
                template_offset_project_signal(
                    det_indx[0],
                    ob.detdata[self.det_data].data,
                    flag_indx[0],
                    flag_data,
                    self.det_flag_mask,
                    step_length,
                    amp_offset,
                    n_amp_views,
                    amplitudes.local,
                    ob.intervals[self.view].data,
                    self.use_accel,
                )
            amp_offset += np.sum(n_amp_views)

    @function_timer
    def _add_prior(self, amplitudes_in, amplitudes_out):
        if not self.use_noise_prior:
            # Not using the noise prior term, nothing to accumulate to output.
            return
        if self.use_accel:
            raise NotImplementedError(
                "offset template add_prior on accelerator not implemented"
            )
        for det in self._all_dets:
            offset = self._det_start[det]
            for iob, ob in enumerate(self.data.obs):
                if det not in ob.local_detectors:
                    continue
                for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                    n_amp_view = self._obs_views[iob][ivw]
                    amp_slice = slice(offset, offset + n_amp_view, 1)
                    amps_in = amplitudes_in.local[amp_slice]
                    amps_out = amplitudes_out.local[amp_slice]
                    amps_out[:] += scipy.signal.convolve(
                        amps_in, self._filters[iob][det][ivw], mode="same"
                    )
                    offset += n_amp_view

    @function_timer
    def _apply_precond(self, amplitudes_in, amplitudes_out):
        if self.use_noise_prior:
            if self.use_accel:
                raise NotImplementedError(
                    "offset template precond on accelerator not implemented"
                )
            # Our design matrix includes a term with the inverse offset covariance.
            # This means that our preconditioner should include this term as well.
            for det in self._all_dets:
                offset = self._det_start[det]
                for iob, ob in enumerate(self.data.obs):
                    if det not in ob.local_detectors:
                        continue
                    # Loop over views
                    views = ob.view[self.view]
                    for ivw, vw in enumerate(views):
                        view_samples = None
                        if vw.start < 0:
                            # This is a view of the whole obs
                            view_samples = ob.n_local_samples
                        else:
                            view_samples = vw.stop - vw.start

                        n_amp_view = self._obs_views[iob][ivw]
                        amp_slice = slice(offset, offset + n_amp_view, 1)

                        amps_in = amplitudes_in.local[amp_slice]
                        amps_out = None
                        if self.precond_width <= 1:
                            # We are using a Toeplitz preconditioner.
                            # scipy.signal.convolve will use either `convolve` or
                            # `fftconvolve` depending on the size of the inputs
                            amps_out = scipy.signal.convolve(
                                amps_in, self._precond[iob][det][ivw][0], mode="same"
                            )
                        else:
                            # Use pre-computed Cholesky decomposition.  Note that this
                            # is the decomposition of the actual preconditioner (not
                            # its inverse), since we are solving Mx=b.
                            amps_out = scipy.linalg.cho_solve_banded(
                                self._precond[iob][det][ivw],
                                amps_in,
                                overwrite_b=False,
                                check_finite=True,
                            )
                        amplitudes_out.local[amp_slice] = amps_out
        else:
            # Since we do not have a noise filter term in our LHS, our diagonal
            # preconditioner is just the application of offset variance.
            if self.use_python:
                self._py_apply_diag_precond(
                    self._offsetvar,
                    amplitudes_in.local,
                    amplitudes_out.local,
                )
            else:
                template_offset_apply_diag_precond(
                    self._offsetvar,
                    amplitudes_in.local,
                    amplitudes_out.local,
                    self.use_accel,
                )
        return

    def _supports_accel(self):
        return True

    def _py_add_to_signal(
        self,
        step_length,
        amp_offset,
        n_amp_views,
        amplitudes,
        data_index,
        det_data,
        intr_data,
    ):
        """Internal python implementation for comparison testing."""
        offset = amp_offset
        for ivw, vw in enumerate(intr_data):
            samples = slice(vw.first, vw.last + 1, 1)
            sampidx = np.arange(vw.first, vw.last + 1, dtype=np.int64)
            amp_vals = np.array(
                [amplitudes[offset + x] for x in (sampidx // step_length)]
            )
            det_data[data_index[0], samples] += amp_vals
            offset += n_amp_views[ivw]

    def _py_project_signal(
        self,
        data_index,
        det_data,
        flag_index,
        flag_data,
        flag_mask,
        step_length,
        amp_offset,
        n_amp_views,
        amplitudes,
        intr_data,
    ):
        """Internal python implementation for comparison testing."""
        offset = amp_offset
        for ivw, vw in enumerate(intr_data):
            samples = slice(vw.first, vw.last + 1, 1)
            ampidx = (
                offset + np.arange(vw.first, vw.last + 1, dtype=np.int64) // step_length
            )
            ddata = det_data[data_index[0]][samples]
            if flag_index[0] >= 0:
                # We have detector flags
                ddata = np.array(
                    ((flag_data[flag_index[0]] & flag_mask) == 0), dtype=np.float64
                )
                ddata *= det_data[data_index[0]][samples]
            np.add.at(amplitudes, ampidx, ddata)
            offset += n_amp_views[ivw]

    def _py_apply_diag_precond(self, offset_var, amp_in, amp_out):
        """Internal python implementation for comparison testing."""
        amp_out[:] = amp_in
        amp_out *= offset_var
