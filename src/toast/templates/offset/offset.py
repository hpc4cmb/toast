# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import json
import os
from collections import OrderedDict

import h5py
import numpy as np
import scipy
import scipy.signal
import traitlets
from astropy import units as u

from ...accelerator import ImplementationType
from ...mpi import MPI
from ...observation import default_values as defaults
from ...timing import function_timer
from ...traits import Bool, Float, Int, Quantity, Unicode, UseEnum, trait_docs
from ...utils import AlignedF64, Logger, rate_from_times
from ...vis import set_matplotlib_backend
from ..amplitudes import Amplitudes
from ..template import Template
from .kernels import (
    offset_add_to_signal,
    offset_apply_diag_precond,
    offset_project_signal,
)


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
    #    det_data_units   : The units of the detector data
    #    det_mask         : Bitmask for per-detector flagging
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

    debug_plots = Unicode(
        None,
        allow_none=True,
        help="If not None, make debugging plots in this directory",
    )

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

        # Compute the step boundaries for every observation and the number of
        # amplitude values on this process.  Every process only stores amplitudes
        # for its locally assigned data.

        if self.use_noise_prior and self.noise_model is None:
            raise RuntimeError("cannot use noise prior without specifying noise_model")

        # Units for inverse variance weighting
        detnoise_units = 1.0 / self.det_data_units**2

        # Use this as an "Ordered Set".  We want the unique detectors on this process,
        # but sorted in order of occurrence.
        all_dets = OrderedDict()

        # Amplitude lengths of all views for each obs
        self._obs_views = dict()

        # Sample rate for each obs.
        self._obs_rate = dict()

        # Frequency bins for the noise prior for each obs.
        self._freq = dict()

        # Good detectors to use for each observation
        self._obs_dets = dict()

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
                    powmin = np.floor(np.log10(1 / obstime)) - 1
                    powmax = min(
                        np.ceil(np.log10(1 / tbase)) + 2, np.log10(self._obs_rate[iob])
                    )
                    self._freq[iob] = np.logspace(powmin, powmax, 1000)

            # Build up detector list
            self._obs_dets[iob] = set()
            for d in ob.select_local_detectors(flagmask=self.det_mask):
                if d not in ob.detdata[self.det_data].detectors:
                    continue
                self._obs_dets[iob].add(d)
                if d not in all_dets:
                    all_dets[d] = None

        self._all_dets = list(all_dets.keys())

        # Go through the data one local detector at a time and compute the offsets
        # into the amplitudes.

        self._det_start = dict()

        offset = 0
        for det in self._all_dets:
            self._det_start[det] = offset
            for iob, ob in enumerate(new_data.obs):
                if det not in self._obs_dets[iob]:
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
        if self._n_local == 0:
            self._offsetvar_raw = None
            self._offsetvar = None
        else:
            self._offsetvar_raw = AlignedF64.zeros(self._n_local)
            self._offsetvar = self._offsetvar_raw.array()

        offset = 0
        for det in self._all_dets:
            for iob, ob in enumerate(new_data.obs):
                if det not in self._obs_dets[iob]:
                    continue

                # "Noise weight" (time-domain inverse variance)
                detnoise = 1.0
                if self.noise_model is not None:
                    detnoise = (
                        ob[self.noise_model]
                        .detector_weight(det)
                        .to_value(detnoise_units)
                    )

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
                    if detnoise <= 0:
                        # This detector is cut in the noise model
                        for amp in range(n_amp_view):
                            self._offsetvar[offset + amp] = 0.0
                            self._amp_flags[offset + amp] = True
                    else:
                        if self.det_flags is None:
                            voff = 0
                            for amp in range(n_amp_view):
                                amplen = step_length
                                if amp == n_amp_view - 1:
                                    amplen = view_samples - voff
                                self._offsetvar[offset + amp] = 1.0 / (
                                    detnoise * amplen
                                )
                                voff += step_length
                        else:
                            flags = views.detdata[self.det_flags][ivw]
                            voff = 0
                            for amp in range(n_amp_view):
                                amplen = step_length
                                if amp == n_amp_view - 1:
                                    amplen = view_samples - voff
                                n_good = amplen - np.count_nonzero(
                                    flags[det][voff : voff + amplen]
                                    & self.det_flag_mask
                                )
                                if (n_good / amplen) <= self.good_fraction:
                                    # This detector is cut or too many samples flagged
                                    self._offsetvar[offset + amp] = 0.0
                                    self._amp_flags[offset + amp] = True
                                else:
                                    # Keep this
                                    self._offsetvar[offset + amp] = 1.0 / (
                                        detnoise * n_good
                                    )
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
                    if det not in self._obs_dets[iob]:
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

                    if self.debug_plots is not None:
                        set_matplotlib_backend()
                        import matplotlib.pyplot as plt

                        fname = os.path.join(
                            self.debug_plots, f"{self.name}_{det}_{ob.name}_psd.pdf"
                        )
                        psdfreq = ob[self.noise_model].freq(det).to_value(u.Hz)
                        psd = (
                            ob[self.noise_model]
                            .psd(det)
                            .to_value(self.det_data_units**2 * u.second)
                        )
                        corrpsd = self._remove_white_noise(psdfreq, psd)

                        fig = plt.figure(figsize=[12, 12])
                        ax = fig.add_subplot(2, 1, 1)
                        ax.loglog(
                            psdfreq,
                            psd,
                            color="black",
                            label="Original PSD",
                        )
                        ax.loglog(
                            psdfreq,
                            corrpsd,
                            color="red",
                            label="Correlated PSD",
                        )
                        ax.set_xlabel("Frequency [Hz]")
                        ax.set_ylabel("PSD [K$^2$ / Hz]")
                        ax.legend(loc="best")

                        ax = fig.add_subplot(2, 1, 2)
                        ax.loglog(
                            self._freq[iob],
                            offset_psd,
                            label=f"Offset PSD",
                        )
                        ax.set_xlabel("Frequency [Hz]")
                        ax.set_ylabel("PSD [K$^2$ / Hz]")
                        ax.legend(loc="best")
                        fig.savefig(fname)
                        plt.close(fig)

                    # "Noise weight" (time-domain inverse variance)
                    detnoise = (
                        ob[self.noise_model]
                        .detector_weight(det)
                        .to_value(detnoise_units)
                    )

                    # Log version of offset PSD and its inverse for interpolation
                    logfreq = np.log(self._freq[iob])
                    logpsd = np.log(offset_psd)
                    logfilter = np.log(1.0 / offset_psd)

                    # Compute the list of filters and preconditioners (one per view)
                    # For this detector.

                    self._filters[iob][det] = list()
                    self._precond[iob][det] = list()

                    if self.debug_plots is not None:
                        ffilter = os.path.join(
                            self.debug_plots, f"{self.name}_{det}_{ob.name}_filters.pdf"
                        )
                        fprec = os.path.join(
                            self.debug_plots, f"{self.name}_{det}_{ob.name}_prec.pdf"
                        )
                        figfilter = plt.figure(figsize=[12, 8])
                        axfilter = figfilter.add_subplot(1, 1, 1)
                        figprec = plt.figure(figsize=[12, 8])
                        axprec = figprec.add_subplot(1, 1, 1)

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

                        if self.debug_plots is not None:
                            axfilter.plot(
                                np.arange(len(noisefilter)),
                                noisefilter,
                                label=f"Noise filter {ivw}",
                            )

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
                            if detnoise != 0:
                                preconditioner[icenter] += 1.0 / detnoise
                            if self.debug_plots is not None:
                                axprec.plot(
                                    np.arange(len(preconditioner)),
                                    preconditioner,
                                    label=f"Toeplitz preconditioner {ivw}",
                                )
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
                            if detnoise != 0:
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
                            if self.debug_plots is not None:
                                axprec.plot(
                                    np.arange(len(preconditioner)),
                                    preconditioner,
                                    label=f"Banded preconditioner {ivw}",
                                )
                        self._precond[iob][det].append((preconditioner, lower))
                        offset += n_amp_view

                    if self.debug_plots is not None:
                        axfilter.set_xlabel("Sample Lag")
                        axfilter.set_ylabel("Amplitude")
                        axfilter.legend(loc="best")
                        figfilter.savefig(ffilter)
                        axprec.set_xlabel("Sample Lag")
                        axprec.set_ylabel("Amplitude")
                        axprec.legend(loc="best")
                        figprec.savefig(fprec)
                        plt.close(figfilter)
                        plt.close(figprec)

        log.verbose(f"Offset variance = {self._offsetvar}")
        return

    # Helper functions for noise / preconditioner calculations

    def _interpolate_psd(self, x, lfreq, lpsd):
        # Threshold for zero frequency
        thresh = 1.0e-6
        lowf = x < thresh
        good = np.logical_not(lowf)

        logx = np.empty_like(x)
        logx[lowf] = np.log(thresh)
        logx[good] = np.log(x[good])
        logresult = np.interp(logx, lfreq, lpsd)
        result = np.exp(logresult)
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

    def _remove_white_noise(self, freq, psd):
        """Remove the white noise component of the PSD."""
        corrpsd = psd.copy()
        n_corrpsd = len(corrpsd)
        plat_off = int(0.8 * n_corrpsd)
        if n_corrpsd - plat_off < 10:
            if n_corrpsd < 10:
                # Crazy spectrum...
                plat_off = 0
            else:
                plat_off = n_corrpsd - 10

        cfreq = np.log(freq[plat_off:])
        cdata = np.log(corrpsd[plat_off:])

        def lin_func(x, a, b, c):
            # Line
            return a * (x - b) + c

        params, params_cov = scipy.optimize.curve_fit(
            lin_func, cfreq, cdata, p0=[0.0, cfreq[-1], cdata[-1]]
        )

        cdata = lin_func(cfreq, params[0], params[1], params[2])
        cdata = np.exp(cdata)
        plat = cdata[-1]

        # Given the range between the white noise plateau and the maximum
        # values of the PSD, we set a minimum value for any spectral bins
        # that are small or negative.
        corrmax = np.amax(corrpsd)
        corrthresh = 1.0e-10 * corrmax - plat
        corrpsd -= plat
        corrpsd[corrpsd < corrthresh] = corrthresh
        return corrpsd

    def _get_offset_psd(self, noise, freq, step_time, det):
        """Compute the PSD of the baseline offsets."""
        psdfreq = noise.freq(det).to_value(u.Hz)
        psd = noise.psd(det).to_value(self.det_data_units**2 * u.second)
        rate = noise.rate(det).to_value(u.Hz)

        # Remove the white noise component from the PSD
        psd = self._remove_white_noise(psdfreq, psd)

        # Log PSD for interpolation
        logfreq = np.log(psdfreq)
        logpsd = np.log(psd)

        # The calculation of `offset_psd` is based on Keihänen, E. et al:
        # "Making CMB temperature and polarization maps with Madam",
        # A&A 510:A57, 2010, with a small algebra correction.

        m_max = 5
        tbase = step_time
        fbase = 1.0 / tbase

        def g(f, m):
            # The frequencies are constructed without the zero frequency,
            # so we do not need to handle it here.
            # result = np.sin(np.pi * f * tbase) ** 2 / (np.pi * (f * tbase + m)) ** 2
            x = np.pi * (f * tbase + m)
            bad = np.abs(x) < 1.0e-30
            good = np.logical_not(bad)
            result = np.empty_like(x)
            result[bad] = 1.0
            result[good] = np.sin(x[good]) ** 2 / x[good] ** 2
            return result

        offset_psd = np.zeros_like(freq)

        # The m = 0 term
        offset_psd = self._interpolate_psd(freq, logfreq, logpsd) * g(freq, 0)

        # The remaining terms
        for m in range(1, m_max):
            # Positive m
            offset_psd[:] += self._interpolate_psd(
                freq + m * fbase, logfreq, logpsd
            ) * g(freq, m)
            # Negative m
            offset_psd[:] += self._interpolate_psd(
                freq - m * fbase, logfreq, logpsd
            ) * g(freq, -m)

        offset_psd *= fbase
        return offset_psd

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(self.data.comm, self._n_global, self._n_local)
        if z.local_flags is not None:
            z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _step_length(self, stime, rate):
        return int(stime * rate + 0.5)

    @function_timer
    def _add_to_signal(self, detector, amplitudes, use_accel=None, **kwargs):
        log = Logger.get()

        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        amp_offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in self._obs_dets[iob]:
                continue
            det_indx = ob.detdata[self.det_data].indices([detector])
            # The step length for this observation
            step_length = self._step_length(
                self.step_time.to_value(u.second), self._obs_rate[iob]
            )

            # The number of amplitudes in each view
            n_amp_views = self._obs_views[iob]

            # # DEBUGGING
            # restore_dev = False
            # prefix = "HOST"
            # if amplitudes.accel_in_use():
            #     amplitudes.accel_update_host()
            #     restore_dev = True
            #     prefix = "DEVICE"
            # print(
            #     f"{prefix} Add to signal input:  {amp_offset}, {n_amp_views}, {amplitudes.local}",
            #     flush=True,
            # )
            # if restore_dev:
            #     amplitudes.accel_update_device()

            # # DEBUGGING
            # restore_dev = False
            # prefix = "HOST"
            # if ob.detdata[self.det_data].accel_in_use():
            #     ob.detdata[self.det_data].accel_update_host()
            #     restore_dev = True
            #     prefix = "DEVICE"
            # tod_min = np.amin(ob.detdata[self.det_data])
            # tod_max = np.amax(ob.detdata[self.det_data])
            # print(
            #     f"{prefix} Add to signal starting TOD output:  {ob.detdata[self.det_data]}, min={tod_min}, max={tod_max}",
            #     flush=True,
            # )
            # if (np.absolute(tod_min) < 1.0e-15) and (np.absolute(tod_max) < 1.0e-15):
            #     ob.detdata[self.det_data][:] = 0
            # if restore_dev:
            #     ob.detdata[self.det_data].accel_update_device()

            offset_add_to_signal(
                step_length,
                amp_offset,
                n_amp_views,
                amplitudes.local,
                amplitudes.local_flags,
                det_indx[0],
                ob.detdata[self.det_data].data,
                ob.intervals[self.view].data,
                impl=implementation,
                use_accel=use_accel,
            )

            # # DEBUGGING
            # restore_dev = False
            # prefix = "HOST"
            # if ob.detdata[self.det_data].accel_in_use():
            #     ob.detdata[self.det_data].accel_update_host()
            #     restore_dev = True
            #     prefix = "DEVICE"
            # print(
            #     f"{prefix} Add to signal output:  {ob.detdata[self.det_data]}",
            #     flush=True,
            # )
            # if restore_dev:
            #     ob.detdata[self.det_data].accel_update_device()

            amp_offset += np.sum(n_amp_views)

    @function_timer
    def _project_signal(self, detector, amplitudes, use_accel=None, **kwargs):
        log = Logger.get()

        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        amp_offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in self._obs_dets[iob]:
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

            # # DEBUGGING
            # restore_dev = False
            # prefix="HOST"
            # if ob.detdata[self.det_data].accel_in_use():
            #     ob.detdata[self.det_data].accel_update_host()
            #     restore_dev = True
            #     prefix="DEVICE"
            # print(f"{prefix} Project signal input:  {ob.detdata[self.det_data]}", flush=True)
            # if restore_dev:
            #     ob.detdata[self.det_data].accel_update_device()

            offset_project_signal(
                det_indx[0],
                ob.detdata[self.det_data].data,
                flag_indx[0],
                flag_data,
                self.det_flag_mask,
                step_length,
                amp_offset,
                n_amp_views,
                amplitudes.local,
                amplitudes.local_flags,
                ob.intervals[self.view].data,
                impl=implementation,
                use_accel=use_accel,
            )

            # restore_dev = False
            # prefix="HOST"
            # if amplitudes.accel_in_use():
            #     amplitudes.accel_update_host()
            #     restore_dev = True
            #     prefix="DEVICE"
            # print(f"{prefix} Project signal output:  {amp_offset}, {n_amp_views}, {amplitudes.local}", flush=True)
            # if restore_dev:
            #     amplitudes.accel_update_device()

            amp_offset += np.sum(n_amp_views)

    @function_timer
    def _add_prior(self, amplitudes_in, amplitudes_out, use_accel=None, **kwargs):
        if not self.use_noise_prior:
            # Not using the noise prior term, nothing to accumulate to output.
            return
        if use_accel:
            raise NotImplementedError(
                "offset template add_prior on accelerator not implemented"
            )
        if self.debug_plots is not None:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

        for det in self._all_dets:
            offset = self._det_start[det]
            for iob, ob in enumerate(self.data.obs):
                if det not in self._obs_dets[iob]:
                    continue
                for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                    n_amp_view = self._obs_views[iob][ivw]
                    amp_slice = slice(offset, offset + n_amp_view, 1)
                    amps_in = amplitudes_in.local[amp_slice]
                    amp_flags_in = amplitudes_in.local_flags[amp_slice]
                    amps_out = amplitudes_out.local[amp_slice]
                    if det in self._filters[iob]:
                        # There is some contribution from this detector
                        amps_out[:] += scipy.signal.convolve(
                            amps_in,
                            self._filters[iob][det][ivw],
                            mode="same",
                            method="direct",
                        )

                        if self.debug_plots is not None:
                            # Find the first unused file name in the sequence
                            iter = -1
                            while iter < 0 or os.path.isfile(fname):
                                iter += 1
                                fname = os.path.join(
                                    self.debug_plots,
                                    f"{self.name}_{det}_{ob.name}_prior_{ivw}_{iter}.pdf",
                                )
                            fig = plt.figure(figsize=[12, 8])
                            ax = fig.add_subplot(1, 1, 1)
                            ax.plot(
                                np.arange(len(amps_in)),
                                amps_in,
                                color="black",
                                label="Input Amplitudes",
                            )
                            ax.plot(
                                np.arange(len(amps_in)),
                                amps_out,
                                color="red",
                                label="Output Amplitudes",
                            )

                        amps_out[amp_flags_in != 0] = 0.0

                        if self.debug_plots is not None:
                            ax.plot(
                                np.arange(len(amps_in)),
                                amps_out,
                                color="green",
                                label="Output Amplitudes (flagged)",
                            )
                            ax.set_xlabel("Amplitude Index")
                            ax.set_ylabel("Value")
                            ax.legend(loc="best")
                            fig.savefig(fname)
                            plt.close(fig)

                    else:
                        amps_out[:] = 0.0
                    offset += n_amp_view

    @function_timer
    def _apply_precond(self, amplitudes_in, amplitudes_out, use_accel=None, **kwargs):
        if self.use_noise_prior:
            if use_accel:
                raise NotImplementedError(
                    "offset template precond on accelerator not implemented"
                )
            # Our design matrix includes a term with the inverse offset covariance.
            # This means that our preconditioner should include this term as well.
            for det in self._all_dets:
                offset = self._det_start[det]
                for iob, ob in enumerate(self.data.obs):
                    if det not in self._obs_dets[iob]:
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
                        amp_flags_in = amplitudes_in.local_flags[amp_slice]
                        amps_out = None
                        if det in self._precond[iob]:
                            # We have a contribution from this detector
                            if self.precond_width <= 1:
                                # We are using a Toeplitz preconditioner.
                                # scipy.signal.convolve will use either `convolve` or
                                # `fftconvolve` depending on the size of the inputs
                                amps_out = scipy.signal.convolve(
                                    amps_in,
                                    self._precond[iob][det][ivw][0],
                                    mode="same",
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
                            amps_out[amp_flags_in != 0] = 0.0
                        else:
                            # This detector is cut
                            amps_out = np.zeros_like(amps_in)
                        amplitudes_out.local[amp_slice] = amps_out
                        offset += n_amp_view
        else:
            # Since we do not have a noise filter term in our LHS, our diagonal
            # preconditioner is just the application of offset variance.

            # Kernel selection
            implementation, use_accel = self.select_kernels(use_accel=use_accel)

            offset_apply_diag_precond(
                self._offsetvar,
                amplitudes_in.local,
                amplitudes_in.local_flags,
                amplitudes_out.local,
                impl=implementation,
                use_accel=use_accel,
            )
        return

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        return True

    @function_timer
    def write(self, amplitudes, out):
        """Write out amplitude values.

        This stores the amplitudes to files for debugging / plotting.  Since the
        Offset amplitudes are unique on each process, we open one file per process
        group and each process in the group communicates their amplitudes to one
        writer.

        Since this function is used mainly for debugging, we are a bit wasteful
        and duplicate the amplitudes in order to make things easier.

        Args:
            amplitudes (Amplitudes):  The amplitude data.
            out (str):  The output file root.

        Returns:
            None

        """

        # Copy of the amplitudes, organized by observation and detector
        obs_det_amps = dict()

        for det in self._all_dets:
            amp_offset = self._det_start[det]
            for iob, ob in enumerate(self.data.obs):
                if det not in self._obs_dets[iob]:
                    continue
                if not ob.is_distributed_by_detector:
                    raise NotImplementedError(
                        "Only observations distributed by detector are supported"
                    )
                # The step length for this observation
                step_length = self._step_length(
                    self.step_time.to_value(u.second), self._obs_rate[iob]
                )

                if ob.name not in obs_det_amps:
                    # First time with this observation, store info about
                    # the offset spans
                    obs_det_amps[ob.name] = dict()
                    props = dict()
                    props["step_length"] = step_length
                    amp_first = list()
                    amp_last = list()
                    amp_start = list()
                    amp_stop = list()
                    for ivw, vw in enumerate(ob.intervals[self.view]):
                        n_amp_view = self._obs_views[iob][ivw]
                        for istep in range(n_amp_view):
                            istart = vw.first + istep * step_length
                            amp_first.append(istart)
                            amp_start.append(ob.shared[self.times].data[istart])
                            if istep == n_amp_view - 1:
                                istop = vw.last
                            else:
                                istop = vw.first + (istep + 1) * step_length
                            amp_last.append(istop)
                            amp_stop.append(ob.shared[self.times].data[istop - 1])
                    props["amp_first"] = np.array(amp_first, dtype=np.int64)
                    props["amp_last"] = np.array(amp_last, dtype=np.int64)
                    props["amp_start"] = np.array(amp_start, dtype=np.float64)
                    props["amp_stop"] = np.array(amp_stop, dtype=np.float64)
                    obs_det_amps[ob.name]["bounds"] = props

                # Loop over views and extract per-detector amplitudes and flags
                det_amps = list()
                det_flags = list()
                views = ob.view[self.view]
                for ivw, vw in enumerate(views):
                    n_amp_view = self._obs_views[iob][ivw]
                    amp_slice = slice(amp_offset, amp_offset + n_amp_view, 1)
                    det_amps.append(amplitudes.local[amp_slice])
                    det_flags.append(amplitudes.local_flags[amp_slice])
                    amp_offset += n_amp_view
                det_amps = np.concatenate(det_amps, dtype=np.float64)
                det_flags = np.concatenate(det_flags, dtype=np.uint8)
                obs_det_amps[ob.name][det] = {
                    "amps": det_amps,
                    "flags": det_flags,
                }

        # Each group writes out its amplitudes.

        # NOTE:  If/when we want to support arbitrary data distributions when
        # writing, we would need to take the data from each process and align
        # them in time rather than just extracting detector data and writing
        # to the datasets.

        for iob, ob in enumerate(self.data.obs):
            obs_local_amps = obs_det_amps[ob.name]
            if self.data.comm.group_size == 1:
                all_obs_amps = [obs_local_amps]
            else:
                all_obs_amps = self.data.comm.comm_group.gather(obs_local_amps, root=0)

            if self.data.comm.group_rank == 0:
                out_file = f"{out}_{ob.name}.h5"
                det_names = set()
                for pdata in all_obs_amps:
                    for k in pdata.keys():
                        if k != "bounds":
                            det_names.add(k)
                det_names = list(sorted(det_names))
                n_det = len(det_names)
                amp_first = all_obs_amps[0]["bounds"]["amp_first"]
                amp_last = all_obs_amps[0]["bounds"]["amp_last"]
                amp_start = all_obs_amps[0]["bounds"]["amp_start"]
                amp_stop = all_obs_amps[0]["bounds"]["amp_stop"]
                n_amp = len(amp_first)
                det_to_row = {y: x for x, y in enumerate(det_names)}
                with h5py.File(out_file, "w") as hf:
                    hf.attrs["step_length"] = all_obs_amps[0]["bounds"]["step_length"]
                    hf.attrs["detectors"] = json.dumps(det_names)
                    hamp_first = hf.create_dataset("amp_first", data=amp_first)
                    hamp_last = hf.create_dataset("amp_last", data=amp_last)
                    hamp_start = hf.create_dataset("amp_start", data=amp_start)
                    hamp_stop = hf.create_dataset("amp_stop", data=amp_stop)
                    hamps = hf.create_dataset(
                        "amplitudes",
                        (n_det, n_amp),
                        dtype=np.float64,
                    )
                    hflags = hf.create_dataset(
                        "flags",
                        (n_det, n_amp),
                        dtype=np.uint8,
                    )
                    for pdata in all_obs_amps:
                        for k, v in pdata.items():
                            if k == "bounds":
                                continue
                            row = det_to_row[k]
                            hslice = (slice(row, row + 1, 1), slice(0, n_amp, 1))
                            dslice = (slice(0, n_amp, 1),)
                            hamps.write_direct(v["amps"], dslice, hslice)
                            hflags.write_direct(v["flags"], dslice, hslice)


def plot(amp_file, compare=dict(), out=None, xlim=None):
    """Plot an amplitude dump file.

    This loads an amplitude file and makes a set of plots.

    Args:
        amp_file (str):  The path to the input file of observation amplitudes.
        compare (dict):  If specified, dictionary of per-detector timestreams
            to plot for comparison.
        out (str):  The output file.
        xlim (tuple):  The X axis sample range to plot.

    Returns:
        None

    """

    if out is not None:
        set_matplotlib_backend(backend="pdf")

    import matplotlib.pyplot as plt

    figdpi = 100

    with h5py.File(amp_file, "r") as hf:
        step_length = hf.attrs["step_length"]
        det_list = json.loads(hf.attrs["detectors"])
        n_det = len(det_list)
        amp_first = hf["amp_first"]
        amp_last = hf["amp_last"]
        amp_start = hf["amp_start"]
        amp_stop = hf["amp_stop"]
        hamps = hf["amplitudes"]
        hflags = hf["flags"]
        n_amp = len(amp_first)

        fig_width = 8
        fig_height = 4
        fig_dpi = 100

        x_samples = np.arange(amp_first[0], amp_last[-1], 1)

        for idet, det in enumerate(det_list):
            outfile = f"{out}_{det}.pdf"
            fig = plt.figure(dpi=fig_dpi, figsize=(fig_width, fig_height))
            ax = fig.add_subplot(1, 1, 1)
            if det in compare:
                dc = np.mean(compare[det])
                ax.plot(
                    x_samples, compare[det] - dc, color="black", label=f"{det} Data"
                )
            ax.step(
                amp_first[:],
                hamps[idet],
                where="post",
                color="red",
                label=f"{det} Offset Amplitudes",
            )
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="best")
            if out is None:
                # Interactive
                plt.show()
            else:
                plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
                plt.close()
