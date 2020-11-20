from collections import OrderedDict
import os
import sys

import numpy as np
import scipy.linalg
import scipy.signal

from toast import Operator
from toast.mpi import MPI

from ..timing import gather_timers, GlobalTimers
from toast.timing import function_timer, Timer
from toast.utils import Logger, Environment
from .sim_det_map import OpSimScan
from .todmap_math import OpAccumDiag, OpScanScale, OpScanMask
from ..tod import OpCacheClear, OpCacheCopy, OpCacheInit, OpFlagsApply, OpFlagGaps
from ..map import covariance_apply, covariance_invert, DistPixels, covariance_rcond
from .. import qarray as qa

from .._libtoast import add_offsets_to_signal, project_signal_offsets


XAXIS, YAXIS, ZAXIS = np.eye(3)

temporary_names = set()


def get_temporary_name():
    i = 0
    while True:
        name = "temporary{:03}".format(i)
        if name not in temporary_names:
            break
        i += 1
    temporary_names.add(name)
    return name


def free_temporary_name(name):
    temporary_names.remove(name)


class TOASTMatrix:
    def apply(self, vector, inplace=False):
        """Every TOASTMatrix can apply itself to a distributed vectors
        of signal, map or template offsets as is appropriate.
        """
        raise NotImplementedError("Virtual apply not implemented in derived class")

    def apply_transpose(self, vector, inplace=False):
        """Every TOASTMatrix can apply itself to a distributed vectors
        of signal, map or template offsets as is appropriate.
        """
        raise NotImplementedError(
            "Virtual apply_transpose not implemented in derived class"
        )


class TOASTVector:
    def dot(self, other):
        raise NotImplementedError("Virtual dot not implemented in derived class")


class UnitMatrix(TOASTMatrix):
    def apply(self, vector, inplace=False):
        if inplace:
            outvec = vector
        else:
            outvec = vector.copy()
        return outvec


class TODTemplate:
    """Parent class for all templates that can be registered with
    TemplateMatrix
    """

    name = None
    namplitude = 0
    comm = None

    def __init___(self, *args, **kwargs):
        raise NotImplementedError("Derived class must implement __init__()")

    def add_to_signal(self, signal, amplitudes):
        """signal += F.a"""
        raise NotImplementedError("Derived class must implement add_to_signal()")

    def project_signal(self, signal, amplitudes):
        """a += F^T.signal"""
        raise NotImplementedError("Derived class must implement project_signal()")

    def add_prior(self, amplitudes_in, amplitudes_out):
        """a' += C_a^{-1}.a"""
        # Not all TODTemplates implement the prior
        return

    def apply_precond(self, amplitudes_in, amplitudes_out):
        """a' = M^{-1}.a"""
        raise NotImplementedError("Derived class must implement apply_precond()")

    def calibrate(self, signal, amplitudes ):
        """ Empty method for derived classes """
        # Not all TODTemplates implement  this method

        return
class SubharmonicTemplate(TODTemplate):
    """This class represents sub-harmonic noise fluctuations.

    Sub-harmonic means that the characteristic frequency of the noise
    modes is lower than 1/T where T is the length of the interval
    being fitted.
    """

    name = "subharmonic"

    def __init__(
        self,
        data,
        detweights,
        order=1,
        intervals=None,
        common_flags=None,
        common_flag_mask=1,
        flags=None,
        flag_mask=1,
    ):
        self.data = data
        self.detweights = detweights
        self.order = order
        self.intervals = intervals
        self.common_flags = common_flags
        self.common_flag_mask = common_flag_mask
        self.flags = flags
        self.flag_mask = flag_mask
        self._last_nsamp = None
        self._last_templates = None
        self.get_steps_and_preconditioner()

    def get_steps_and_preconditioner(self):
        """Assign each template an amplitude"""
        self.templates = []
        self.slices = []
        self.preconditioners = []
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            common_flags = tod.local_common_flags(self.common_flags)
            common_flags = (common_flags & self.common_flag_mask) != 0
            if (self.intervals is not None) and (self.intervals in obs):
                intervals = obs[self.intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)
            slices = {}  # this observation
            preconditioners = {}  # this observation
            for ival in local_intervals:
                todslice = slice(ival.first, ival.last + 1)
                for idet, det in enumerate(tod.local_dets):
                    ind = slice(self.namplitude, self.namplitude + self.order + 1)
                    self.templates.append([ind, iobs, det, todslice])
                    self.namplitude += self.order + 1
                    preconditioner = self._get_preconditioner(
                        det, tod, todslice, common_flags, self.detweights[iobs][det]
                    )
                    if det not in preconditioners:
                        preconditioners[det] = []
                        slices[det] = []
                    preconditioners[det].append(preconditioner)
                    slices[det].append(ind)
            self.slices.append(slices)
            self.preconditioners.append(preconditioners)
        return

    def _get_preconditioner(self, det, tod, todslice, common_flags, detweight):
        """Calculate the preconditioner for the given interval and detector"""
        flags = tod.local_flags(det, self.flags)[todslice]
        good = (flags & self.flag_mask) == 0
        good[common_flags[todslice]] = False
        norder = self.order + 1
        preconditioner = np.zeros([norder, norder])
        templates = self._get_templates(todslice)
        for row in range(norder):
            for col in range(row, norder):
                preconditioner[row, col] = np.dot(
                    templates[row][good], templates[col][good]
                )
                preconditioner[row, col] *= detweight
                if row != col:
                    preconditioner[col, row] = preconditioner[row, col]
        preconditioner = np.linalg.inv(preconditioner)
        return preconditioner

    def add_to_signal(self, signal, amplitudes):
        subharmonic_amplitudes = amplitudes[self.name]
        for ibase, (ind, iobs, det, todslice) in enumerate(self.templates):
            templates = self._get_templates(todslice)
            amps = subharmonic_amplitudes[ind]
            for template, amplitude in zip(templates, amps):
                signal[iobs, det, todslice] += template * amplitude
        return

    def _get_templates(self, todslice):
        """Develop hierarchy of subharmonic modes matching the given length

        The basis functions are (orthogonal) Legendre polynomials
        """
        nsamp = todslice.stop - todslice.start
        if nsamp != self._last_nsamp:
            templates = np.zeros([self.order + 1, nsamp])
            r = np.linspace(-1, 1, nsamp)
            for order in range(self.order + 1):
                if order == 0:
                    templates[order] = 1
                elif order == 1:
                    templates[order] = r
                else:
                    templates[order] = (
                        (2 * order - 1) * r * templates[order - 1]
                        - (order - 1) * templates[order - 2]
                    ) / order
            self._last_nsamp = nsamp
            self._last_templates = templates
        return self._last_templates

    def project_signal(self, signal, amplitudes):
        subharmonic_amplitudes = amplitudes[self.name]
        for ibase, (ind, iobs, det, todslice) in enumerate(self.templates):
            templates = self._get_templates(todslice)
            amps = subharmonic_amplitudes[ind]
            for order, template in enumerate(templates):
                amps[order] = np.dot(signal[iobs, det, todslice], template)
        pass

    def apply_precond(self, amplitudes_in, amplitudes_out):
        """Standard diagonal preconditioner accounting for the fact that
        the templates are not orthogonal in the presence of flagging and masking
        """
        subharmonic_amplitudes_in = amplitudes_in[self.name]
        subharmonic_amplitudes_out = amplitudes_out[self.name]
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            for det in tod.local_dets:
                slices = self.slices[iobs][det]
                preconditioners = self.preconditioners[iobs][det]
                for ind, preconditioner in zip(slices, preconditioners):
                    subharmonic_amplitudes_out[ind] = np.dot(
                        preconditioner, subharmonic_amplitudes_in[ind]
                    )
        return


class GainTemplate(TODTemplate):
    """This class aims at estimating the amplitudes of gain fluctuations
    by means of templates within the TOAST mapmaker.  """

    name = "Gain"

    def __init__(
        self,
        data,
        detweights,
        order=1,
        common_flags=None,
        common_flag_mask=1,
        flags=None,
        flag_mask=1,
        templatename =None,

    ):
        self.data = data
        self.comm = data.comm.comm_group
        self.order = order
        self.norder = order + 1
        self.detweights= detweights
        self.common_flags = common_flags
        self.common_flag_mask = common_flag_mask
        self.flags = flags
        self.flag_mask = flag_mask
        self.template_name = templatename
        self._estimate_offsets()
        self.namplitude =  self.norder + self.list_of_offsets[-1][-1]
        self._estimate_preconditioner()
        return

    def _estimate_offsets (self ):
        """ Precompute the amplitude offsets """

        offset =0
        self.list_of_offsets =[]
        for  obs in  (self.data.obs):
            tod = obs["tod"]
            tmplist =[]
            for det in tod.local_dets :
                tmplist .append(offset)
                offset +=self.norder
            self.list_of_offsets.append(tmplist )


    def _get_polynomials (self, N, local_offset, local_N):
        x = 2 * np.arange(N) / (N - 1)   -1
        todslice = slice(local_offset, local_offset + local_N)
        L = np.zeros((local_N, self.norder))
        for i in range(self. norder):
             L[:,i]=  scipy.special.legendre(i)(x )
        return  L

    @function_timer
    def _estimate_preconditioner (self):
        self.preconditioners =[]
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            nsample = tod.total_samples
            # For each observation, sample indices start from 0
            local_offset, local_nsample = tod.local_samples
            L = self._get_polynomials(nsample, local_offset, local_nsample)
            LT =L.T.copy()

            tmplist =[]
            for idet, det in enumerate( tod.local_dets):
                detweight = self.detweights[iobs][det]
                T = tod.local_signal(det, self.template_name)

                for row in LT: row *= (T*np.sqrt(detweight))
                M = LT.dot(LT.T)
                #try:
                tmplist.append(np.linalg.inv(M))
                #except LinAlgError :
                ## TODO: what if linalg.inv fails??

            self.preconditioners.append(tmplist)


    @function_timer
    def add_to_signal(self, signal,  amplitudes):
        """signal += F.a"""
        poly_amplitudes = amplitudes[self.name]
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            nsample = tod.total_samples
            # For each observation, sample indices start from 0
            local_offset, local_nsample = tod.local_samples
            legendre_poly = self._get_polynomials(nsample, local_offset, local_nsample)
            todslice = slice(local_offset, local_offset + local_nsample)
            for idet, det in enumerate(tod.local_dets):
                ind = self.list_of_offsets[iobs ][idet ]
                amplitude_slice= slice(ind ,ind+self.norder )
                poly_amps = poly_amplitudes[amplitude_slice ]
                delta_gain = legendre_poly.dot(poly_amps)
                signal_estimate = tod.local_signal(det, self.template_name)
                gain_fluctuation = signal_estimate * delta_gain
                signal[iobs, det, todslice] += gain_fluctuation
        return


    @function_timer
    def project_signal(self, signal, amplitudes):
        """a += F^T.signal"""

        poly_amplitudes = amplitudes[self.name]
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            nsample = tod.total_samples
            # For each observation, sample indices start from 0
            local_offset, local_nsample = tod.local_samples
            legendre_poly = self._get_polynomials(nsample, local_offset, local_nsample)
            todslice = slice(local_offset, local_offset + local_nsample)
            LT= legendre_poly.T.copy()
            for idet, det in enumerate( tod.local_dets):
                ind = self.list_of_offsets[iobs ][idet ]
                amplitude_slice= slice(ind ,ind+self.norder )
                signal_estimate = tod.local_signal(det, self.template_name)
                for row in LT : row *= (signal_estimate )
                poly_amplitudes[amplitude_slice] += np.dot(LT, signal[iobs, det, todslice] )
        return

    def write_gain_fluctuation(self, amplitudes, filename):
        ### # WARNING: need to communicate the amplitudes across
        ### the processors to save them into disc ,
        # the way this is implemented so far  if nprocs>1  every processor writes to disc
        # the amplitudes into the very same file...
        np.savez( filename, amplitudes=amplitudes[self.name])
        #raise RuntimeError("Saving gain fluctuation not implemented")
        return

    def apply_precond(self, amplitudes_in, amplitudes_out):
        """a' = M^{-1}.b """
        b = amplitudes_in[self.name]
        a = amplitudes_out[self.name]
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]

            for idet, det in enumerate( tod.local_dets):
                ind = self.list_of_offsets[iobs ][idet ]
                amplitude_slice= slice(ind ,ind+self.norder )
                M = self.preconditioners[iobs ][idet ]
                a[amplitude_slice] = M.dot(b[amplitude_slice])
        return

    def calibrate(self, signal, amplitudes ):
        """
        Estimate the optimal gain fluctuations from the amplitudes estimated
        at   PCG convergence .
        """
        poly_amplitudes = amplitudes[self.name]
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            nsample = tod.total_samples
            # For each observation, sample indices start from 0
            local_offset, local_nsample = tod.local_samples
            legendre_poly = self._get_polynomials(nsample, local_offset, local_nsample)
            todslice = slice(local_offset, local_offset + local_nsample)
            for idet, det in enumerate(tod.local_dets):
                ind = self.list_of_offsets[iobs ][idet ]
                amplitude_slice= slice(ind ,ind+self.norder )
                poly_amps = poly_amplitudes[amplitude_slice ]
                delta_gain = legendre_poly.dot(poly_amps)
        return delta_gain



class Fourier2DTemplate(TODTemplate):
    """This class represents atmospheric fluctuations in front of the
    focalplane as 2D Fourier modes."""

    name = "Fourier2D"

    def __init__(
        self,
        data,
        detweights,
        focalplane_radius=None,  # degrees
        order=1,
        fit_subharmonics=True,
        intervals=None,
        common_flags=None,
        common_flag_mask=1,
        flags=None,
        flag_mask=1,
        correlation_length=10,
        correlation_amplitude=10,
    ):
        self.data = data
        self.comm = data.comm.comm_group
        self.detweights = detweights
        self.focalplane_radius = focalplane_radius
        self.order = order
        self.fit_subharmonics = fit_subharmonics
        self.norder = order + 1
        self.nmode = (2 * order) ** 2 + 1
        if self.fit_subharmonics:
            self.nmode += 2
        self.intervals = intervals
        self.common_flags = common_flags
        self.common_flag_mask = common_flag_mask
        self.flags = flags
        self.flag_mask = flag_mask
        self._get_templates()
        self.correlation_length = correlation_length
        self.correlation_amplitude = correlation_amplitude
        if correlation_length:
            self._get_prior()
        return

    @function_timer
    def _get_prior(self):
        """Evaluate C_a^{-1} for the 2D polynomial coefficients based
        on the correlation length.
        """
        if self.correlation_length:
            # Correlation length is given in seconds and we cannot assume
            # that each observation has the same sampling rate.  Therefore,
            # we will build the filter for each observation
            self.filters = []  # all observations
            self.preconditioners = []  # all observations
            for iobs, obs in enumerate(self.data.obs):
                tod = obs["tod"]
                times = tod.local_times()
                corr = (
                    np.exp((times[0] - times) / self.correlation_length)
                    * self.correlation_amplitude
                )
                ihalf = times.size // 2
                corr[ihalf + 1 :] = corr[ihalf - 1 : 0 : -1]
                fcorr = np.fft.rfft(corr)
                invcorr = np.fft.irfft(1 / fcorr)
                self.filters.append(invcorr)
            # Scale the filter by the prescribed correlation strength
            # and the number of modes at each angular scale
            self.filter_scale = np.zeros(self.nmode)
            self.filter_scale[0] = 1
            offset = 1
            if self.fit_subharmonics:
                self.filter_scale[1:3] = 2
                offset += 2
            self.filter_scale[offset:] = 4
            self.filter_scale *= self.correlation_amplitude
        return

    @function_timer
    def _get_templates(self):
        """Evaluate and normalize the polynomial templates.

        Each template corresponds to a fixed value for each detector
        and depends on the position of the detector.
        """
        self.templates = []

        def evaluate_template(theta, phi, radius):
            values = np.zeros(self.nmode)
            values[0] = 1
            offset = 1
            if self.fit_subharmonics:
                values[1:3] = theta / radius, phi / radius
                offset += 2
            if self.order > 0:
                rinv = np.pi / radius
                orders = np.arange(self.order) + 1
                thetavec = np.zeros(self.order * 2)
                phivec = np.zeros(self.order * 2)
                thetavec[::2] = np.cos(orders * theta * rinv)
                thetavec[1::2] = np.sin(orders * theta * rinv)
                phivec[::2] = np.cos(orders * phi * rinv)
                phivec[1::2] = np.sin(orders * phi * rinv)
                values[offset:] = np.outer(thetavec, phivec).ravel()
            return values

        self.norms = []
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            common_flags = tod.local_common_flags(self.common_flags)
            common_flags = (common_flags & self.common_flag_mask) != 0
            nsample = tod.total_samples
            obs_templates = {}
            focalplane = obs["focalplane"]
            if self.focalplane_radius:
                radius = np.radians(self.focalplane_radius)
            else:
                try:
                    radius = np.radians(focalplane.radius)
                except AttributeError:
                    # Focalplane is just a dictionary
                    radius = np.radians(obs["fpradius"])
            norms = np.zeros([nsample, self.nmode])
            local_offset, local_nsample = tod.local_samples
            todslice = slice(local_offset, local_offset + local_nsample)
            for det in tod.local_dets:
                flags = tod.local_flags(det, self.flags)
                good = ((flags & self.flag_mask) | common_flags) == 0
                detweight = self.detweights[iobs][det]
                det_quat = focalplane[det]["quat"]
                x, y, z = qa.rotate(det_quat, ZAXIS)
                theta, phi = np.arcsin([x, y])
                obs_templates[det] = evaluate_template(theta, phi, radius)
                norms[todslice] += np.outer(good, obs_templates[det] ** 2 * detweight)
            self.comm.allreduce(norms)
            good = norms != 0
            norms[good] = 1 / norms[good]
            self.norms.append(norms.ravel())
            self.templates.append(obs_templates)
            self.namplitude += nsample * self.nmode

        self.norms = np.hstack(self.norms)

        return

    @function_timer
    def add_to_signal(self, signal, amplitudes):
        """signal += F.a"""
        poly_amplitudes = amplitudes[self.name]
        amplitude_offset = 0
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            nsample = tod.total_samples
            # For each observation, sample indices start from 0
            local_offset, local_nsample = tod.local_samples
            todslice = slice(local_offset, local_offset + local_nsample)
            obs_amplitudes = poly_amplitudes[
                amplitude_offset : amplitude_offset + nsample * self.nmode
            ].reshape([nsample, self.nmode])[todslice]
            for det in tod.local_dets:
                templates = self.templates[iobs][det]
                signal[iobs, det, todslice] += np.sum(obs_amplitudes * templates, 1)
            amplitude_offset += nsample * self.nmode
        return

    @function_timer
    def project_signal(self, signal, amplitudes):
        """a += F^T.signal"""
        poly_amplitudes = amplitudes[self.name]
        amplitude_offset = 0
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            nsample = tod.total_samples
            # For each observation, sample indices start from 0
            local_offset, local_nsample = tod.local_samples
            todslice = slice(local_offset, local_offset + local_nsample)
            obs_amplitudes = poly_amplitudes[
                amplitude_offset : amplitude_offset + nsample * self.nmode
            ].reshape([nsample, self.nmode])
            if self.comm is not None:
                my_amplitudes = np.zeros_like(obs_amplitudes)
            else:
                my_amplitudes = obs_amplitudes
            for det in tod.local_dets:
                templates = self.templates[iobs][det]
                my_amplitudes[todslice] += np.outer(
                    signal[iobs, det, todslice], templates
                )
            if self.comm is not None:
                self.comm.allreduce(my_amplitudes)
                obs_amplitudes += my_amplitudes
            amplitude_offset += nsample * self.nmode
        return

    def add_prior(self, amplitudes_in, amplitudes_out):
        """a' += C_a^{-1}.a"""
        if self.correlation_length:
            poly_amplitudes_in = amplitudes_in[self.name]
            poly_amplitudes_out = amplitudes_out[self.name]
            amplitude_offset = 0
            for obs, noisefilter in zip(self.data.obs, self.filters):
                tod = obs["tod"]
                nsample = tod.total_samples
                obs_amplitudes_in = poly_amplitudes_in[
                    amplitude_offset : amplitude_offset + nsample * self.nmode
                ].reshape([nsample, self.nmode])
                obs_amplitudes_out = poly_amplitudes_out[
                    amplitude_offset : amplitude_offset + nsample * self.nmode
                ].reshape([nsample, self.nmode])
                # import pdb
                # import matplotlib.pyplot as plt
                # pdb.set_trace()
                for mode in range(self.nmode):
                    scale = self.filter_scale[mode]
                    obs_amplitudes_out[:, mode] += scipy.signal.convolve(
                        obs_amplitudes_in[:, mode],
                        noisefilter * scale,
                        mode="same",
                    )
                amplitude_offset += nsample * self.nmode
        return

    def apply_precond(self, amplitudes_in, amplitudes_out):
        """a' = M^{-1}.a"""
        poly_amplitudes_in = amplitudes_in[self.name]
        poly_amplitudes_out = amplitudes_out[self.name]
        poly_amplitudes_out[:] = poly_amplitudes_in * self.norms
        return


class OffsetTemplate(TODTemplate):
    """This class represents noise fluctuations as a step function"""

    name = "offset"

    def __init__(
        self,
        data,
        detweights,
        step_length=1000000,
        intervals=None,
        use_noise_prior=True,
        common_flags=None,
        common_flag_mask=1,
        flags=None,
        flag_mask=1,
        precond_width=20,
    ):
        self.data = data
        self.detweights = detweights
        self.step_length = step_length
        self.intervals = intervals
        self.common_flags = common_flags
        self.common_flag_mask = common_flag_mask
        self.flags = flags
        self.flag_mask = flag_mask
        self.precond_width = precond_width
        self.get_steps()
        self.use_noise_prior = use_noise_prior
        if self.use_noise_prior:
            self.get_filters_and_preconditioners()
        return

    @function_timer
    def get_filters_and_preconditioners(self):
        """Compute and store the filter and associated preconditioner
        for every detector and every observation
        """
        log = Logger.get()
        self.filters = []  # all observations
        self.preconditioners = []  # all observations
        for iobs, obs in enumerate(self.data.obs):
            if "noise" not in obs:
                # If the observations do not include noise PSD:s, we
                # we cannot build filters.
                if len(self.filters) > 0:
                    log.warning(
                        'Observation "{}" does not have noise information'
                        "".format(obs["name"])
                    )
                continue
            tod = obs["tod"]
            # Determine the binning for the noise prior
            times = tod.local_times()
            dtime = np.amin(np.diff(times))
            fsample = 1 / dtime
            obstime = times[-1] - times[0]
            tbase = self.step_length
            fbase = 1 / tbase
            powmin = np.floor(np.log10(1 / obstime)) - 1
            powmax = min(np.ceil(np.log10(1 / tbase)) + 2, fsample)
            freq = np.logspace(powmin, powmax, 1000)
            # Now build the filter for each detector
            noise = obs["noise"]
            noisefilters = {}  # this observation
            preconditioners = {}  # this observation
            for det in tod.local_dets:
                offset_psd = self._get_offset_psd(noise, freq, det)
                # Store real space filters for every interval and every detector.
                (
                    noisefilters[det],
                    preconditioners[det],
                ) = self._get_noisefilter_and_preconditioner(
                    freq, offset_psd, self.offset_slices[iobs][det]
                )
            self.filters.append(noisefilters)
            self.preconditioners.append(preconditioners)
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
    def _get_noisefilter_and_preconditioner(self, freq, offset_psd, offset_slices):
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

        noisefilters = []
        preconditioners = []
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

    @function_timer
    def get_steps(self):
        """Divide each interval into offset steps"""
        self.offset_templates = []
        self.offset_slices = []  # slices in all observations
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            common_flags = tod.local_common_flags(self.common_flags)
            common_flags = (common_flags & self.common_flag_mask) != 0
            if (self.intervals is not None) and (self.intervals in obs):
                intervals = obs[self.intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)
            times = tod.local_times()
            offset_slices = {}  # slices in this observation
            for ival in local_intervals:
                length = times[ival.last] - times[ival.first]
                nbase = int(np.ceil(length / self.step_length))
                # Divide the interval into steps, allowing for irregular sampling
                todslices = []
                start_times = np.arange(nbase) * self.step_length + ival.start
                start_indices = np.searchsorted(times, start_times)
                stop_indices = np.hstack([start_indices[1:], [ival.last]])
                todslices = []
                for istart, istop in zip(start_indices, stop_indices):
                    todslices.append(slice(istart, istop))
                for idet, det in enumerate(tod.local_dets):
                    istart = self.namplitude
                    sigmasqs = []
                    for todslice in todslices:
                        sigmasq = self._get_sigmasq(
                            tod, det, todslice, common_flags, self.detweights[iobs][det]
                        )
                        # Register the baseline offset
                        self.offset_templates.append(
                            [self.namplitude, iobs, det, todslice, sigmasq]
                        )
                        sigmasqs.append(sigmasq)
                        self.namplitude += 1
                    # Keep a record of ranges of offsets that correspond
                    # to one detector and one interval.
                    # This is the domain we apply the noise filter in.
                    if det not in offset_slices:
                        offset_slices[det] = []
                    offset_slices[det].append(
                        (slice(istart, self.namplitude), sigmasqs)
                    )
            self.offset_slices.append(offset_slices)
        return

    @function_timer
    def _get_sigmasq(self, tod, det, todslice, common_flags, detweight):
        """calculate a rough estimate of the baseline variance
        for diagonal preconditioner
        """
        flags = tod.local_flags(det, self.flags)[todslice]
        good = (flags & self.flag_mask) == 0
        good[common_flags[todslice]] = False
        ngood = np.sum(good)
        sigmasq = 1
        if detweight != 0:
            sigmasq /= detweight
        if ngood != 0:
            sigmasq /= ngood
        return sigmasq

    @function_timer
    def add_to_signal(self, signal, amplitudes):
        offset_amplitudes = amplitudes[self.name]
        last_obs = None
        last_det = None
        last_ref = None
        todslices = []
        itemplates = []
        for itemplate, iobs, det, todslice, sigmasq in self.offset_templates:
            if iobs != last_obs or det != last_det:
                if len(todslices) != 0:
                    add_offsets_to_signal(
                        last_ref, todslices, offset_amplitudes, np.array(itemplates)
                    )
                todslices = []
                itemplates = []
                last_obs = iobs
                last_det = det
                last_ref = signal[iobs, det, :]
            todslices.append(todslice)
            itemplates.append(itemplate)
        if len(todslices) != 0:
            add_offsets_to_signal(
                last_ref, todslices, offset_amplitudes, np.array(itemplates)
            )
        return

    @function_timer
    def project_signal(self, signal, amplitudes):
        offset_amplitudes = amplitudes[self.name]
        last_obs = None
        last_det = None
        last_ref = None
        todslices = []
        itemplates = []
        for itemplate, iobs, det, todslice, sqsigma in self.offset_templates:
            if iobs != last_obs or det != last_det:
                if len(todslices) != 0:
                    project_signal_offsets(
                        last_ref, todslices, offset_amplitudes, np.array(itemplates)
                    )
                todslices = []
                itemplates = []
                last_obs = iobs
                last_det = det
                last_ref = signal[iobs, det, :]
            todslices.append(todslice)
            itemplates.append(itemplate)
        if len(todslices) != 0:
            project_signal_offsets(
                last_ref, todslices, offset_amplitudes, np.array(itemplates)
            )
        return

    @function_timer
    def add_prior(self, amplitudes_in, amplitudes_out):
        if not self.use_noise_prior:
            return
        offset_amplitudes_in = amplitudes_in[self.name]
        offset_amplitudes_out = amplitudes_out[self.name]
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            for det in tod.local_dets:
                slices = self.offset_slices[iobs][det]
                filters = self.filters[iobs][det]
                for (offsetslice, sigmasqs), noisefilter in zip(slices, filters):
                    amps_in = offset_amplitudes_in[offsetslice]
                    # scipy.signal.convolve will use either `convolve` or `fftconvolve`
                    # depending on the size of the inputs
                    amps_out = scipy.signal.convolve(amps_in, noisefilter, mode="same")
                    offset_amplitudes_out[offsetslice] += amps_out
        return

    @function_timer
    def apply_precond(self, amplitudes_in, amplitudes_out):
        offset_amplitudes_in = amplitudes_in[self.name]
        offset_amplitudes_out = amplitudes_out[self.name]
        if self.use_noise_prior:
            # C_a preconditioner
            for iobs, obs in enumerate(self.data.obs):
                tod = obs["tod"]
                for det in tod.local_dets:
                    slices = self.offset_slices[iobs][det]
                    preconditioners = self.preconditioners[iobs][det]
                    for (offsetslice, sigmasqs), preconditioner in zip(
                        slices, preconditioners
                    ):
                        amps_in = offset_amplitudes_in[offsetslice]
                        if self.precond_width <= 1:
                            # Use C_a prior
                            # scipy.signal.convolve will use either `convolve` or `fftconvolve`
                            # depending on the size of the inputs
                            amps_out = scipy.signal.convolve(
                                amps_in, preconditioner, mode="same"
                            )
                        else:
                            # Use pre-computed Cholesky decomposition
                            amps_out = scipy.linalg.cho_solve_banded(
                                preconditioner,
                                amps_in,
                                overwrite_b=False,
                                check_finite=True,
                            )
                        offset_amplitudes_out[offsetslice] = amps_out
        else:
            # Diagonal preconditioner
            offset_amplitudes_out[:] = offset_amplitudes_in
            for itemplate, iobs, det, todslice, sigmasq in self.offset_templates:
                offset_amplitudes_out[itemplate] *= sigmasq
        return


class TemplateMatrix(TOASTMatrix):
    def __init__(self, data, comm, templates=None):
        """Initialize the template matrix with a given baseline length"""
        self.data = data
        self.comm = comm
        self.templates = OrderedDict()
        for template in templates:
            self.register_template(template)
        return

    @function_timer
    def register_template(self, template):
        """Add template to the list of templates to fit"""
        self.templates[template.name] = template

    @function_timer
    def apply(self, amplitudes):
        """Compute and return y = F.a"""
        new_signal = self.zero_signal()
        for template in self.templates.values():
            template.add_to_signal(new_signal, amplitudes)
        return new_signal

    @function_timer
    def apply_transpose(self, signal):
        """Compute and return a = F^T.y"""
        new_amplitudes = self.zero_amplitudes()
        for template in self.templates.values():
            template.project_signal(signal, new_amplitudes)
        return new_amplitudes

    @function_timer
    def add_prior(self, amplitudes, new_amplitudes):
        """Compute a' += C_a^{-1}.a"""
        for template in self.templates.values():
            template.add_prior(amplitudes, new_amplitudes)
        return

    @function_timer
    def apply_precond(self, amplitudes):
        """Compute a' = M^{-1}.a"""
        new_amplitudes = self.zero_amplitudes()
        for template in self.templates.values():
            template.apply_precond(amplitudes, new_amplitudes)
        return new_amplitudes

    @function_timer
    def zero_amplitudes(self):
        """Return a null amplitudes object"""
        new_amplitudes = TemplateAmplitudes(self.templates, self.comm)
        return new_amplitudes

    @function_timer
    def zero_signal(self):
        """Return a distributed vector of signal set to zero.

        The zero signal object will use the same TOD objects but different cache prefix
        """
        new_signal = Signal(self.data, temporary=True, init_val=0)
        return new_signal

    @function_timer
    def clean_signal(self, signal, amplitudes, in_place=True):
        """Clean the given distributed signal vector by subtracting
        the templates multiplied by the given amplitudes.
        """
        # DEBUG begin
        """
        import pdb
        import matplotlib.pyplot as plt
        plt.figure(figsize=[18, 12])
        for sig in [signal]:
            tod = sig.data.obs[0]["tod"]
            for idet, det in enumerate(tod.local_dets):
                plt.subplot(2, 2, idet + 1)
                plt.plot(tod.local_signal(det, sig.name), label=sig.name, zorder=50)
        """
        # DEBUG end
        if in_place:
            outsignal = signal
        else:
            outsignal = signal.copy()
        template_tod = self.apply(amplitudes)
        outsignal -= template_tod
        # DEBUG begin
        """
        for sig, zorder in [(template_tod, 100), (outsignal, 0)]:
            tod = sig.data.obs[0]["tod"]
            for idet, det in enumerate(tod.local_dets):
                plt.subplot(2, 2, idet + 1)
                plt.plot(tod.local_signal(det, sig.name), label=sig.name, zorder=zorder)
        plt.legend(loc="best")
        plt.savefig("test.png")
        plt.close()
        #pdb.set_trace()
        """
        # DEBUG end
        return outsignal

    @function_timer
    def calibrate_signal(self, signal, amplitudes, in_place=True):
        """Apply the estimate gain fluctuation to the distributed signal  vector by dividing
        the templates multiplied by the given amplitudes.
        """

        if in_place:
            outsignal = signal
        else:
            outsignal = signal.copy()
        for  template in self.templates.values():
            delta_gain = template.calibrate(outsignal, amplitudes)
            import pdb
            pdb.set_trace()
            print(delta_gain,template.name )
            if delta_gain is not None :
                outsignal /= delta_gain

        return outsignal

class TemplateAmplitudes(TOASTVector):
    """TemplateAmplitudes objects hold local and shared template amplitudes"""

    def __init__(self, templates, comm):
        self.comm = comm
        self.amplitudes = OrderedDict()
        self.comms = OrderedDict()
        for template in templates.values():
            self.amplitudes[template.name] = np.zeros(template.namplitude)
            self.comms[template.name] = template.comm
        return

    @function_timer
    def __str__(self):
        result = "template amplitudes:"
        for name, values in self.amplitudes.items():
            result += '\n"{}" : \n{}'.format(name, values)
        return result

    @function_timer
    def dot(self, other):
        """Compute the dot product between the two amplitude vectors"""
        total = 0
        for name, values in self.amplitudes.items():
            comm = self.comms[name]
            if comm is None or comm.rank == 0:
                total += np.dot(values, other.amplitudes[name])
        if self.comm is not None:
            total = self.comm.allreduce(total, op=MPI.SUM)
        return total

    @function_timer
    def __getitem__(self, key):
        return self.amplitudes[key]

    @function_timer
    def __setitem__(self, key, value):
        self.amplitudes[name][:] = value
        return

    @function_timer
    def copy(self):
        new_amplitudes = TemplateAmplitudes(OrderedDict()  , self.comm)
        for name, values in self.amplitudes.items():
            new_amplitudes.amplitudes[name] = self.amplitudes[name].copy()
            new_amplitudes.comms[name] = self.comms[name]
        return new_amplitudes

    @function_timer
    def __iadd__(self, other):
        """Add the provided amplitudes to this one"""
        if isinstance(other, TemplateAmplitudes):
            for name, values in self.amplitudes.items():
                values += other.amplitudes[name]
        else:
            for name, values in self.amplitudes.items():
                values += other
        return self

    @function_timer
    def __isub__(self, other):
        """Subtract the provided amplitudes from this one"""
        if isinstance(other, TemplateAmplitudes):
            for name, values in self.amplitudes.items():
                values -= other.amplitudes[name]
        else:
            for name, values in self.amplitudes.items():
                values -= other
        return self

    @function_timer
    def __imul__(self, other):
        """Scale the amplitudes"""
        for name, values in self.amplitudes.items():
            values *= other
        return self

    @function_timer
    def __itruediv__(self, other):
        """Divide the amplitudes"""
        for name, values in self.amplitudes.items():
            values /= other
        return self


class TemplateCovariance(TOASTMatrix):
    def __init__(self):
        pass


class ProjectionMatrix(TOASTMatrix):
    """Projection matrix:
        Z = I - P (P^T N^{-1} P)^{-1} P^T N^{-1}
          = I - P B,
    where
         `P` is the pointing matrix
         `N` is the noise matrix and
         `B` is the binning operator
    """

    def __init__(
        self,
        data,
        comm,
        detweights,
        nnz,
        white_noise_cov_matrix,
        common_flag_mask=1,
        flag_mask=1,
    ):
        self.data = data
        self.comm = comm
        self.detweights = detweights
        self.dist_map = DistPixels(data, comm=self.comm, nnz=nnz, dtype=np.float64)
        self.white_noise_cov_matrix = white_noise_cov_matrix
        self.common_flag_mask = common_flag_mask
        self.flag_mask = flag_mask

    @function_timer
    def apply(self, signal):
        """Return Z.y"""
        self.bin_map(signal.name)
        new_signal = signal.copy()
        scanned_signal = Signal(self.data, temporary=True, init_val=0)
        self.scan_map(scanned_signal.name)
        new_signal -= scanned_signal
        return new_signal

    @function_timer
    def bin_map(self, name):
        if self.dist_map.data is not None:
            self.dist_map.data.fill(0.0)
        # FIXME: OpAccumDiag should support separate detweights for each observation
        build_dist_map = OpAccumDiag(
            zmap=self.dist_map,
            name=name,
            detweights=self.detweights[0],
            common_flag_mask=self.common_flag_mask,
            flag_mask=self.flag_mask,
        )
        build_dist_map.exec(self.data)
        self.dist_map.allreduce()
        covariance_apply(self.white_noise_cov_matrix, self.dist_map)
        return

    @function_timer
    def scan_map(self, name):
        scansim = OpSimScan(input_map=self.dist_map, out=name)
        scansim.exec(self.data)
        return


class NoiseMatrix(TOASTMatrix):
    def __init__(
        self, comm, detweights, weightmap=None, common_flag_mask=1, flag_mask=1
    ):
        self.comm = comm
        self.detweights = detweights
        self.weightmap = weightmap
        self.common_flag_mask = common_flag_mask
        self.flag_mask = flag_mask

    @function_timer
    def apply(self, signal, in_place=False):
        """Multiplies the signal with N^{-1}.

        Note that the quality flags cause the corresponding diagonal
        elements of N^{-1} to be zero.
        """
        if in_place:
            new_signal = signal
        else:
            new_signal = signal.copy()
        for iobs, detweights in enumerate(self.detweights):
            for det, detweight in detweights.items():
                new_signal[iobs, det, :] *= detweight
        # Set flagged samples to zero
        new_signal.apply_flags(self.common_flag_mask, self.flag_mask)
        # Scale the signal with the weight map
        new_signal.apply_weightmap(self.weightmap)
        return new_signal

    def apply_transpose(self, signal):
        # Symmetric matrix
        return self.apply(signal)


class PointingMatrix(TOASTMatrix):
    def __init__(self):
        pass


class Signal(TOASTVector):
    """Signal class wraps the TOAST data object but represents only
    one cached signal flavor.
    """

    def __init__(self, data, name=None, init_val=None, temporary=False):
        self.data = data
        self.temporary = temporary
        if self.temporary:
            self.name = get_temporary_name()
        else:
            self.name = name
        if init_val is not None:
            cacheinit = OpCacheInit(name=self.name, init_val=init_val)
            cacheinit.exec(data)
        return

    def __del__(self):
        if self.temporary:
            cacheclear = OpCacheClear(self.name)
            cacheclear.exec(self.data)
            free_temporary_name(self.name)
        return

    @function_timer
    def apply_flags(self, common_flag_mask, flag_mask):
        """Set the signal at flagged samples to zero"""
        flags_apply = OpFlagsApply(
            name=self.name, common_flag_mask=common_flag_mask, flag_mask=flag_mask
        )
        flags_apply.exec(self.data)
        return

    @function_timer
    def apply_weightmap(self, weightmap):
        """Scale the signal with the provided weight map"""
        if weightmap is None:
            return
        scanscale = OpScanScale(distmap=weightmap, name=self.name)
        scanscale.exec(self.data)
        return

    @function_timer
    def copy(self):
        """Return a new Signal object with independent copies of the
        signal vectors.
        """
        new_signal = Signal(self.data, temporary=True)
        copysignal = OpCacheCopy(self.name, new_signal.name, force=True)
        copysignal.exec(self.data)
        return new_signal

    @function_timer
    def __getitem__(self, key):
        """Return a reference to a slice of TOD cache"""
        iobs, det, todslice = key
        tod = self.data.obs[iobs]["tod"]
        return tod.local_signal(det, self.name)[todslice]

    @function_timer
    def __setitem__(self, key, value):
        """Set slice of TOD cache"""
        iobs, det, todslice = key
        tod = self.data.obs[iobs]["tod"]
        tod.local_signal(det, self.name)[todslice] = value
        return

    @function_timer
    def __iadd__(self, other):
        """Add the provided Signal object to this one"""
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            for det in tod.local_dets:
                if isinstance(other, Signal):
                    self[iobs, det, :] += other[iobs, det, :]
                else:
                    self[iobs, det, :] += other
        return self

    @function_timer
    def __isub__(self, other):
        """Subtract the provided Signal object from this one"""
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            for det in tod.local_dets:
                if isinstance(other, Signal):
                    self[iobs, det, :] -= other[iobs, det, :]
                else:
                    self[iobs, det, :] -= other
        return self

    @function_timer
    def __imul__(self, other):
        """Scale the signal"""
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            for det in tod.local_dets:
                self[iobs, det, :] *= other
        return self

    @function_timer
    def __itruediv__(self, other):
        """Divide the signal"""
        for iobs, obs in enumerate(self.data.obs):
            tod = obs["tod"]
            for det in tod.local_dets:
                self[iobs, det, :] /= other
        return self


class PCGSolver:
    """Solves `x` in A.x = b"""

    def __init__(
        self,
        comm,
        templates,
        noise,
        projection,
        signal,
        niter_min=3,
        niter_max=100,
        convergence_limit=1e-12,
    ):
        self.comm = comm
        if comm is None:
            self.rank = 0
        else:
            self.rank = comm.rank
        self.templates = templates
        self.noise = noise
        self.projection = projection
        self.signal = signal
        self.niter_min = niter_min
        self.niter_max = niter_max
        self.convergence_limit = convergence_limit

        self.rhs = self.templates.apply_transpose(
            self.noise.apply(self.projection.apply(self.signal))
        )
        # print("RHS {}: {}".format(self.signal.name, self.rhs))  # DEBUG
        return

    @function_timer
    def apply_lhs(self, amplitudes):
        """Return A.x"""
        new_amplitudes = self.templates.apply_transpose(
            self.noise.apply(self.projection.apply(self.templates.apply(amplitudes)))
        )
        self.templates.add_prior(amplitudes, new_amplitudes)
        return new_amplitudes

    @function_timer
    def solve(self):
        """Standard issue PCG solution of A.x = b

        Returns:
            x : the least squares solution
        """
        log = Logger.get()
        timer0 = Timer()
        timer0.start()
        timer = Timer()
        timer.start()
        # Initial guess is zero amplitudes
        guess = self.templates.zero_amplitudes()
        # print("guess:", guess)  # DEBUG
        # print("RHS:", self.rhs)  # DEBUG
        residual = self.rhs.copy()
        # print("residual(1):", residual)  # DEBUG
        residual -= self.apply_lhs(guess)
        # print("residual(2):", residual)  # DEBUG
        precond_residual = self.templates.apply_precond(residual)
        proposal = precond_residual.copy()
        sqsum = precond_residual.dot(residual)
        init_sqsum, best_sqsum, last_best = sqsum, sqsum, sqsum
        if self.rank == 0:
            log.info("Initial residual: {}".format(init_sqsum))
        # Iterate to convergence
        for iiter in range(self.niter_max):
            if not np.isfinite(sqsum):
                raise RuntimeError("Residual is not finite")
            alpha = sqsum
            alpha /= proposal.dot(self.apply_lhs(proposal))
            alpha_proposal = proposal.copy()
            alpha_proposal *= alpha
            guess += alpha_proposal
            residual -= self.apply_lhs(alpha_proposal)
            del alpha_proposal
            # Prepare for next iteration
            precond_residual = self.templates.apply_precond(residual)
            beta = 1 / sqsum
            # Check for convergence
            sqsum = precond_residual.dot(residual)
            if self.rank == 0:
                timer.report_clear(
                    "Iter = {:4} relative residual: {:12.4e}".format(
                        iiter, sqsum / init_sqsum
                    )
                )
            if sqsum < init_sqsum * self.convergence_limit or sqsum < 1e-30:
                if self.rank == 0:
                    timer0.report_clear(
                        "PCG converged after {} iterations".format(iiter)
                    )
                break
            best_sqsum = min(sqsum, best_sqsum)
            if iiter % 10 == 0 and iiter >= self.niter_min:
                if last_best < best_sqsum * 2:
                    if self.rank == 0:
                        timer0.report_clear(
                            "PCG stalled after {} iterations".format(iiter)
                        )
                    break
                last_best = best_sqsum
            # Select the next direction
            beta *= sqsum
            proposal *= beta
            proposal += precond_residual
        # log.info("{} : Solution: {}".format(self.rank, guess))  # DEBUG
        return guess


class OpMapMaker(Operator):

    # Choose one bit in the common flags for storing gap information
    gap_bit = 2 ** 7
    # Choose one bit in the quality flags for storing processing mask
    mask_bit = 2 ** 7

    def __init__(
        self,
        nside=64,
        nnz=3,
        name=None,
        outdir="out",
        outprefix="",
        write_hits=True,
        zip_maps=False,
        write_wcov_inv=True,
        write_wcov=True,
        write_binned=True,
        write_destriped=True,
        write_rcond=True,
        rcond_limit=1e-3,
        baseline_length=100000,
        maskfile=None,
        weightmapfile=None,
        common_flag_mask=1,
        flag_mask=1,
        intervals="intervals",
        subharmonic_order=None,
        fourier2D_order=None,
        fourier2D_subharmonics=False,
        gain_templatename=None,
        gain_poly_order= None,
        iter_min=3,
        iter_max=100,
        use_noise_prior=True,
        precond_width=20,
        pixels="pixels",
    ):
        self.nside = nside
        self.npix = 12 * self.nside ** 2
        self.name = name
        self.nnz = nnz
        self.ncov = self.nnz * (self.nnz + 1) // 2
        self.outdir = outdir
        self.outprefix = outprefix
        self.write_hits = write_hits
        self.zip_maps = zip_maps
        self.write_wcov_inv = write_wcov_inv
        self.write_wcov = write_wcov
        self.write_binned = write_binned
        self.write_destriped = write_destriped
        self.write_rcond = write_rcond
        self.rcond_limit = rcond_limit
        self.baseline_length = baseline_length
        self.maskfile = maskfile
        self.weightmap = None
        self.weightmapfile = weightmapfile
        self.common_flag_mask = common_flag_mask
        self.flag_mask = flag_mask
        self.intervals = intervals
        self.subharmonic_order = subharmonic_order
        self.fourier2D_order = fourier2D_order
        self.fourier2D_subharmonics = fourier2D_subharmonics
        self.gain_poly_order = gain_poly_order
        self.gain_templatename = gain_templatename
        self.iter_min = iter_min
        self.iter_max = iter_max
        self.use_noise_prior = use_noise_prior
        self.precond_width = precond_width
        self.pixels = pixels

    def report_timing(self):
        # gt.stop_all()
        all_timers = gather_timers(comm=self.comm)
        names = OrderedDict()
        names["OpMapMaker.exec"] = OrderedDict(
            [
                ("OpMapMaker.flag_gaps", None),
                ("OpMapMaker.get_detweights", None),
                ("OpMapMaker.initialize_binning", None),
                ("OpMapMaker.bin_map", None),
                ("OpMapMaker.load_mask", None),
                ("OpMapMaker.load_weightmap", None),
                ("OpMapMaker.get_templatematrix", None),
                ("OpMapMaker.get_noisematrix", None),
                ("OpMapMaker.get_projectionmatrix", None),
                ("OpMapMaker.get_solver", None),
                (
                    "PCGSolver.solve",
                    OrderedDict(
                        [
                            ("TemplateMatrix.zero_amplitudes", None),
                            ("PCGSolver.apply_lhs", None),
                            ("TemplateMatrix.apply_precond", None),
                        ]
                    ),
                ),
                ("TemplateMatrix.clean_signal", None),
            ]
        )
        names["OpMapMaker.exec"]["PCGSolver.solve"][
            "PCGSolver.apply_lhs"
        ] = OrderedDict(
            [
                (
                    "TemplateMatrix.apply_transpose",
                    OrderedDict(
                        [
                            ("OffsetTemplate.project_signal", None),
                            ("SubharmonicTemplate.project_signal", None),
                            ("fourier2DTemplate.project_signal", None),
                            ("GainTemplate.project_signal", None),
                        ]
                    ),
                ),
                ("NoiseMatrix.apply", None),
                (
                    "ProjectionMatrix.apply",
                    OrderedDict(
                        [
                            (
                                "ProjectionMatrix.bin_map",
                                OrderedDict(
                                    [
                                        (
                                            "OpAccumDiag.exec",
                                            OrderedDict(
                                                [
                                                    (
                                                        "OpAccumDiag.exec.apply_flags",
                                                        None,
                                                    ),
                                                    (
                                                        "OpAccumDiag.exec.global_to_local",
                                                        None,
                                                    ),
                                                    ("cov_accum_zmap", None),
                                                ]
                                            ),
                                        ),
                                        ("covariance_apply", None),
                                    ]
                                ),
                            ),
                            (
                                "ProjectionMatrix.scan_map",
                                OrderedDict(
                                    [
                                        (
                                            "OpSimScan.exec",
                                            OrderedDict(
                                                [
                                                    (
                                                        "OpSimScan.exec.global_to_local",
                                                        None,
                                                    ),
                                                    ("OpSimScan.exec.scan_map", None),
                                                ]
                                            ),
                                        )
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "TemplateMatrix.apply",
                    OrderedDict(
                        [
                            ("OffsetTemplate.add_to_signal", None),
                            ("SubharmonicTemplate.add_to_signal", None),
                            ("fourier2DTemplate.add_to_signal", None),
                            ("GainTemplate.add_to_signal", None),
                        ]
                    ),
                ),
                ("TemplateMatrix.add_prior", None),
            ]
        )
        if self.rank == 0:
            print("all_timers:", all_timers)  # DEBUG

            def report_line(name, indent):
                full_name = name
                if full_name not in all_timers:
                    full_name += " (function_timer)"
                if full_name not in all_timers:
                    return
                t = all_timers[full_name]["time_max"]
                print(indent, "{:.<60}{:8.1f}".format(name, t))
                return

            def report(names, indent):
                if names is None:
                    return
                if isinstance(names, str):
                    report_line(names, indent)
                else:
                    for name, entries in names.items():
                        report_line(name, indent)
                        report(entries, " " * 8 + indent)

            report(names, "-")
            print(flush=True)
        return

    @function_timer
    def get_noisematrix(self, data):
        timer = Timer()
        timer.start()
        noise = NoiseMatrix(
            self.comm,
            self.detweights,
            self.weightmap,
            common_flag_mask=(self.common_flag_mask | self.gap_bit),
            flag_mask=(self.flag_mask | self.mask_bit),
        )
        if self.rank == 0:
            timer.report_clear("Initialize projection matrix")
        return noise

    @function_timer
    def get_projectionmatrix(self, data):
        timer = Timer()
        timer.start()
        projection = ProjectionMatrix(
            data,
            self.comm,
            self.detweights,
            self.nnz,
            self.white_noise_cov_matrix,
            common_flag_mask=(self.common_flag_mask | self.gap_bit),
            # Do not add mask_bit here since it is not
            # included in the white noise matrices
            flag_mask=self.flag_mask,
        )
        if self.rank == 0:
            timer.report_clear("Initialize projection matrix")
        return projection

    @function_timer
    def get_templatematrix(self, data):
        timer = Timer()
        timer.start()
        log = Logger.get()
        templatelist = []
        if self.baseline_length is not None:
            if self.rank == 0:
                log.info(
                    "Initializing offset template, step_length = {}".format(
                        self.baseline_length
                    )
                )
            templatelist.append(
                OffsetTemplate(
                    data,
                    self.detweights,
                    step_length=self.baseline_length,
                    intervals=self.intervals,
                    common_flag_mask=(self.common_flag_mask | self.gap_bit),
                    flag_mask=(self.flag_mask | self.mask_bit),
                    use_noise_prior=self.use_noise_prior,
                    precond_width=self.precond_width,
                )
            )
        if self.subharmonic_order is not None:
            if self.rank == 0:
                log.info(
                    "Initializing subharmonic template, order = {}".format(
                        self.subharmonic_order
                    )
                )
            templatelist.append(
                SubharmonicTemplate(
                    data,
                    self.detweights,
                    order=self.subharmonic_order,
                    intervals=self.intervals,
                    common_flag_mask=(self.common_flag_mask | self.gap_bit),
                    flag_mask=(self.flag_mask | self.mask_bit),
                )
            )
        if self.fourier2D_order is not None:
            log.info(
                "Initializing fourier2D template, order = {}, subharmonics = {}".format(
                    self.fourier2D_order,
                    self.fourier2D_subharmonics,
                )
            )
            templatelist.append(
                Fourier2DTemplate(
                    data,
                    self.detweights,
                    order=self.fourier2D_order,
                    fit_subharmonics=self.fourier2D_subharmonics,
                    intervals=self.intervals,
                    common_flag_mask=(self.common_flag_mask | self.gap_bit),
                    flag_mask=(self.flag_mask | self.mask_bit),
                )
            )
        if self.gain_templatename is not None:
                    log.info(
                        f"Initializing Gain template, with Legendre polynomials,  order = {self.gain_poly_order} and {self.gain_templatename} as signal template."  )

                    templatelist.append(
                        GainTemplate(
                            data,
                            detweights=self.detweights,
                            order=self.gain_poly_order,
                            common_flag_mask=(self.common_flag_mask | self.gap_bit),
                            flag_mask=(self.flag_mask | self.mask_bit),
                            templatename =self.gain_templatename
                        )
                    )
        if len(templatelist) == 0:
            if self.rank == 0:
                log.info("No templates to fit, no destriping done.")
            templates = None
        else:
            templates = TemplateMatrix(data, self.comm, templatelist)
        if self.rank == 0:
            timer.report_clear("Initialize templates")
        return templates

    @function_timer
    def get_solver(self, data, templates, noise, projection, signal):
        timer = Timer()
        timer.start()
        solver = PCGSolver(
            self.comm,
            templates,
            noise,
            projection,
            signal,
            niter_min=self.iter_min,
            niter_max=self.iter_max,
        )
        if self.rank == 0:
            timer.report_clear("Initialize PCG solver")
        return solver

    @function_timer
    def load_mask(self, data):
        """Load processing mask and generate appropriate flag bits"""
        if self.maskfile is None:
            return
        log = Logger.get()
        timer = Timer()
        timer.start()
        if self.rank == 0 and not os.path.isfile(self.maskfile):
            raise RuntimeError(
                "Processing mask does not exist: {}".format(self.maskfile)
            )
        distmap = DistPixels(data, comm=self.comm, nnz=1, dtype=np.float32)
        distmap.read_healpix_fits(self.maskfile)
        if self.rank == 0:
            timer.report_clear("Read processing mask from {}".format(self.maskfile))

        scanmask = OpScanMask(distmap=distmap, flagmask=self.mask_bit)
        scanmask.exec(data)

        if self.rank == 0:
            timer.report_clear("Apply processing mask")

        return

    @function_timer
    def load_weightmap(self, data):
        """Load weight map"""
        if self.weightmapfile is None:
            return
        log = Logger.get()
        timer = Timer()
        timer.start()
        if self.rank == 0 and not os.path.isfile(self.weightmapfile):
            raise RuntimeError(
                "Weight map does not exist: {}".format(self.weightmapfile)
            )
        self.weightmap = DistPixels(data, comm=self.comm, nnz=1, dtype=np.float32)
        self.weightmap.read_healpix_fits(self.weightmapfile)
        if self.rank == 0:
            timer.report_clear("Read weight map from {}".format(self.weightmapfile))
        return

    @function_timer
    def exec(self, data, comm=None):
        log = Logger.get()
        timer = Timer()

        # Initialize objects
        if comm is None:
            self.comm = data.comm.comm_world
        else:
            self.comm = comm
        if self.comm is None:
            self.rank = 0
        else:
            self.rank = self.comm.rank
        self.flag_gaps(data)
        self.get_detweights(data)
        self.initialize_binning(data)
        if self.write_binned:
            self.bin_map(data, "binned")
        self.load_mask(data)
        self.load_weightmap(data)

        # Solve template amplitudes

        templates = self.get_templatematrix(data)
        if templates is None:
            return
        noise = self.get_noisematrix(data)
        projection = self.get_projectionmatrix(data)
        signal = Signal(data, name=self.name)
        solver = self.get_solver(data, templates, noise, projection, signal)
        timer.start()
        amplitudes = solver.solve()
        if self.rank == 0:
            timer.report_clear("Solve amplitudes")
        # DEBUG begin
        #if self.rank ==0  and :
        #    templates.templates["Gain"].write_gain_fluctuation(amplitudes, "gain_amplitudes.npz")
        # DEBUG end

        # Clean TOD
        if self.gain_templatename is not None:
            templates.calibrate_signal(signal, amplitudes )
        else:
            templates.clean_signal(signal, amplitudes)
        if self.rank == 0:
            timer.report_clear("Clean TOD")

        if self.write_destriped:
            self.bin_map(data, "destriped")

        return

    @function_timer
    def flag_gaps(self, data):
        """Add flag bits between the intervals"""
        timer = Timer()
        timer.start()
        flag_gaps = OpFlagGaps(common_flag_value=self.gap_bit, intervals=self.intervals)
        flag_gaps.exec(data)
        if self.rank == 0:
            timer.report_clear("Flag gaps")
        return

    @function_timer
    def bin_map(self, data, suffix):
        log = Logger.get()
        timer = Timer()

        dist_map = DistPixels(data, comm=self.comm, nnz=self.nnz, dtype=np.float64)
        if dist_map.data is not None:
            dist_map.data.fill(0.0)
        # FIXME: OpAccumDiag should support separate detweights for each observation
        build_dist_map = OpAccumDiag(
            zmap=dist_map,
            name=self.name,
            detweights=self.detweights[0],
            common_flag_mask=(self.common_flag_mask | self.gap_bit),
            flag_mask=self.flag_mask,
        )
        build_dist_map.exec(data)
        dist_map.allreduce()
        if self.rank == 0:
            timer.report_clear("  Build noise-weighted map")

        covariance_apply(self.white_noise_cov_matrix, dist_map)
        if self.rank == 0:
            timer.report_clear("  Apply noise covariance")

        fname = os.path.join(self.outdir, self.outprefix + suffix + ".fits")
        if self.zip_maps:
            fname += ".gz"
        dist_map.write_healpix_fits(fname)
        if self.rank == 0:
            timer.report_clear("  Write map to {}".format(fname))

        return

    @function_timer
    def get_detweights(self, data):
        """Each observation will have its own detweight dictionary"""
        timer = Timer()
        timer.start()
        self.detweights = []
        for obs in data.obs:
            tod = obs["tod"]
            if "noise" in obs:
                noise = obs["noise"]
            else:
                noise = None
            detweights = {}
            for det in tod.local_dets:
                if noise is None:
                    noisevar = 1
                else:
                    # Determine an approximate white noise level,
                    # accounting for the fact that the PSD may have a
                    # transfer function roll-off near Nyquist
                    freq = noise.freq(det)
                    psd = noise.psd(det)
                    rate = noise.rate(det)
                    ind = np.logical_and(freq > rate * 0.2, freq < rate * 0.4)
                    noisevar = np.median(psd[ind])
                detweights[det] = 1 / noisevar
            self.detweights.append(detweights)
        if self.rank == 0:
            timer.report_clear("Get detector weights")
        return

    @function_timer
    def initialize_binning(self, data):
        log = Logger.get()
        timer = Timer()
        timer.start()

        if self.rank == 0:
            os.makedirs(self.outdir, exist_ok=True)

        self.white_noise_cov_matrix = DistPixels(
            data, comm=self.comm, nnz=self.ncov, dtype=np.float64
        )
        if self.white_noise_cov_matrix.data is not None:
            self.white_noise_cov_matrix.data.fill(0.0)

        hits = DistPixels(data, comm=self.comm, nnz=1, dtype=np.int64)
        if hits.data is not None:
            hits.data.fill(0)

        # compute the hits and covariance once, since the pointing and noise
        # weights are fixed.
        # FIXME: OpAccumDiag should support separate weights for each observation

        build_wcov = OpAccumDiag(
            detweights=self.detweights[0],
            invnpp=self.white_noise_cov_matrix,
            hits=hits,
            common_flag_mask=(self.common_flag_mask | self.gap_bit),
            flag_mask=self.flag_mask,
        )
        build_wcov.exec(data)

        if self.comm is not None:
            self.comm.Barrier()
        if self.rank == 0:
            timer.report_clear("Accumulate N_pp'^1")

        self.white_noise_cov_matrix.allreduce()

        if self.comm is not None:
            self.comm.Barrier()
        if self.rank == 0:
            timer.report_clear("All reduce N_pp'^1")

        if self.write_hits:
            hits.allreduce()
            fname = os.path.join(self.outdir, self.outprefix + "hits.fits")
            if self.zip_maps:
                fname += ".gz"
            hits.write_healpix_fits(fname)
            if self.rank == 0:
                log.info("Wrote hits to {}".format(fname))
            if self.rank == 0:
                timer.report_clear("Write hits")

        if self.write_wcov_inv:
            fname = os.path.join(self.outdir, self.outprefix + "invnpp.fits")
            if self.zip_maps:
                fname += ".gz"
            self.white_noise_cov_matrix.write_healpix_fits(fname)
            if self.rank == 0:
                log.info("Wrote inverse white noise covariance to {}".format(fname))
            if self.rank == 0:
                timer.report_clear("Write N_pp'^1")

        if self.write_rcond:
            # Reciprocal condition numbers
            rcond = covariance_rcond(self.white_noise_cov_matrix)
            if self.rank == 0:
                timer.report_clear("Compute reciprocal condition numbers")
            fname = os.path.join(self.outdir, self.outprefix + "rcond.fits")
            if self.zip_maps:
                fname += ".gz"
            rcond.write_healpix_fits(fname)
            if self.rank == 0:
                log.info("Wrote reciprocal condition numbers to {}".format(fname))
            if self.rank == 0:
                timer.report_clear("Write rcond")

        # Invert the white noise covariance in each pixel
        covariance_invert(self.white_noise_cov_matrix, self.rcond_limit)
        if self.rank == 0:
            timer.report_clear("Invert N_pp'^1")

        if self.write_wcov:
            fname = os.path.join(self.outdir, self.outprefix + "npp.fits")
            if self.zip_maps:
                fname += ".gz"
            self.white_noise_cov_matrix.write_healpix_fits(fname)
            if self.rank == 0:
                log.info("Wrote white noise covariance to {}".format(fname))
            if self.rank == 0:
                timer.report_clear("Write N_pp'")

        return
