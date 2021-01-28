# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from astropy import units as u

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance, Float

from ..data import Data

from .template import Template


@trait_docs
class Fourier2D(Template):
    """This class represents atmospheric fluctuations in front of the focalplane
    as 2D Fourier modes.

    """

    # Notes:  The TraitConfig base class defines a "name" attribute.  The Template
    # class (derived from TraitConfig) defines the following traits already:
    #    data             : The Data instance we are working with
    #    view             : The timestream view we are using
    #    det_data         : The detector data key with the timestreams
    #    flags            : Optional detector solver flags
    #    flag_mask        : Bit mask for detector solver flags
    #

    correlation_length = Quantity(10.0 * u.second, help="Correlation length in time")

    correlation_amplitude = Float(10.0, help="Scale factor of the filter")

    order = Int(1, help="The filter order")

    fit_subharmonics = Bool(True, help="If True, fit subharmonics")

    noise_model = Unicode(
        None,
        allow_none=True,
        help="Observation key containing the optional noise model",
    )

    @traitlets.validate("order")
    def _check_order(self, proposal):
        od = proposal["value"]
        if od < 1:
            raise traitlets.TraitError("Filter order should be >= 1")
        return od

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
        self._norder = self.order + 1
        self._nmode = (2 * self.order) ** 2 + 1
        if self.fit_subharmonics:
            self._nmode += 2

        def evaluate_template(theta, phi, radius):
            """Helper function to get the template values for a detector."""
            values = np.zeros(self._nmode)
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

        # The detector templates and norms for each observation
        self.templates = dict()
        self.norms = dict()

        # Amplitude lengths of all views for each obs
        self._obs_view_namp = dict()

        # Starting amplitude for each view within each obs
        self._obs_view_offset = dict()

        # Sample rate for each obs.
        self._obs_rate = dict()

        offset = 0

        for iob, ob in enumerate(new_data.obs):
            # Compute sample rate from timestamps
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(ob.shared[self.times])
            self._obs_rate[iob] = rate

            # Focalplane radius
            radius = np.radians(ob.telescope.focalplane.radius)

            noise = None
            if self.noise_model in ob:
                noise = ob[self.noise_model]

            # Track number of offset amplitudes per view.
            self._obs_view_namp[iob] = list()
            self._obs_view_offset[iob] = list()

            for view_slice in ob.view[self.view]:
                slice_len = None
                if view_slice.start is None:
                    # This is a view of the whole obs
                    slice_len = ob.n_local_samples
                else:
                    slice_len = view_slice.stop - view_slice.start

                view_norms = np.zeros((slice_len, self._nmode))
                view_templates = dict()

                for det in ob.local_detectors:
                    detweight = 1.0
                    if noise is not None:
                        detweight = noise.detector_weight(det)
                    det_quat = ob.focalplane.detector_quats[det]
                    x, y, z = qa.rotate(det_quat, ZAXIS)
                    theta, phi = np.arcsin([x, y])
                    view_templates[det] = evaluate_template(theta, phi, radius)

                view_n_amp = slice_len * self._nmode
                self._obs_view_namp[iob].append(view_n_amp)
                self._obs_view_offset[iob].append(offset)
                offset += view_n_amp

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

        self._get_templates()
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
            values = np.zeros(self._nmode)
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

    def _zeros(self):
        raise NotImplementedError("Derived class must implement _zeros()")

    def _add_to_signal(self, detector, amplitudes):
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

    def _project_signal(self, detector, amplitudes):
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

    def _add_prior(self, amplitudes_in, amplitudes_out):
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

    def _apply_precond(self, amplitudes_in, amplitudes_out):
        poly_amplitudes_in = amplitudes_in[self.name]
        poly_amplitudes_out = amplitudes_out[self.name]
        poly_amplitudes_out[:] = poly_amplitudes_in * self.norms
