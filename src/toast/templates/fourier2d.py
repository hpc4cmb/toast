# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import numpy as np
import scipy
import scipy.signal
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import AlignedF64, Logger
from .amplitudes import Amplitudes
from .template import Template


@trait_docs
class Fourier2D(Template):
    """This class models 2D Fourier modes across the focalplane.

    Since the modes are shared across detectors, our amplitudes are organized by
    observation and views within each observation.  Each detector projection
    will traverse all the local amplitudes.

    """

    # Notes:  The TraitConfig base class defines a "name" attribute.  The Template
    # class (derived from TraitConfig) defines the following traits already:
    #    data             : The Data instance we are working with
    #    view             : The timestream view we are using
    #    det_data         : The detector data key with the timestreams
    #    det_data_units   : The units of the detector data
    #    det_flags        : Optional detector solver flags
    #    det_flag_mask    : Bit mask for detector solver flags
    #

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

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

    def clear(self):
        """Delete the underlying C-allocated memory."""
        if hasattr(self, "_norms"):
            del self._norms
        if hasattr(self, "_norms_raw"):
            self._norms_raw.clear()
            del self._norms_raw

    def __del__(self):
        self.clear()

    def _initialize(self, new_data):
        zaxis = np.array([0.0, 0.0, 1.0])

        # This function is called whenever a new data trait is assigned to the template.
        # Clear any C-allocated buffers from previous uses.
        self.clear()

        self._norder = self.order + 1
        self._nmode = (2 * self.order) ** 2 + 1
        if self.fit_subharmonics:
            self._nmode += 2

        # The inverse variance units
        invvar_units = 1.0 / (self.det_data_units**2)

        # Every process determines their local amplitude ranges.

        # The local ranges of amplitudes (in terms of global indices)
        self._local_ranges = list()

        # Starting local amplitude for each view within each obs
        self._obs_view_local_offset = dict()

        # Starting global amplitude for each view within each obs
        self._obs_view_global_offset = dict()

        # Number of amplitudes in each local view for each obs
        self._obs_view_namp = dict()

        # This is the total number of amplitudes for each observation, across all
        # views.
        self._obs_total_namp = dict()

        # Use this as an "Ordered Set".  We want the unique detectors on this process,
        # but sorted in order of occurrence.
        all_dets = OrderedDict()

        local_offset = 0
        global_offset = 0

        for iob, ob in enumerate(new_data.obs):
            self._obs_view_namp[iob] = list()
            self._obs_view_local_offset[iob] = list()
            self._obs_view_global_offset[iob] = list()

            # Build up detector list
            for d in ob.local_detectors:
                if d not in all_dets:
                    all_dets[d] = None

            obs_n_amp = 0

            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                # First obs sample of this view
                obs_start = ob.local_index_offset

                view_len = None
                if vw.start is None:
                    # This is a view of the whole obs
                    view_len = ob.n_local_samples
                else:
                    view_len = vw.stop - vw.start
                    obs_start += vw.start

                obs_offset = obs_start * self._nmode

                self._obs_view_local_offset[iob].append(local_offset)
                self._obs_view_global_offset[iob].append(global_offset + obs_offset)

                view_n_amp = view_len * self._nmode
                obs_n_amp += view_n_amp
                self._obs_view_namp[iob].append(view_n_amp)

                self._local_ranges.append((global_offset + obs_offset, view_n_amp))

                local_offset += view_n_amp

            # To get the total number of amplitudes in this observation, we must
            # accumulate across the grid row communicator.
            if ob.comm_row is not None:
                obs_n_amp = ob.comm_row.allreduce(obs_n_amp)
            self._obs_total_namp[iob] = obs_n_amp
            global_offset += obs_n_amp

        self._all_dets = list(all_dets.keys())

        # The global number of amplitudes for our process group and our local process.
        # Since different groups have different observations, their amplitude values
        # are completely disjoint.  We create Amplitudes with the `use_group` option
        # and so only have to consider the full set if we are doing things like I/O
        # (i.e. nothing needed by this class).

        self._n_global = np.sum(
            [self._obs_total_namp[x] for x, y in enumerate(new_data.obs)]
        )

        self._n_local = np.sum([x[1] for x in self._local_ranges])

        # Allocate norms.  This data is the same size as a set of amplitudes,
        # so we allocate it in C memory.

        self._norms_raw = AlignedF64.zeros(self._n_local)
        self._norms = self._norms_raw.array()

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

        # The detector templates for each observation
        self._templates = dict()

        # The noise filter for each observation
        self._filters = dict()

        for iob, ob in enumerate(new_data.obs):
            # Focalplane for this observation
            fp = ob.telescope.focalplane

            # Focalplane radius
            radius = 0.5 * fp.field_of_view.to_value(u.radian)

            noise = None
            if self.noise_model in ob:
                noise = ob[self.noise_model]

            self._templates[iob] = list()
            self._filters[iob] = list()

            obs_local_namp = 0

            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                view_len = None
                if vw.start is None:
                    # This is a view of the whole obs
                    view_len = ob.n_local_samples
                else:
                    view_len = vw.stop - vw.start

                # Build the filter for this view

                corr_len = self.correlation_length.to_value(u.second)
                times = views.shared[self.times][ivw]
                corr = (
                    np.exp((times[0] - times) / corr_len) * self.correlation_amplitude
                )
                ihalf = times.size // 2
                if times.size % 2 == 0:
                    corr[ihalf:] = corr[ihalf - 1 :: -1]
                else:
                    corr[ihalf + 1 :] = corr[ihalf - 1 :: -1]
                fcorr = np.fft.rfft(corr)
                too_small = fcorr < (1.0e-6 * self.correlation_amplitude)
                fcorr[too_small] = 1.0e-6 * self.correlation_amplitude
                invcorr = np.fft.irfft(1 / fcorr)
                self._filters[iob].append(invcorr)

                # Now compute templates and norm for this view

                view_templates = dict()

                good = np.empty(view_len, dtype=np.float64)
                norm_slice = slice(
                    self._obs_view_local_offset[iob][ivw],
                    self._obs_view_local_offset[iob][ivw]
                    + self._obs_view_namp[iob][ivw],
                    1,
                )
                norms_view = self._norms[norm_slice].reshape((-1, self._nmode))

                for det in ob.local_detectors:
                    detweight = 1.0
                    if noise is not None:
                        detweight = noise.detector_weight(det).to_value(invvar_units)
                    det_quat = fp[det]["quat"]
                    x, y, z = qa.rotate(det_quat, zaxis)
                    theta, phi = np.arcsin([x, y])
                    view_templates[det] = evaluate_template(theta, phi, radius)

                    good[:] = 1.0
                    if self.det_flags is not None:
                        flags = views.detdata[self.det_flags][ivw][det]
                        good[(flags & self.det_flag_mask) != 0] = 0
                    norms_view += np.outer(good, view_templates[det] ** 2 * detweight)

                obs_local_namp += self._obs_view_namp[iob][ivw]
                self._templates[iob].append(view_templates)

            # Reduce norm values across the process grid column
            norm_slice = slice(
                self._obs_view_local_offset[iob][0],
                self._obs_view_local_offset[iob][0] + obs_local_namp,
                1,
            )
            norms_view = self._norms[norm_slice]
            if ob.comm_col is not None:
                temp = np.array(norms_view)
                ob.comm_col.Allreduce(temp, norms_view, op=MPI.SUM)
                del temp

            # Invert norms
            good = norms_view != 0
            norms_view[good] = 1.0 / norms_view[good]

        # Set the filter scale by the prescribed correlation strength
        # and the number of modes at each angular scale
        self._filter_scale = np.zeros(self._nmode)
        self._filter_scale[0] = 1
        offset = 1
        if self.fit_subharmonics:
            self._filter_scale[1:3] = 2
            offset += 2
        self._filter_scale[offset:] = 4
        self._filter_scale *= self.correlation_amplitude

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        # Return amplitudes distributed over the group communicator and using our
        # local ranges.
        z = Amplitudes(
            self.data.comm,
            self._n_global,
            self._n_local,
            local_ranges=self._local_ranges,
        )
        # Amplitude flags are not used by this template- if some samples are flagged
        # across all detectors then they will just not contribute to the projection.
        # z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _add_to_signal(self, detector, amplitudes, **kwargs):
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            if detector not in ob.detdata[self.det_data].detectors:
                continue
            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                amp_slice = slice(
                    self._obs_view_local_offset[iob][ivw],
                    self._obs_view_local_offset[iob][ivw]
                    + self._obs_view_namp[iob][ivw],
                    1,
                )
                views.detdata[self.det_data][ivw][detector] += np.sum(
                    amplitudes.local[amp_slice].reshape((-1, self._nmode))
                    * self._templates[iob][ivw][detector],
                    1,
                )

    def _project_signal(self, detector, amplitudes, **kwargs):
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            if detector not in ob.detdata[self.det_data].detectors:
                continue
            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                amp_slice = slice(
                    self._obs_view_local_offset[iob][ivw],
                    self._obs_view_local_offset[iob][ivw]
                    + self._obs_view_namp[iob][ivw],
                    1,
                )
                amp_view = amplitudes.local[amp_slice].reshape((-1, self._nmode))
                amp_view[:] += np.outer(
                    views.detdata[self.det_data][ivw][detector],
                    self._templates[iob][ivw][detector],
                )

    def _add_prior(self, amplitudes_in, amplitudes_out, **kwargs):
        for iob, ob in enumerate(self.data.obs):
            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                amp_slice = slice(
                    self._obs_view_local_offset[iob][ivw],
                    self._obs_view_local_offset[iob][ivw]
                    + self._obs_view_namp[iob][ivw],
                    1,
                )
                in_view = amplitudes_in.local[amp_slice].reshape((-1, self._nmode))
                out_view = amplitudes_out.local[amp_slice].reshape((-1, self._nmode))
                for mode in range(self._nmode):
                    scale = self._filter_scale[mode]
                    out_view[:, mode] += scipy.signal.convolve(
                        in_view[:, mode],
                        self._filters[iob][ivw] * scale,
                        mode="same",
                    )

    def _apply_precond(self, amplitudes_in, amplitudes_out, **kwargs):
        amplitudes_out.local[:] = amplitudes_in.local
        amplitudes_out.local *= self._norms
