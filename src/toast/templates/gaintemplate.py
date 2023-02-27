# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import numpy as np
import scipy

from ..mpi import MPI
from ..traits import Int, Unicode, trait_docs
from .amplitudes import Amplitudes
from .template import Template


@trait_docs
class GainTemplate(Template):
    """This class aims at fitting and mitigating gain fluctuations in the data.
    The fluctuations  are modeled as a linear combination of Legendre polynomials (up
    to a given order, commonly `n<5` ) weighted by the so called _gain amplitudes_.
    The gain template is therefore obtained by estimating the polynomial amplitudes by
    assuming a _signal estimate_ provided by a template map (encoding a coarse estimate
    of the underlying signal.)

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

    order = Int(1, help="The order of Legendre polynomials to fit the gain amplitudes ")

    template_name = Unicode(
        None,
        allow_none=True,
        help="detdata key encoding the signal estimate to fit the gain amplitudes",
    )

    noise_model = Unicode(
        None,
        allow_none=True,
        help="Observation key containing the   noise model ",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_polynomials(self, N):
        norder = self.order + 1
        L = np.zeros((N, norder), dtype=np.float64)
        x = np.linspace(-1.0, 1.0, num=N, endpoint=True)
        for i in range(norder):
            L[:, i] = scipy.special.legendre(i)(x)
        return L

    def _initialize(self, new_data):
        self.norder = self.order + 1
        # Use this as an "Ordered Set".  We want the unique detectors on this process,
        # but sorted in order of occurrence.
        all_dets = OrderedDict()

        for iob, ob in enumerate(new_data.obs):
            # Build up detector list
            for d in ob.local_detectors:
                if d not in all_dets:
                    all_dets[d] = None

        self._all_dets = list(all_dets.keys())

        # The inverse variance units
        invvar_units = 1.0 / (self.det_data_units**2)

        # Go through the data one local detector at a time and compute the offsets into
        # the amplitudes.

        # The starting amplitude for each detector within the local amplitude data.
        self._det_start = dict()

        offset = 0
        for det in self._all_dets:
            self._det_start[det] = offset
            for iob, ob in enumerate(new_data.obs):
                if det not in ob.local_detectors:
                    continue
                # We have one set of amplitudes for each detector in each view
                offset += len(ob.view[self.view]) * (self.order + 1)

        # Now we know the total number of amplitudes.

        self._n_local = offset
        if new_data.comm.comm_world is None:
            self._n_global = self._n_local
        else:
            self._n_global = new_data.comm.comm_world.allreduce(
                self._n_local, op=MPI.SUM
            )

        # The preconditioner for each obs / view / detector
        self._precond = dict()
        self._templates = dict()

        # Build the preconditioner .
        # it is easier to just build these by looping in observation order rather than
        # detector order.

        for iob, ob in enumerate(new_data.obs):
            # Build the templates and preconditioners for every view.
            self._templates[iob] = list()
            self._precond[iob] = dict()
            norder = self.order + 1

            noise = None
            if self.noise_model in ob:
                noise = ob[self.noise_model]
            # import pdb; pdb.set_trace()
            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                view_len = None
                if vw.start is None:
                    # This is a view of the whole obs
                    view_len = ob.n_local_samples
                else:
                    view_len = vw.stop - vw.start
                # get legendre polynomials
                L = self._get_polynomials(view_len)
                # store them in the template dictionary
                self._templates[iob].append(L)

                self._precond[iob][ivw] = dict()
                for det in ob.local_detectors:
                    detweight = 1.0
                    if noise is not None:
                        detweight = noise.detector_weight(det).to_value(invvar_units)

                    good = slice(0, view_len, 1)
                    if self.det_flags is not None:
                        flags = views.detdata[self.det_flags][ivw][det]
                        good = (flags & self.det_flag_mask) == 0

                    prec = np.zeros((norder, norder), dtype=np.float64)
                    T = ob.detdata[self.template_name][det]

                    LT = L.T.copy()
                    for row in LT:
                        row *= T * np.sqrt(detweight)
                    M = LT.dot(LT.T)
                    self._precond[iob][ivw][det] = np.linalg.inv(M)

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(self.data.comm, self._n_global, self._n_local)
        # No explicit flagging of amplitudes in this template...
        # z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _add_to_signal(self, detector, amplitudes, **kwargs):
        norder = self.order + 1
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                legendre_poly = self._templates[iob][ivw]
                poly_amps = amplitudes.local[offset : offset + norder]
                delta_gain = legendre_poly.dot(poly_amps)
                signal_estimate = ob.detdata[self.template_name][detector]
                gain_fluctuation = signal_estimate * delta_gain
                vw[detector] += gain_fluctuation

    def _project_signal(self, detector, amplitudes, **kwargs):
        norder = self.order + 1
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                legendre_poly = self._templates[iob][ivw]
                signal_estimate = ob.detdata[self.template_name][detector]
                if self.det_flags is not None:
                    flagview = ob.view[self.view].detdata[self.det_flags][ivw]
                    mask = (flagview[detector] & self.det_flag_mask) == 0
                else:
                    mask = 1
                LT = legendre_poly.T.copy()
                for row in LT:
                    row *= signal_estimate
                poly_amps = amplitudes.local[offset : offset + norder]
                poly_amps += np.dot(LT, vw[detector] * mask)

    def _add_prior(self, amplitudes_in, amplitudes_out, **kwargs):
        # No prior for this template, nothing to accumulate to output.
        return

    def _apply_precond(self, amplitudes_in, amplitudes_out, **kwargs):
        norder = self.order + 1
        for det in self._all_dets:
            offset = self._det_start[det]
            for iob, ob in enumerate(self.data.obs):
                if det not in ob.local_detectors:
                    continue
                views = ob.view[self.view]
                for ivw, vw in enumerate(views):
                    amps_in = amplitudes_in.local[offset : offset + norder]
                    amps_out = amplitudes_out.local[offset : offset + norder]
                    amps_out[:] = np.dot(self._precond[iob][ivw][det], amps_in)
                    offset += norder
