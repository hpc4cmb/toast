# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import numpy as np

from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..traits import Bool, Float, Instance, Int, Unicode, trait_docs
from ..utils import Logger
from .amplitudes import Amplitudes
from .template import Template


@trait_docs
class SubHarmonic(Template):
    """This class represents sub-harmonic noise fluctuations.

    Sub-harmonic means that the characteristic frequency of the noise
    modes is lower than 1/T where T is the length of the interval
    being fitted.

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
    #    det_flags        : Optional detector solver flags
    #    det_flag_mask    : Bit mask for detector solver flags
    #

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    order = Int(1, help="The filter order")

    noise_model = Unicode(
        None,
        allow_none=True,
        help="Observation key containing the optional noise model",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
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

        # The templates for each view of each obs
        self._templates = dict()

        # The preconditioner for each obs / view / detector
        self._precond = dict()

        # We are not constructing any data objects that are in the same order as the
        # amplitudes (we are just building dictionaries for lookups).  In this case,
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

            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                view_len = None
                if vw.start is None:
                    # This is a view of the whole obs
                    view_len = ob.n_local_samples
                else:
                    view_len = vw.stop - vw.start

                templates = np.zeros((norder, view_len), dtype=np.float64)
                r = np.linspace(-1.0, 1.0, view_len)
                for order in range(norder):
                    if order == 0:
                        templates[order] = 1.0
                    elif order == 1:
                        templates[order] = r
                    else:
                        templates[order] = (
                            (2 * order - 1) * r * templates[order - 1]
                            - (order - 1) * templates[order - 2]
                        ) / order
                self._templates[iob].append(templates)

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
                    for row in range(norder):
                        for col in range(row, norder):
                            prec[row, col] = np.dot(
                                templates[row][good], templates[col][good]
                            )
                            prec[row, col] *= detweight
                            if row != col:
                                prec[col, row] = prec[row, col]
                    self._precond[iob][ivw][det] = np.linalg.inv(prec)

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
                amp_view = amplitudes.local[offset : offset + norder]
                for order in range(norder):
                    vw[detector] += self._templates[iob][ivw][order] * amp_view[order]
                offset += norder

    def _project_signal(self, detector, amplitudes, **kwargs):
        norder = self.order + 1
        offset = self._det_start[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in ob.local_detectors:
                continue
            for ivw, vw in enumerate(ob.view[self.view].detdata[self.det_data]):
                amp_view = amplitudes.local[offset : offset + norder]
                for order, template in enumerate(self._templates[iob][ivw]):
                    amp_view[order] = np.dot(vw[detector], template)
                offset += norder

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
