# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
from time import time
import warnings

from astropy import units as u
import numpy as np
import traitlets

from ..mpi import MPI, MPI_Comm, use_mpi, Comm

from .operator import Operator
from .. import qarray as qa
from ..timing import function_timer
from ..traits import trait_docs, Int, Unicode, Bool, Dict, Quantity, Instance
from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned


from .._libtoast import filter_polynomial




XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class PolyFilter2D(Operator):
    """Operator to regress out 2D polynomials across the focal plane."""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode("signal", help="Observation detdata key apply the gain error to")

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are filtered.",
    )

    order = Int(1, allow_none=False, help="Polynomial order")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    poly_flag_mask = Int(0, help="Bit mask value for intervals that fail to filter")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        """Apply the 2D polynomial filter to the signal.

        Args:
            data (toast.Data): The distributed data.

        """
        if detectors is not None:
            raise RuntimeError("PolyFilter2D cannot be run in batch mode")
        nmode = (self.order + 1) * (self.order + 2) // 2
        norder = self.order + 1
        pat = re.compile(self.pattern)

        for obs in data.obs:
            t0 = time()
            t_template = 0
            t_get_norm = 0
            t_apply_norm = 0
            t_solve = 0
            t_clean = 0

            comm = obs.dist.comm_row

            detector_index = {}
            ndet = 0
            for det in obs.all_detectors:
                if pat.match(det) is None:
                    continue
                detector_index[det] = ndet
                ndet += 1

            # Number of detectors may limit the number of modes we can constrain
            nmode = min(nmode, ndet)

            detector_templates = np.zeros([ndet, nmode])
            mode = 0
            xorders = np.zeros(nmode)
            yorders = np.zeros(nmode)
            for order in range(norder):
                for yorder in range(order + 1):
                    xorder = order - yorder
                    xorders[mode] = xorder
                    yorders[mode] = yorder
                    mode += 1
                    if mode == nmode:
                        break
                if mode == nmode:
                    break

            for det in obs.local_detectors:
                if det not in detector_index:
                    continue
                idet = detector_index[det]
                det_quat = obs.telescope.focalplane[det]["quat"]
                x, y, z = qa.rotate(det_quat, ZAXIS)
                theta, phi = np.arcsin([x, y])
                detector_templates[idet] = theta ** xorders * phi ** yorders

            # Iterate over each interval

            views = obs.view[self.view]
            for iview, view in enumerate(views):
                if view.start is None:
                    # This is a view of the whole obs
                    nsample = obs.n_local_samples
                else:
                    nsample = view.stop - view.start

                # Accumulate the linear regression templates

                templates = np.zeros([ndet, nmode, nsample])
                proj = np.zeros([nmode, nsample])

                t1 = time()

                norms = np.zeros(nmode)

                shared_flags = views.shared[self.shared_flags][iview]
                shared_mask = (shared_flags & self.shared_flag_mask) == 0

                for idet, det in enumerate(obs.local_detectors):
                    if det not in detector_index:
                        continue
                    ind = detector_index[det]

                    signal = views.detdata[self.det_data][iview][idet]
                    det_flags = views.detdata[self.det_flags][iview][idet]
                    det_mask = (det_flags & self.det_flag_mask) == 0

                    mask = np.logical_and(shared_mask, det_mask)

                    template = detector_templates[ind]
                    templates[idet] = np.outer(template, mask)
                    proj += np.outer(template, signal * mask)
                    norms += template ** 2

                t_template += time() - t1

                t1 = time()
                comm.allreduce(templates)
                comm.allreduce(proj)
                comm.allreduce(norms)
                good = norms != 0
                norms[good] = norms[good] ** -0.5
                t_get_norm += time() - t1

                # Noise-weight

                t1 = time()
                templates = np.transpose(
                    templates, [1, 0, 2]
                ).copy()  # nmode x ndet x nsample
                for mode, norm in enumerate(norms):
                    if norm:
                        templates[mode] *= norm
                        proj[mode] *= norm
                t_apply_norm += time() - t1

                # Solve the linear regression amplitudes.  Each task
                # inverts different template matrices

                t1 = time()
                templates = np.transpose(
                    templates, [2, 1, 0]
                ).copy()  # nsample x ndet x nmode
                proj = proj.T.copy()  # nsample x nmode
                coeff = np.zeros([nsample, nmode])
                for isample in range(nsample):
                    if isample % comm.size != comm.rank:
                        continue
                    templatesT = templates[isample].T.copy()  # ndet x nmode
                    ccinv = np.dot(templatesT, templates[isample])
                    try:
                        cc = np.linalg.inv(ccinv)
                        coeff[isample] = np.dot(cc, proj[isample])
                    except np.linalg.LinAlgError:
                        coeff[isample] = 0
                comm.allreduce(coeff)
                t_solve += time() - t1

                t1 = time()

                for isample in range(nsample):
                    if np.all(coeff[isample] == 0):
                        shared_flags[isample] |= self.poly_flag_mask

                templates = np.transpose(
                    templates, [1, 2, 0]
                ).copy()  # ndet x nmode x nsample
                coeff = coeff.T.copy()  # nmode x nsample

                for idet, det in enumerate(obs.local_detectors):
                    if det not in detector_index:
                        continue
                    ind = detector_index[det]
                    signal = views.detdata[self.det_data][iview][idet]
                    for mode in range(nmode):
                        signal -= coeff[mode] * templates[ind, mode]

                t_clean += time() - t1

            """
            print(
                "Time per observation: {:.1f} s\n"
                "   templates : {:6.1f} s\n"
                "    get_norm : {:6.1f} s\n"
                "  apply_norm : {:6.1f} s\n"
                "       solve : {:6.1f} s\n"
                "       clean : {:6.1f} s".format(
                    time() - t0, t_template, t_get_norm, t_apply_norm, t_solve, t_clean
                ),
                flush=True,
            )
            """

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.shared_flags],
            "detdata": [self.det_data, self.det_flags],
            "intervals": [self.view],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov

    def _accelerators(self):
        return list()


@trait_docs
class PolyFilter(Operator):
    """Operator which applies polynomial filtering to the TOD."""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode("signal", help="Observation detdata key apply the gain error to")

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are filtered.",
    )

    order = Int(1, allow_none=False, help="Polynomial order")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    poly_flag_mask = Int(0, help="Bit mask value for intervals that fail to filter")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.pattern is None:
            pat = None
        else:
            pat = re.compile(self.pattern)

        for obs in data.obs:
            # Get the detectors we are using for this observation
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            views = obs.view[self.view]
            for iview, view in enumerate(views):
                shared_flags = obs.shared[self.shared_flags].data & self.shared_flag_mask

                if self.view is not None:
                    local_starts = []
                    local_stops = []
                    for interval in obs.intervals[self.view]:
                        local_starts.append(interval.first)
                        local_stops.append(intervallast)
                else:
                    local_starts = [0]
                    local_stops = [obs.n_local_samples - 1]

                local_starts = np.array(local_starts)
                local_stops = np.array(local_stops)

                for idet, det in enumerate(dets):
                    # Test the detector pattern
                    if pat.match(det) is None:
                        continue

                    det_flags = obs.detdata[self.det_flags][idet] & self.det_flag_mask
                    signal = obs.detdata[self.det_data][idet]

                    flags = shared_flags | det_flags

                    filter_polynomial(self.order, flags, [signal], local_starts, local_stops)

                    obs.detdata[self.det_flags][idet][flags] & self.poly_flag_mask

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.shared_flags],
            "detdata": [self.det_data, self.det_flags],
            "intervals": [self.view],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov

    def _accelerators(self):
        return list()


class CommonModeFilter(Operator):
    """Operator to regress out common mode at each time stamp."""

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode("signal", help="Observation detdata key apply the gain error to")

    pattern = Unicode(
        f".*",
        allow_none=True,
        help="Regex pattern to match against detector names. Only detectors that "
        "match the pattern are filtered.",
    )

    order = Int(1, allow_none=False, help="Polynomial order")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    poly_flag_mask = Int(0, help="Bit mask value for intervals that fail to filter")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    focalplane_key = Unicode(None, allow_none=True, help="Which focalplane key to match")

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        """Apply the common mode filter to the signal.

        Args:
            data (toast.Data): The distributed data.

        """
        if detectors is not None:
            raise RuntimeError("CommonModeFilter cannot be run in batch mode")

        pat = re.compile(self.pattern)

        for obs in data.obs:
            focalplane = obs.telescope.focalplane
            # communicator for processes with the same sample range
            comm = obs.comm_col

            detectors = obs.all_detectors
            if self.focalplane_key is None:
                values = [None]
            else:
                values = set()
                for det in detectors:
                    values.add(focalplane[det][self.focalplane_key])
                values = sorted(values)

            nsample = obs.n_local_samples

            for value in values:
                local_dets = []
                for idet, det in enumerate(obs.local_detectors):
                    if pat.match(det) is None:
                        continue
                    if value is not None and \
                       focalplane[det][self.focalplane_key] != value:
                        continue
                    local_dets.append((idet, det))

                template = np.zeros(nsample)
                hits = np.zeros(nsample)
                shared_flags = obs.shared[self.shared_flags].data
                shared_mask = (shared_flags & self.shared_flag_mask) == 0
                for idet, det in local_dets:
                    signal = obs.detdata[self.det_data][idet]
                    det_flags = obs.detdata[self.det_flags][idet]
                    det_mask = (det_flags & self.det_flag_mask) == 0
                    mask = np.logical_and(shared_mask, det_mask)
                    template[mask] += signal[mask]
                    hits[mask] += 1

                if comm is not None:
                    comm.Barrier()
                    comm.Allreduce(MPI.IN_PLACE, template, op=MPI.SUM)
                    comm.Allreduce(MPI.IN_PLACE, hits, op=MPI.SUM)

                good = hits != 0
                template[good] /= hits[good]

                for idet, det in local_dets:
                    obs.detdata[self.det_data][idet] -= template
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.shared_flags],
            "detdata": [self.det_data, self.det_flags],
            "intervals": [self.view],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov

    def _accelerators(self):
        return list()
