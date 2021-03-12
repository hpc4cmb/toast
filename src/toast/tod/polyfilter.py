
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import re

from time import time

import healpy as hp
import numpy as np

from .._libtoast import filter_polynomial

from ..op import Operator

from ..timing import function_timer

from .. import qarray as qa


XAXIS, YAXIS, ZAXIS = np.eye(3)


class OpPolyFilter2D(Operator):
    """Operator to regress out 2D polynomials across the focal plane.

    Args:
    order (int):  Order of the filtering polynomial.
    pattern (str):  Regex pattern to match against detector names.
        Only detectors that match the pattern are filtered.
    name (str):  Name of the output signal cache object will be
        <name_in>_<detector>.  If the object exists, it is used as
        input.  Otherwise signal is read using the tod read method.
    common_flag_name (str):  Cache name of the output common flags.
        If it already exists, it is used.  Otherwise flags
        are read from the tod object and stored in the cache under
        common_flag_name.
    common_flag_mask (byte):  Bitmask to use when flagging data
       based on the common flags.
    flag_name (str):  Cache name of the output detector flags will
        be <flag_name>_<detector>.  If the object exists, it is
        used.  Otherwise flags are read from the tod object.
    flag_mask (byte):  Bitmask to use when flagging data
       based on the detector flags.
    poly_flag_mask (byte):  Bitmask to use when adding flags based
       on polynomial filter failures.
    intervals (str):  Name of the valid intervals in observation.
    buffer_length ((int): Number of samples to filter at a time.
        Default is usually fine.
    """

    def __init__(
        self,
        order=1,
        pattern=r".*",
        name=None,
        common_flag_name=None,
        common_flag_mask=255,
        flag_name=None,
        flag_mask=255,
        poly_flag_mask=1,
        intervals="intervals",
        buffer_length=1000,
        verbose=True,
    ):
        self._order = order
        self._nmode = (order + 1) * (order + 2) // 2
        self._pattern = pattern
        self._name = name
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._poly_flag_mask = poly_flag_mask
        self._intervals = intervals
        self._buffer_length = buffer_length
        self._verbose = verbose

        # Call the parent class constructor.
        super().__init__()

    @function_timer
    def exec(self, data):
        """Apply the 2D polynomial filter to the signal.

        Args:
            data (toast.Data): The distributed data.

        """
        norder = self._order + 1
        nmode = self._nmode
        world_comm = data.comm.comm_world
        if world_comm is None:
            world_rank = 0
        else:
            world_rank = world_comm.rank

        for obs in data.obs:
            t0 = time()
            t_template = 0
            t_get_norm = 0
            t_apply_norm = 0
            t_solve = 0
            t_clean = 0

            tod = obs["tod"]
            times = tod.local_times()
            # communicator for processes with the same sample range
            comm = tod.grid_comm_col
            if comm is None:
                comm_rank = 0
                comm_size = 1
            else:
                comm_rank = comm.rank
                comm_size = comm.size
            detectors = tod.detectors
            ndet = len(detectors)
            detector_index = {}
            pat = re.compile(self._pattern)

            ndet = 0
            for det in detectors:
                if pat.match(det) is None:
                    continue
                detector_index[det] = ndet
                ndet += 1
            # Number of detectors may limit the number of modes we can constrain
            nmode = min(self._nmode, ndet)

            focalplane = obs["focalplane"]

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

            yrot = qa.rotation(YAXIS, np.pi / 2)
            for det in tod.local_dets:
                if det not in detector_index:
                    continue
                idet = detector_index[det]
                det_quat = focalplane[det]["quat"]
                vec = qa.rotate(det_quat, ZAXIS)
                theta, phi = hp.vec2dir(qa.rotate(yrot, vec))
                theta -= np.pi / 2
                detector_templates[idet] = theta ** xorders * phi ** yorders

            if self._intervals in obs:
                intervals = obs[self._intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)
            if len(local_intervals) == 0:
                # No intervals to filter
                continue
            common_ref = tod.local_common_flags(self._common_flag_name)

            # Iterate over each interval

            for ival in local_intervals:
                istart = ival.first
                while istart < ival.last + 1:
                    istop = min(istart + self._buffer_length, ival.last + 1)
                    ind = slice(istart, istop)
                    nsample = istop - istart

                    templates = np.zeros([ndet, nmode])
                    masks = np.zeros([ndet, nsample], dtype=np.bool)
                    proj = np.zeros([nmode, nsample])
                    norms = np.zeros(nmode)

                    t1 = time()

                    for det in tod.local_dets:
                        if det not in detector_index:
                            continue
                        idet = detector_index[det]

                        ref = tod.local_signal(det, self._name)[ind]
                        flag_ref = tod.local_flags(det, self._flag_name)[ind]

                        flg = common_ref[ind] & self._common_flag_mask
                        flg |= flag_ref & self._flag_mask
                        mask = flg == 0

                        # We might want to remove the interval mean if the
                        # data were not already 1D-filtered
                        # ref -= np.mean(ref[mask])

                        template = detector_templates[idet]
                        templates[idet] = template
                        masks[idet] = mask
                        proj += np.outer(template, ref * mask)
                        norms += template ** 2

                        del ref
                        del flag_ref

                    t_template += time() - t1

                    t1 = time()
                    if comm is not None:
                        comm.Barrier()
                        comm.Allreduce(MPI.IN_PLACE, templates, op=MPI.SUM)
                        comm.Allreduce(MPI.IN_PLACE, masks, op=MPI.LOR)
                        comm.Allreduce(MPI.IN_PLACE, proj, op=MPI.SUM)
                        comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)

                    good = norms != 0
                    norms[good] = norms[good] ** -0.5
                    t_get_norm += time() - t1

                    t1 = time()
                    templates = templates.T.copy()  # nmode x ndet
                    for mode, norm in enumerate(norms):
                        if norm:
                            templates[mode] *= norm
                            proj[mode] *= norm
                    t_apply_norm += time() - t1

                    t1 = time()
                    proj = proj.T.copy()  # nsample x nmode
                    coeff = np.zeros([nsample, nmode])
                    masks = np.transpose(masks)  # nsample x ndet
                    for isample in range(nsample):
                        if isample % comm_size != comm_rank:
                            continue
                        mask = masks[isample]
                        templatesT = mask * templates  # nmode * ndet
                        invcov = np.dot(templatesT, templates.T)
                        try:
                            cov = np.linalg.inv(invcov)
                            coeff[isample] = np.dot(cov, proj[isample])
                        except np.linalg.LinAlgError:
                            coeff[isample] = 0
                    if comm is not None:
                        comm.Allreduce(MPI.IN_PLACE, coeff, op=MPI.SUM)
                    t_solve += time() - t1

                    t1 = time()

                    for isample in range(nsample):
                        if np.all(coeff[isample] == 0):
                            common_ref[istart + isample] |= self._poly_flag_mask

                    templates = templates.T.copy()  # ndet x nmode x nsample
                    coeff = coeff.T.copy()  # nmode x nsample

                    for det in tod.local_dets:
                        if det not in detector_index:
                            continue
                        idet = detector_index[det]
                        ref = tod.local_signal(det, self._name)[ind]
                        for mode in range(nmode):
                            ref -= coeff[mode] * templates[idet, mode]

                    t_clean += time() - t1

                    del templates
                    istart = istop

            del common_ref

            if world_rank == 0 or (comm_rank == 0 and self._verbose > 1):
                print(
                    "Time per observation: {:.1f} s\n"
                    "   templates : {:6.1f} s\n"
                    "    get_norm : {:6.1f} s\n"
                    "  apply_norm : {:6.1f} s\n"
                    "       solve : {:6.1f} s\n"
                    "       clean : {:6.1f} s".format(
                        time() - t0,
                        t_template,
                        t_get_norm,
                        t_apply_norm,
                        t_solve,
                        t_clean,
                    ),
                    flush=True,
                )

        return


class OpPolyFilter(Operator):
    """Operator which applies polynomial filtering to the TOD.

    This applies polynomial filtering to the valid intervals of each TOD.

    Args:
        order (int):  Order of the filtering polynomial.
        pattern (str):  Regex pattern to match against detector names.
            Only detectors that match the pattern are filtered.
        name (str):  Name of the output signal cache object will be
            <name_in>_<detector>.  If the object exists, it is used as
            input.  Otherwise signal is read using the tod read method.
        common_flag_name (str):  Cache name of the output common flags.
            If it already exists, it is used.  Otherwise flags
            are read from the tod object and stored in the cache under
            common_flag_name.
        common_flag_mask (byte):  Bitmask to use when flagging data
           based on the common flags.
        flag_name (str):  Cache name of the output detector flags will
            be <flag_name>_<detector>.  If the object exists, it is
            used.  Otherwise flags are read from the tod object.
        flag_mask (byte):  Bitmask to use when flagging data
           based on the detector flags.
        poly_flag_mask (byte):  Bitmask to use when adding flags based
           on polynomial filter failures.
        intervals (str):  Name of the valid intervals in observation.
    """

    def __init__(
        self,
        order=1,
        pattern=r".*",
        name=None,
        common_flag_name=None,
        common_flag_mask=255,
        flag_name=None,
        flag_mask=255,
        poly_flag_mask=1,
        intervals="intervals",
    ):
        self._order = order
        self._pattern = pattern
        self._name = name
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._poly_flag_mask = poly_flag_mask
        self._intervals = intervals

        # Call the parent class constructor.
        super().__init__()

    @function_timer
    def exec(self, data):
        """Apply the polynomial filter to the signal.

        Args:
            data (toast.Data): The distributed data.

        """
        world_comm = data.comm.comm_world
        if world_comm is None:
            world_rank = 0
        else:
            world_rank = world_comm.rank

        for obs in data.obs:
            tod = obs["tod"]
            if self._intervals in obs:
                intervals = obs[self._intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)
            if len(local_intervals) == 0:
                # No intervals to filter
                continue
            common_ref = tod.local_common_flags(self._common_flag_name)

            pat = re.compile(self._pattern)

            for det in tod.local_dets:
                # Test the detector pattern
                if pat.match(det) is None:
                    continue

                ref = tod.local_signal(det, self._name)
                flag_ref = tod.local_flags(det, self._flag_name)

                # Iterate over each interval

                local_starts = []
                local_stops = []
                for ival in local_intervals:
                    local_starts.append(ival.first)
                    local_stops.append(ival.last)

                local_starts = np.array(local_starts)
                local_stops = np.array(local_stops)

                flg = common_ref & self._common_flag_mask
                flg |= flag_ref & self._flag_mask

                filter_polynomial(self._order, flg, [ref], local_starts, local_stops)

                flag_ref[flg != 0] |= self._poly_flag_mask

                del ref
                del flag_ref

            del common_ref

        return
