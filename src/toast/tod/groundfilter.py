# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from toast.mpi import MPI
from toast.op import Operator

import numpy as np
from numpy.polynomial.chebyshev import chebval
import toast.timing as timing


class OpGroundFilter(Operator):
    """
    Operator which applies ground template filtering to constant
    elevation scans.

    Args:
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
        ground_flag_mask (byte):  Bitmask to use when adding flags based
           on ground filter failures.
        trend_order (int):  Order of a Chebyshev polynomial to fit along
            with the ground template
        filter_order (int):  Order of a Chebyshev polynomial to fit as a
            function of azimuth
        detrend (bool):  Subtract the fitted trend along with the
             ground template
        intervals (str):  Name of the valid intervals in observation
        split_template (bool):  Apply a different template for left and
             right scans
    """

    def __init__(
        self,
        name=None,
        common_flag_name=None,
        common_flag_mask=255,
        flag_name=None,
        flag_mask=255,
        ground_flag_mask=1,
        trend_order=5,
        filter_order=5,
        detrend=False,
        intervals="intervals",
        split_template=False,
    ):

        self._name = name
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._ground_flag_mask = ground_flag_mask
        self._trend_order = trend_order
        self._filter_order = filter_order
        self._detrend = detrend
        self._intervals = intervals
        self._split_template = split_template

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    def fit_templates(self, tod, det, templates, ref, good):

        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        proj = np.zeros(ntemplate)

        # Bin the local data

        if det in tod.local_dets:
            for i in range(ntemplate):
                proj[i] = np.sum((templates[i] * ref)[good])
                for j in range(i, ntemplate):
                    invcov[i, j] = np.sum((templates[i] * templates[j])[good])
                    # Symmetrize invcov
                    if i != j:
                        invcov[j, i] = invcov[i, j]

        if tod._sampranks > 1:
            # Reduce the binned data.  The detector signals is
            # distributed across the group communicator.
            cgroup.Allreduce(MPI.IN_PLACE, invcov, op=MPI.SUM)
            cgroup.Allreduce(MPI.IN_PLACE, proj, op=MPI.SUM)

        # Assemble the joint template

        if det in tod.local_dets:
            try:
                cov = np.linalg.inv(invcov)
                coeff = np.dot(cov, proj)
            except np.linalg.LinAlgError as e:
                print('linalg.inv failed: "{}"'.format(e), flush=True)
                # np.linalg.lstsq will find a least squares minimum
                # even if the covariance matrix is not invertible
                coeff = np.linalg.lstsq(invcov, proj, rcond=1e-30)[0]
                if np.any(np.isnan(coeff)) or np.std(coeff) < 1e-30:
                    raise RuntimeError("lstsq FAILED")
        else:
            coeff = None

        return coeff

    def exec(self, data):
        """
        Apply the ground filter to the signal.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        # the two-level pytoast communicator
        comm = data.comm
        # the communicator within the group
        cgroup = comm.comm_group

        # Each group loops over its own CES:es

        for obs in data.obs:
            tod = obs["tod"]
            nsamp_tot = tod.total_samples
            my_offset, my_nsamp = tod.local_samples
            if self._intervals in obs:
                intervals = obs[self._intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)

            # Construct trend templates.  Full domain for x is [-1, 1]

            x = np.arange(my_offset, my_offset + my_nsamp) / nsamp_tot * 2 - 1
            ntrend = self._trend_order
            # Do not include the offset in the trend.  It will be part of
            # of the ground template
            cheby_trend = chebval(x, np.eye(ntrend + 1), tensor=True)[1:]

            try:
                (azmin, azmax, _, _) = tod.scan_range
                az = tod.read_boresight_az()
            except Exception as e:
                raise RuntimeError(
                    "Failed to get boresight azimuth from TOD.  Perhaps it is "
                    'not ground TOD? "{}"'.format(e)
                )

            # Cache the output common flags
            common_ref = tod.local_common_flags(self._common_flag_name)

            # The azimuth vector is assumed to be arranged so that the
            # azimuth increases monotonously even across the zero meridian.

            phase = (az - azmin) / (azmax - azmin) * 2 - 1
            nfilter = self._filter_order + 1
            cheby_templates = chebval(phase, np.eye(nfilter), tensor=True)
            if not self._split_template:
                cheby_filter = cheby_templates
            else:
                # Create separate templates for alternating scans
                cheby_filter = []
                mask1 = common_ref != 0
                mask2 = mask1.copy()
                for i, ival in enumerate(local_intervals):
                    mask = [mask1, mask2][i % 2]
                    mask[ival.first : ival.last + 1] = True
                for template in cheby_templates:
                    for mask in mask1, mask2:
                        temp = template.copy()
                        temp[mask] = 0
                        cheby_filter.append(temp)

            templates = []
            for temp in cheby_trend, cheby_filter:
                for template in temp:
                    templates.append(template)

            for det in tod.detectors:
                if det in tod.local_dets:
                    ref = tod.local_signal(det, self._name)
                    flag_ref = tod.local_flags(det, self._flag_name)
                    good = np.logical_and(
                        common_ref & self._common_flag_mask == 0,
                        flag_ref & self._flag_mask == 0,
                    )
                    del flag_ref
                else:
                    ref = None
                    good = None

                coeff = self.fit_templates(tod, det, templates, ref, good)

                if det in tod.local_dets:
                    # Trend
                    trend = np.zeros_like(ref)
                    for cc, template in zip(coeff[:ntrend], cheby_trend):
                        trend += cc * template
                    if self._detrend:
                        ref[good] -= trend[good]
                    # Ground template
                    grtemplate = np.zeros_like(ref)
                    for cc, template in zip(coeff[ntrend:], cheby_filter):
                        grtemplate += cc * template
                    ref[good] -= grtemplate[good]
                    ref[np.logical_not(good)] = 0
                    del ref

            del common_ref

        return
