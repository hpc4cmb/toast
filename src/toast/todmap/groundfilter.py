# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import numpy as np

from numpy.polynomial.chebyshev import chebval

from .._libtoast import bin_templates, add_templates, chebyshev

from ..op import Operator

from ..utils import Logger

from ..timing import function_timer


class OpGroundFilter(Operator):
    """Operator which applies ground template filtering to constant
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

        # Call the parent class constructor.
        super().__init__()

    @function_timer
    def build_templates(self, tod, obs):
        """Construct the local ground template hierarchy"""
        if self._intervals in obs:
            intervals = obs[self._intervals]
        else:
            intervals = None
        local_intervals = tod.local_intervals(intervals)

        # Construct trend templates.  Full domain for x is [-1, 1]

        my_offset, my_nsamp = tod.local_samples
        nsamp_tot = tod.total_samples
        x = np.arange(my_offset, my_offset + my_nsamp) / nsamp_tot * 2 - 1

        # Do not include the offset in the trend.  It will be part of
        # of the ground template
        # cheby_trend = chebval(x, np.eye(self._trend_order + 1), tensor=True)[1:]
        cheby_trend = np.zeros([self._trend_order, x.size])
        chebyshev(x, cheby_trend, 1, self._trend_order + 1)

        try:
            (azmin, azmax, _, _) = tod.scan_range
            az = tod.read_boresight_az()
        except Exception as e:
            raise RuntimeError(
                "Failed to get boresight azimuth from TOD.  Perhaps it is "
                'not ground TOD? "{}"'.format(e)
            )

        # The azimuth vector is assumed to be arranged so that the
        # azimuth increases monotonously even across the zero meridian.

        phase = (az - azmin) / (azmax - azmin) * 2 - 1
        nfilter = self._filter_order + 1
        # cheby_templates = chebval(phase, np.eye(nfilter), tensor=True)
        cheby_templates = np.zeros([nfilter, phase.size])
        chebyshev(phase, cheby_templates, 0, nfilter)
        if not self._split_template:
            cheby_filter = cheby_templates
        else:
            # Create separate templates for alternating scans
            common_ref = tod.local_common_flags(self._common_flag_name)
            cheby_filter = []
            mask1 = common_ref & tod.LEFTRIGHT_SCAN == 0
            mask2 = common_ref & tod.RIGHTLEFT_SCAN == 0
            for template in cheby_templates:
                for mask in mask1, mask2:
                    temp = template.copy()
                    temp[mask] = 0
                    cheby_filter.append(temp)
            del common_ref
            cheby_filter = np.vstack(cheby_filter)

        templates = np.vstack([cheby_trend, cheby_filter])

        return templates, cheby_trend, cheby_filter

    @function_timer
    def bin_templates(self, ref, templates, good, invcov, proj):

        # Bin the local data

        for i in range(ntemplate):
            temp = templates[i] * good
            proj[i] = np.dot(temp, ref)
            for j in range(i, ntemplate):
                invcov[i, j] = np.dot(temp, templates[j])
                # Symmetrize invcov
                if i != j:
                    invcov[j, i] = invcov[i, j]
            del temp

        return

    @function_timer
    def fit_templates(self, tod, det, templates, ref, good):
        log = Logger.get()
        comm = tod.mpicomm
        rank = 0
        if comm is not None:
            rank = comm.rank
        detranks, sampranks = tod.grid_size

        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        proj = np.zeros(ntemplate)
        if ref is not None:
            # self.bin_templates(ref, templates, good, invcov, proj)
            bin_templates(ref, templates, good.astype(np.uint8), invcov, proj)

        if sampranks > 1:
            # Reduce the binned data.  The detector signals is
            # distributed across the group communicator.
            comm.Allreduce(MPI.IN_PLACE, invcov, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, proj, op=MPI.SUM)

        # Assemble the joint template
        if ref is not None:
            try:
                cov = np.linalg.inv(invcov)
                coeff = np.dot(cov, proj)
            except np.linalg.LinAlgError as e:
                msg = 'linalg.inv failed with "{}"'.format(e)
                log.warning(msg)
                # np.linalg.lstsq will find a least squares minimum
                # even if the covariance matrix is not invertible
                coeff = np.linalg.lstsq(invcov, proj, rcond=1e-30)[0]
                if np.any(np.isnan(coeff)) or np.std(coeff) < 1e-30:
                    raise RuntimeError("lstsq FAILED")
        else:
            coeff = None

        return coeff

    @function_timer
    def subtract_templates(self, ref, good, coeff, cheby_trend, cheby_filter):
        # Trend
        if self._detrend:
            trend = np.zeros_like(ref)
            # for cc, template in zip(coeff[: self._trend_order], cheby_trend):
            #    trend += cc * template
            add_templates(trend, cheby_trend, coeff[: self._trend_order])
            ref[good] -= trend[good]
        # Ground template
        grtemplate = np.zeros_like(ref)
        # for cc, template in zip(coeff[self._trend_order :], cheby_filter):
        #    grtemplate += cc * template
        add_templates(grtemplate, cheby_filter, coeff[self._trend_order :])
        ref[good] -= grtemplate[good]
        ref[np.logical_not(good)] = 0
        return

    @function_timer
    def exec(self, data):
        """Apply the ground filter to the signal.

        Args:
            data (toast.Data): The distributed data.

        """
        # Each group loops over its own CES:es
        for obs in data.obs:
            tod = obs["tod"]

            # Cache the output common flags
            common_ref = tod.local_common_flags(self._common_flag_name)

            templates, cheby_trend, cheby_filter = self.build_templates(tod, obs)

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
                if ref is not None:
                    self.subtract_templates(ref, good, coeff, cheby_trend, cheby_filter)

                del ref

            del common_ref

        return
