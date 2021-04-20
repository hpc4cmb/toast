# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

from ..mpi import MPI

import numpy as np

from .._libtoast import bin_templates, add_templates, legendre

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
        trend_order (int):  Order of a Legendre polynomial to fit along
            with the ground template
        filter_order (int):  Order of a Legendre polynomial to fit as a
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
        verbose=True,
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
        self.verbose = verbose

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
        legendre_trend = np.zeros([self._trend_order, x.size])
        legendre(x, legendre_trend, 1, self._trend_order + 1)

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
        legendre_templates = np.zeros([nfilter, phase.size])
        legendre(phase, legendre_templates, 0, nfilter)
        if not self._split_template:
            legendre_filter = legendre_templates
        else:
            # Create separate templates for alternating scans
            common_ref = tod.local_common_flags(self._common_flag_name)
            legendre_filter = []
            mask1 = common_ref & tod.LEFTRIGHT_SCAN == 0
            mask2 = common_ref & tod.RIGHTLEFT_SCAN == 0
            for template in legendre_templates:
                for mask in mask1, mask2:
                    temp = template.copy()
                    temp[mask] = 0
                    legendre_filter.append(temp)
            del common_ref
            legendre_filter = np.vstack(legendre_filter)

        templates = np.vstack([legendre_trend, legendre_filter])

        return templates, legendre_trend, legendre_filter

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
        # communicator for processes with the same detectors
        comm = tod.grid_comm_row
        ngood = np.sum(good)
        if comm is not None:
            ngood = comm.allreduce(ngood)
        if ngood == 0:
            return None

        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        proj = np.zeros(ntemplate)

        bin_templates(ref, templates, good.astype(np.uint8), invcov, proj)

        if comm is not None:
            # Reduce the binned data.  The detector signals is
            # distributed across the group communicator.
            comm.Allreduce(MPI.IN_PLACE, invcov, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, proj, op=MPI.SUM)

        # Assemble the joint template
        rcond = 1 / np.linalg.cond(invcov)
        self.rcondsum += rcond
        if rcond > 1e-6:
            self.ngood += 1
            cov = np.linalg.inv(invcov)
            coeff = np.dot(cov, proj)
        else:
            self.nsingular += 1
            log.debug(
                f"Ground template matrix is poorly conditioned, "
                f"rcond = {rcond}, doing least squares fitting."
            )
            # np.linalg.lstsq will find a least squares minimum
            # even if the covariance matrix is not invertible
            coeff = np.linalg.lstsq(invcov, proj, rcond=1e-30)[0]
            if np.any(np.isnan(coeff)) or np.std(coeff) < 1e-30:
                raise RuntimeError("lstsq FAILED")

        return coeff

    @function_timer
    def subtract_templates(self, ref, good, coeff, legendre_trend, legendre_filter):
        # Trend
        if self._detrend:
            trend = np.zeros_like(ref)
            add_templates(trend, legendre_trend, coeff[: self._trend_order])
            ref[good] -= trend[good]
        # Ground template
        grtemplate = np.zeros_like(ref)
        add_templates(grtemplate, legendre_filter, coeff[self._trend_order :])
        ref[good] -= grtemplate[good]
        ref[np.logical_not(good)] = 0
        return

    @function_timer
    def exec(self, data):
        """Apply the ground filter to the signal.

        Args:
            data (toast.Data): The distributed data.

        """

        t0 = time()

        self.comm = data.comm.comm_world
        if self.comm is None:
            self.rank = 0
            self.ntask = 1
        else:
            self.rank = self.comm.rank
            self.ntask = self.comm.size
        gcomm = data.comm.comm_group
        self.group = data.comm.group
        if gcomm is None:
            self.grank = 0
        else:
            self.grank = gcomm.rank

        self.nsingular = 0
        self.ngood = 0
        self.rcondsum = 0

        # Each group loops over its own CES:es
        for iobs, obs in enumerate(data.obs):
            tod = obs["tod"]
            if (self.rank == 0 and self.verbose) or (
                self.grank == 0 and self.verbose > 1
            ):
                print(
                    "{:4} : OpGroundFilter: Processing observation {} / {}".format(
                        self.group, iobs + 1, len(data.obs)
                    ),
                    flush=True,
                )

            # Cache the output common flags
            common_ref = tod.local_common_flags(self._common_flag_name)

            t1 = time()
            templates, legendre_trend, legendre_filter = self.build_templates(tod, obs)
            if self.grank == 0 and self.verbose > 1:
                print(
                    "{:4} : OpGroundFilter: Built templates in {:.1f}s".format(
                        self.group, time() - t1
                    ),
                    flush=True,
                )

            for idet, det in enumerate(tod.local_dets):
                if self.grank == 0 and self.verbose > 1:
                    print(
                        "{:4} : OpGroundFilter:   Processing detector # {} / {}".format(
                            self.group, idet + 1, len(tod.local_dets)
                        ),
                        flush=True,
                    )
                ref = tod.local_signal(det, self._name)
                flag_ref = tod.local_flags(det, self._flag_name)
                good = np.logical_and(
                    common_ref & self._common_flag_mask == 0,
                    flag_ref & self._flag_mask == 0,
                )
                del flag_ref

                t1 = time()
                coeff = self.fit_templates(tod, det, templates, ref, good)
                if self.grank == 0 and self.verbose > 1:
                    print(
                        "{:4} : OpGroundFilter: Fit templates in {:.1f}s".format(
                            self.group, time() - t1
                        ),
                        flush=True,
                    )

                if coeff is None:
                    continue

                t1 = time()
                self.subtract_templates(
                    ref, good, coeff, legendre_trend, legendre_filter
                )
                if self.grank == 0 and self.verbose > 1:
                    print(
                        "{:4} : OpGroundFilter: Subtract templates in {:.1f}s".format(
                            self.group, time() - t1
                        ),
                        flush=True,
                    )

                del ref

            del common_ref

        if self.comm is not None:
            self.nsingular = self.comm.allreduce(self.nsingular)
            self.ngood = self.comm.allreduce(self.ngood)
            self.rcondsum = self.comm.allreduce(self.rcondsum)
        if self.rank == 0:
            print(
                "Applied ground filter in {:.1f} s.  Average rcond of template matrix was {}".format(
                    time() - t0, self.rcondsum / (self.nsingular + self.ngood)
                ),
                flush=True,
            )

        return
