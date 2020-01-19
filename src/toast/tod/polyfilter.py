# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re

import numpy as np

from .._libtoast import filter_polynomial

from ..op import Operator

from ..timing import function_timer


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
        for obs in data.obs:
            tod = obs["tod"]
            if self._intervals in obs:
                intervals = obs[self._intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)
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
