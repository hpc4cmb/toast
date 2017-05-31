# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import re

import numpy as np

from ..op import Operator
from ..dist import Comm, Data
from .tod import TOD


class OpPolyFilter(Operator):
    """
    Operator which applies polynomial filtering to the valid intervals
    of the TOD.

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
        flag_name (str):  Cache Name of the output detector flags will
            be <flag_name>_<detector>.  If the object exists, it is
            used.  Otherwise flags are read from the tod object.
        flag_mask (byte):  Bitmask to use when flagging data
           based on the detector flags.
        poly_flag_mask (byte):  Bitmask to use when adding flags based
           on polynomial filter failures.
    """

    def __init__(self, order=1, pattern=r'.*', name='signal',
                 common_flag_name='common_flags', common_flag_mask=255,
                 flag_name='flags', flag_mask=255, poly_flag_mask=1):

        self._order = order
        self._pattern = pattern
        self._name = name
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._poly_flag_mask = poly_flag_mask

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    def exec(self, data):
        """
        Apply the polynomial filter to the signal.

        Args:
            data (toast.Data): The distributed data.
        """
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        for obs in data.obs:
            tod = obs['tod']
            tod_first = tod.local_samples[0]
            tod_nsamp = tod.local_samples[1]

            # get the total list of intervals
            intervals = None
            if 'intervals' in obs.keys():
                intervals = obs['intervals']
            if intervals is None:
                intervals = [Interval(start=0.0, stop=0.0, first=0,
                                      last=(tod.total_samples-1))]

             # Cache the output common flags
            cachename = self._common_flag_name
            if tod.cache.exists(cachename):
                common_ref = tod.cache.reference(cachename)
            else:
                common_flag = tod.read_common_flags()
                common_ref = tod.cache.put(cachename, common_flag)
                del common_flag

            pat = re.compile(self._pattern)

            for det in tod.local_dets:
                # Test the detector pattern

                if not pat.match(det):
                    continue

                # Cache the output signal
                cachename = '{}_{}'.format(self._name, det)
                if tod.cache.exists(cachename):
                    ref = tod.cache.reference(cachename)
                else:
                    signal = tod.read(detector=det)
                    ref = tod.cache.put(cachename, signal)

                # Cache the output flags
                cachename = '{}_{}'.format(self._flag_name, det)
                if tod.cache.exists(cachename):
                    flag_ref = tod.cache.reference(cachename)
                else:
                    # read_flags always returns both common and detector
                    # flags but we already cached the common flags.
                    flag, dummy = tod.read_flags(detector=det)
                    flag_ref = tod.cache.put(cachename, flag)
                    del flag, dummy

                # Iterate over each interval

                for ival in intervals:
                    if ival.last >= tod_first and \
                       ival.first < tod_first + tod_nsamp:
                        # This interval overlaps with this tod.
                        # Apply the filter to the overlap.
                        local_start = ival.first - tod_first
                        local_stop = ival.last - tod_first
                        if local_start < 0:
                            local_start = 0
                        if local_stop > tod_nsamp:
                            local_stop = tod_nsamp
                        ind = slice(local_start, local_stop)
                        n = local_stop - local_start
                        sig = ref[ind]
                        good = np.logical_and(
                            common_ref[ind] & self._common_flag_mask == 0,
                            flag_ref[ind] & self._flag_mask == 0)
                        # The X-coordinate should be symmetric
                        x = np.arange(n) * 10 / n - 5
                        try:
                            # Subtracting the mean improves the
                            # condition number
                            sig -= np.mean(sig[good])
                            # Fit and subtract the polynomial
                            p = np.polyfit(x[good], sig[good], self._order)
                            sig -= np.polyval(p, x)
                        except:
                            # Polynomial fitting failed, flag the
                            # failed region.
                            flag_ref[ind] |= self._poly_flag_mask

                del ref
                del flag_ref

            del common_ref

        return

