# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from ..op import Operator
from .. import timing as timing


class Interval(object):
    """
    Class storing a time and sample range.

    Args:
        start (float): The start time of the interval in seconds.
        stop (float): The stop time of the interval in seconds.
        first (int): The first sample index of the interval.
        last (int): The last sample index (inclusive) of the interval.
    """
    def __init__(self, start=None, stop=None, first=None, last=None):
        self._start = start
        self._stop = stop
        self._first = first
        self._last = last

    def __repr__(self):
        return '<Interval {} - {} ({} - {})>'.format(self._start, self._stop, self._first, self._last)

    @property
    def start(self):
        """
        (float): the start time of the interval.
        """
        if self._start is None:
            raise RuntimeError("Start time is not yet assigned")
        return self._start

    @start.setter
    def start(self, val):
        if val < 0.0:
            raise ValueError("Negative start time is not valid")
        self._start = val

    @property
    def stop(self):
        """
        (float): the start time of the interval.
        """
        if self._stop is None:
            raise RuntimeError("Stop time is not yet assigned")
        return self._stop

    @stop.setter
    def stop(self, val):
        if val < 0.0:
            raise ValueError("Negative stop time is not valid")
        self._stop = val

    @property
    def first(self):
        """
        (int): the first sample of the interval.
        """
        if self._first is None:
            raise RuntimeError("First sample is not yet assigned")
        return self._first

    @first.setter
    def first(self, val):
        if val < 0:
            raise ValueError("Negative first sample is not valid")
        self._first = val

    @property
    def last(self):
        """
        (int): the first sample of the interval.
        """
        if self._last is None:
            raise RuntimeError("Last sample is not yet assigned")
        return self._last

    @last.setter
    def last(self, val):
        if val < 0:
            raise ValueError("Negative last sample is not valid")
        self._last = val

    @property
    def range(self):
        """
        (float): the number seconds in the interval.
        """
        b = self.start
        e = self.stop
        return (e - b)

    @property
    def samples(self):
        """
        (int): the number samples in the interval.
        """
        b = self.first
        e = self.last
        return (e - b + 1)
    

class OpFlagGaps(Operator):
    """
    Operator which applies common flags to gaps between valid intervals.

    Args:
        common_flag_name (str): the name of the cache object 
            to use for the common flags.  If None, use the TOD.
        common_flag_value (int): the integer bit mask (0-255) 
            that should be bitwise ORed with the existing flags.
    """

    def __init__(self, common_flag_name=None, common_flag_value=1):
        self._common_flag_name = common_flag_name
        self._common_flag_value = common_flag_value
        super().__init__()

    def exec(self, data):
        """
        Flag samples between valid intervals.

        This iterates over all observations and flags samples
        which lie outside the list of intervals.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timing.auto_timer(type(self).__name__)
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
            intrvls = obs['intervals']

            if intrvls is None:
                continue

            local_offset = tod.local_samples[0]
            local_nsamp = tod.local_samples[1]

            # first, flag all samples
            gapflags = np.zeros(local_nsamp, dtype=np.uint8)
            gapflags.fill(self._common_flag_value)

            # now un-flag samples in valid intervals
            for ival in intrvls:
                if (ival.last >= local_offset
                    and ival.first < (local_offset + local_nsamp)):
                    local_start = ival.first - local_offset
                    local_stop = ival.last - local_offset
                    if local_start < 0:
                        local_start = 0
                    if local_stop > local_nsamp - 1:
                        local_stop = local_nsamp - 1
                    gapflags[local_start:local_stop+1] = 0

            if self._common_flag_name is None:
                # set TOD common flags
                flags = tod.read_common_flags(local_start=0, n=local_nsamp)
                flags |= gapflags
                tod.write_common_flags(local_start=0, flags=flags)
            else:
                # use the cache
                if not tod.cache.exists(self._common_flag_name):
                    tod.cache.create(self._common_flag_name, np.uint8, (local_nsamp,))
                comref = tod.cache.reference(self._common_flag_name)
                comref[:] |= gapflags

        return



