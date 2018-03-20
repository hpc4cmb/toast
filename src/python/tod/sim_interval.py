# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from ..dist import distribute_uniform
from .. import timing as timing

from .interval import Interval



def regular_intervals(n, start, first, rate, duration, gap):
    """
    Function to generate regular intervals.

    This creates a list of intervals, given a start time/sample and time
    span for the interval and the gap in time between intervals.  The 
    length of the interval and the gap are rounded to the nearest sample
    and all intervals in the list are created using those lengths.

    Optionally, the full data span can be subdivided into some number
    of contiguous subchunks.

    Args:
        n (int): the number of intervals.
        start (float): the start time in seconds.
        first (int): the first sample index, which occurs at "start".
        rate (float): the sample rate in Hz.
        duration (float): the length of the interval in seconds.
        gap (float): the length of the gap in seconds.

    Returns:
        (list): a list of Interval objects.
    """
    autotimer = timing.auto_timer()
    invrate = 1.0 / rate

    # Compute the whole number of samples that fit within the
    # requested time span (rounded to nearest sample).
    totsamples = int(0.5 + (duration + gap) * rate)
    dursamples = int(0.5 + duration * rate)

    intervals = []

    for i in range(n):
        ifirst = first + i * totsamples
        ilast = ifirst + dursamples - 1
        # The time span between interval starts (the first sample of one interval
        # to the first sample of the next) includes the one extra sample time.
        istart = start + i * (totsamples * invrate)
        # The stop time is the timestamp of the last valid sample (thus the -1).
        istop = istart + (dursamples - 1) * invrate
        intervals.append(Interval(start=istart, stop=istop, first=ifirst, last=ilast))

    return intervals



