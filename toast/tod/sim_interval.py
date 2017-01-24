# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np

from ..dist import distribute_uniform

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
    invrate = 1.0 / rate

    # Compute the whole number of samples that fit within the
    # requested time span (rounded to nearest sample).
    totsamples = int(0.5 + (duration + gap) * rate) + 1
    dursamples = int(0.5 + duration * rate) + 1
    gapsamples = totsamples - dursamples

    # Compute the actual time span for this number of samples
    tottime = (totsamples - 1) * invrate
    durtime = (dursamples - 1) * invrate
    gaptime = tottime - durtime

    intervals = []

    for i in range(n):
        ifirst = first + i * totsamples
        ilast = ifirst + dursamples - 1
        istart = start + i * tottime
        istop = istart + (dursamples - 1) * invrate
        intervals.append(Interval(start=istart, stop=istop, first=ifirst, last=ilast))

    return intervals



