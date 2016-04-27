# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np

from .interval import Interval



def regular_intervals(n, start, first, rate, duration, gap):
    """
    Function to generate regular intervals.

    This creates a list of intervals, given a start time/sample and time
    span for the interval and the gap in time between intervals.  The 
    length of the interval and the gap are rounded to the nearest sample
    and all intervals in the list are created using those lengths.

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
    dursamples = int(duration * rate) + 1
    gapsamples = int(gap * rate) + 1
    totsamples = dursamples + gapsamples
    durtime = (dursamples - 1) / rate
    gaptime = (gapsamples - 1) / rate
    tottime = durtime + gaptime

    intervals = []

    for i in range(n):
        ifirst = first + i * totsamples
        ilast = ifirst + dursamples - 1
        istart = start + i * tottime
        istop = istart + durtime
        intervals.append(Interval(start=istart, stop=istop, first=ifirst, last=ilast))

    return intervals
