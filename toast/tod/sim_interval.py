# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np

from ..dist import distribute_uniform

from .interval import Interval



def regular_intervals(n, start, first, rate, duration, gap, chunks=1):
    """
    Function to generate regular intervals.

    This creates a list of intervals, given a start time/sample and time
    span for the interval and the gap in time between intervals.  The 
    length of the interval and the gap are rounded to the nearest sample
    and all intervals in the list are created using those lengths.

    Optionally, the valid data span can be subdivided into some number
    of contiguous subchunks.

    Args:
        n (int): the number of intervals.
        start (float): the start time in seconds.
        first (int): the first sample index, which occurs at "start".
        rate (float): the sample rate in Hz.
        duration (float): the length of the interval in seconds.
        gap (float): the length of the gap in seconds.
        chunks (int): divide the valid data in "duration" into this
            number of contiguous chunks.

    Returns:
        (list): a list of Interval objects.
    """
    invrate = 1.0 / rate

    # Compute the whole number of samples that fit completely within the
    # requested time span.
    totsamples = int((duration + gap) * rate) + 1
    dursamples = int(duration * rate) + 1
    gapsamples = totsamples - dursamples

    # Compute the actual time span for this number of samples
    tottime = (totsamples - 1) * invrate
    durtime = (dursamples - 1) * invrate
    gaptime = tottime - durtime

    # If we have sub-chunks, compute them now
    chnks = distribute_uniform(dursamples, chunks)

    intervals = []

    for i in range(n):
        ifirst = first + i * totsamples
        istart = start + i * tottime
        for c in chnks:
            cfirst = ifirst + c[0]
            clast = cfirst + c[1] - 1
            cstart = istart + (c[0] * invrate)
            cstop = istart + ((c[0] + c[1] - 1) * invrate)
            intervals.append(Interval(start=cstart, stop=cstop, first=cfirst, last=clast))

    return intervals



