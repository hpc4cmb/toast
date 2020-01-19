# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..timing import function_timer

from .interval import Interval


@function_timer
def regular_intervals(n, start, first, rate, duration, gap):
    """Function to generate simulated regular intervals.

    This creates a list of intervals, given a start time/sample and time
    span for the interval and the gap in time between intervals.  The
    length of the interval and the total interval + gap are rounded down to the
    nearest sample and all intervals in the list are created using those
    lengths.

    If the time span is an exact multiple of the sampling, then the
    final sample is excluded.  The reason we always round down to the whole
    number of samples that fits inside the time range is so that the requested
    time span boundary (one hour, one day, etc) will fall in between the last
    sample of one interval and the first sample of the next.

    Example:  you want to simulate science observations of length 22 hours and
    then have 4 hours of down time (e.g. a cooler cycle).  Specifying a
    duration of 22*3600 and a gap of 4*3600 will result in a total time for the
    science + gap of a fraction of a sample less than 26 hours.  So the
    requested 26 hour mark will fall between the last sample of one regular
    interval and the first sample of the next.  Note that this fraction of
    a sample will accumulate if you have many, many intervals.

    This function is intended only for simulations- in the case of real data,
    the timestamp of every sample is known and boundaries between changes in
    the experimental configuration are already specified.

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
    # requested time span (rounded down to a whole number).  Check for the
    # case of the time span being an exact number of samples- in which case
    # the final sample is excluded.

    lower = int((duration + gap) * rate)
    totsamples = None
    if np.absolute(lower * invrate - (duration + gap)) > 1.0e-12:
        totsamples = lower + 1
    else:
        totsamples = lower

    lower = int(duration * rate)
    dursamples = None
    if np.absolute(lower * invrate - duration) > 1.0e-12:
        dursamples = lower + 1
    else:
        dursamples = lower

    gapsamples = totsamples - dursamples

    intervals = []

    for i in range(n):
        ifirst = first + i * totsamples
        ilast = ifirst + dursamples - 1
        # The time span between interval starts (the first sample of one
        # interval to the first sample of the next) includes the one extra
        # sample time.
        istart = start + i * (totsamples * invrate)
        # The stop time is the timestamp of the last valid sample (thus the -1).
        istop = istart + ((dursamples - 1) * invrate)
        intervals.append(Interval(start=istart, stop=istop, first=ifirst, last=ilast))

    return intervals
