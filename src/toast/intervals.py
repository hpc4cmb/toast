# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections.abc import Sequence

import numpy as np

from .accelerator import (
    AcceleratorObject,
    accel_data_create,
    accel_data_delete,
    accel_data_present,
    accel_data_update_device,
    accel_data_update_host,
    use_accel_jax,
    use_accel_omp,
)
from .timing import function_timer
from .utils import Logger

if use_accel_jax:
    from .jax.intervals import INTERVALS_JAX


def build_interval_dtype():
    dtdbl = np.dtype("double")
    dtll = np.dtype("longlong")
    fmts = [dtdbl.char, dtdbl.char, dtll.char, dtll.char]
    offs = [
        0,
        dtdbl.alignment,
        2 * dtdbl.alignment,
        2 * dtdbl.alignment + dtll.alignment,
    ]
    return np.dtype(
        {
            "names": ["start", "stop", "first", "last"],
            "formats": fmts,
            "offsets": offs,
        }
    )


interval_dtype = build_interval_dtype()


class IntervalList(Sequence, AcceleratorObject):
    """An list of Intervals which supports logical operations.

    The timestamps define the valid local range of intervals.  When constructing
    from intervals, timespans, or samplespans, the inputs are truncated to the
    allowed range given by the timestamps.

    Args:
        timestamps (array):  Array of local sample times, required.
        intervals (list):  An existing IntervalsList or raw intervals array.
        timespans (list):  A list of tuples containing start and stop times.
        samplespans (list):  A list of tuples containing first and last (inclusive)
            sample ranges.

    """

    def __init__(self, timestamps, intervals=None, timespans=None, samplespans=None):
        super().__init__()
        self.timestamps = timestamps
        if intervals is not None:
            if timespans is not None or samplespans is not None:
                raise RuntimeError(
                    "If constructing from intervals, other spans should be None"
                )
            timespans = [(x.start, x.stop) for x in intervals]
            indices = self._find_indices(timespans)
            self.data = np.array(
                [
                    (self.timestamps[x[0]], self.timestamps[x[1]], x[0], x[1])
                    for x in indices
                ],
                dtype=interval_dtype,
            ).view(np.recarray)
        elif timespans is not None:
            if samplespans is not None:
                raise RuntimeError("Cannot construct from both time and sample spans")
            if len(timespans) == 0:
                self.data = np.zeros(0, dtype=interval_dtype).view(np.recarray)
            else:
                # Construct intervals from time ranges
                for i in range(len(timespans) - 1):
                    if timespans[i][1] > timespans[i + 1][0]:
                        raise RuntimeError("Timespans must be sorted and disjoint")
                indices = self._find_indices(timespans)
                self.data = np.array(
                    [
                        (self.timestamps[x[0]], self.timestamps[x[1]], x[0], x[1])
                        for x in indices
                    ],
                    dtype=interval_dtype,
                ).view(np.recarray)
        elif samplespans is not None:
            if len(samplespans) == 0:
                self.data = np.zeros(0, dtype=interval_dtype).view(np.recarray)
            else:
                # Construct intervals from sample ranges
                for i in range(len(samplespans) - 1):
                    if samplespans[i][1] >= samplespans[i + 1][0]:
                        raise RuntimeError("Sample spans must be sorted and disjoint")
                builder = list()
                for first, last in samplespans:
                    if last < 0 or first >= len(self.timestamps):
                        continue
                    if first < 0:
                        first = 0
                    if last >= len(self.timestamps):
                        last = len(self.timestamps) - 1
                    builder.append((timestamps[first], timestamps[last], first, last))
                self.data = np.array(builder, dtype=interval_dtype).view(np.recarray)
        else:
            # No data yet
            self.data = np.zeros(0, dtype=interval_dtype).view(np.recarray)

    def _find_indices(self, timespans):
        start_indx = np.searchsorted(
            self.timestamps, [x[0] for x in timespans], side="left"
        )
        stop_indx = np.searchsorted(
            self.timestamps, [x[1] for x in timespans], side="right"
        )
        stop_indx -= 1
        # Remove accidental overlap caused by timespan boundary occurring
        # exactly over a time stamp.
        for i in range(start_indx.size - 1):
            if stop_indx[i] == start_indx[i + 1]:
                stop_indx[i] -= 1
        out = list()
        for start, stop in zip(start_indx, stop_indx):
            if stop < 0 or start >= len(self.timestamps):
                continue
            out.append((start, stop))
        return out

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        raise RuntimeError("Cannot delete individual elements from an IntervalList")
        return

    def __contains__(self, item):
        for ival in self.data:
            if ival == item:
                return True
        return False

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.data.__repr__()

    def __eq__(self, other):
        if len(self.data) != len(other):
            return False
        if len(self.timestamps) != len(other.timestamps):
            return False
        if not np.isclose(self.timestamps[0], other.timestamps[0]) or not np.isclose(
            self.timestamps[-1], other.timestamps[-1]
        ):
            return False
        for s, o in zip(self.data, other.data):
            if s.first != o.first:
                return False
            if s.last != o.last:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def simplify(self):
        if len(self.data) == 0:
            return
        propose = list()
        first = self.data[0].first
        last = self.data[0].last
        for i in range(1, len(self.data)):
            cur_first = self.data[i].first
            cur_last = self.data[i].last
            if cur_first == last + 1:
                # This interval is contiguous with the previous one
                last = cur_last
            else:
                # There is a gap
                propose.append(
                    (self.timestamps[first], self.timestamps[last], first, last)
                )
                first = cur_first
                last = cur_last
        propose.append((self.timestamps[first], self.timestamps[last], first, last))
        if len(propose) < len(self.data):
            # Need to update
            self.data = np.array(propose, dtype=interval_dtype).view(np.recarray)

    def __invert__(self):
        if len(self.data) == 0:
            return
        neg = list()
        # Handle range before first interval
        if not np.isclose(self.timestamps[0], self.data[0].start):
            last = self.data[0].first - 1
            neg.append((self.timestamps[0], self.timestamps[last], 0, last))
        for i in range(len(self.data) - 1):
            # Handle gaps between intervals
            cur_last = self.data[i].last
            next_first = self.data[i + 1].first
            if next_first != cur_last + 1:
                # There are some samples in between
                neg.append(
                    (
                        self.timestamps[cur_last + 1],
                        self.timestamps[next_first - 1],
                        cur_last + 1,
                        next_first - 1,
                    )
                )
        # Handle range after last interval
        if not np.isclose(self.timestamps[-1], self.data[-1].stop):
            first = self.data[-1].last + 1
            neg.append(
                (
                    self.timestamps[first],
                    self.timestamps[-1],
                    first,
                    len(self.timestamps) - 1,
                )
            )
        return IntervalList(
            self.timestamps,
            intervals=np.array(neg, dtype=interval_dtype).view(np.recarray),
        )

    def __and__(self, other):
        if len(self.timestamps) != len(other.timestamps):
            raise RuntimeError(
                "Cannot do AND operation on intervals with different timestamps"
            )
        if not np.isclose(self.timestamps[0], other.timestamps[0]) or not np.isclose(
            self.timestamps[-1], other.timestamps[-1]
        ):
            raise RuntimeError(
                "Cannot do AND operation on intervals with different timestamps"
            )
        if len(self.data) == 0 or len(other) == 0:
            return IntervalList(self.timestamps)
        result = list()
        curself = 0
        curother = 0

        # Walk both sequences, building up the intersection.
        while (curself < len(self.data)) and (curother < len(other)):
            low = max(self.data[curself].first, other[curother].first)
            high = min(self.data[curself].last, other[curother].last)
            if low <= high:
                result.append((self.timestamps[low], self.timestamps[high], low, high))
            if self.data[curself].last < other[curother].last:
                curself += 1
            else:
                curother += 1

        return IntervalList(
            self.timestamps,
            intervals=np.array(result, dtype=interval_dtype).view(np.recarray),
        )

    def __or__(self, other):
        if len(self.timestamps) != len(other.timestamps):
            raise RuntimeError(
                "Cannot do OR operation on intervals with different timestamps"
            )
        if not np.isclose(self.timestamps[0], other.timestamps[0]) or not np.isclose(
            self.timestamps[-1], other.timestamps[-1]
        ):
            raise RuntimeError(
                "Cannot do OR operation on intervals with different timestamps"
            )
        if len(self.data) == 0:
            return IntervalList(self.timestamps, intervals=other.data)
        elif len(other) == 0:
            return IntervalList(self.timestamps, intervals=self.data)

        result = list()
        res_first = None
        res_last = None
        curself = 0
        curother = 0

        # Walk both sequences.
        done_self = False
        done_other = False
        while (not done_self) or (not done_other):
            next = None
            if done_self:
                next = other[curother]
                curother += 1
            elif done_other:
                next = self.data[curself]
                curself += 1
            else:
                if self.data[curself].first < other[curother].first:
                    next = self.data[curself]
                    curself += 1
                else:
                    next = other[curother]
                    curother += 1
            if curself >= len(self.data):
                done_self = True
            if curother >= len(other):
                done_other = True

            if res_first is None:
                res_first = next.first
                res_last = next.last
            else:
                # We use '<' here instead of '<=', so that intervals which are next to
                # each other (but not overlapping) are not combined.  If the combination
                # is desired, the simplify() method can be used.
                if next.first < res_last + 1:
                    # We overlap last interval
                    if next.last > res_last:
                        # This interval extends beyond the last interval
                        res_last = next.last
                else:
                    # We have a break, close out previous interval and start a new one
                    result.append(
                        (
                            self.timestamps[res_first],
                            self.timestamps[res_last],
                            res_first,
                            res_last,
                        )
                    )
                    res_first = next.first
                    res_last = next.last
        # Close out final interval
        result.append(
            (self.timestamps[res_first], self.timestamps[res_last], res_first, res_last)
        )

        return IntervalList(
            self.timestamps,
            intervals=np.array(result, dtype=interval_dtype).view(np.recarray),
        )

    def _accel_exists(self):
        if use_accel_omp:
            return accel_data_present(self.data, self._accel_name)
        elif use_accel_jax:
            # specialised for the INTERVALS_JAX dtype
            return isinstance(self.data, INTERVALS_JAX)
        else:
            return False

    def _accel_create(self):
        if use_accel_omp:
            self.data = accel_data_create(self.data, self._accel_name)
        elif use_accel_jax:
            # specialised for the INTERVALS_JAX dtype
            # NOTE: this call is timed at the INTERVALS_JAX level
            self.data = INTERVALS_JAX(self.data)

    def _accel_update_device(self):
        if use_accel_omp:
            self.data = accel_data_update_device(self.data, self._accel_name)
        elif use_accel_jax:
            # specialised for the INTERVALS_JAX dtype
            # NOTE: this call is timed at the INTERVALS_JAX level
            self.data = INTERVALS_JAX(self.data)

    def _accel_update_host(self):
        if use_accel_omp:
            self.data = accel_data_update_host(self.data, self._accel_name)
        elif use_accel_jax:
            # specialised for the INTERVALS_JAX dtype
            # this moves the data back into a numpy array
            # NOTE: this call is timed at the INTERVALS_JAX level
            self.data = self.data.to_host()

    def _accel_delete(self):
        if use_accel_omp:
            self.data = accel_data_delete(self.data, self._accel_name)
        elif use_accel_jax and self._accel_exists():
            # Ensures data has been properly reset
            # if we observe that its type is still a GPU type
            # does NOT move data back from GPU
            self.data = self.data.host_data


@function_timer
def regular_intervals(n, start, first, rate, duration, gap):
    """Function to generate regular intervals with gaps.

    This creates a raw numpy array of interval_dtype (*not* an IntervalList object,
    which requires full timestamp information), given a start time/sample and time
    span for the interval and the gap in time between intervals.  The
    length of the interval and the total interval + gap are rounded down to the
    nearest sample and all intervals in the list are created using those
    lengths.

    If the time span is an exact multiple of the sampling, then the
    final sample is excluded.  The reason we always round down to the whole
    number of samples that fits inside the time range is so that the requested
    time span boundary (one hour, one day, etc) will fall in between the last
    sample of one interval and the first sample of the next.

    Args:
        n (int): the number of intervals.
        start (float): the start time in seconds.
        first (int): the first sample index, which occurs at "start".
        rate (float): the sample rate in Hz.
        duration (float): the length of the interval in seconds.
        gap (float): the length of the gap in seconds.

    Returns:
        (recarray):  A recarray of the intervals

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

    intervals = list()

    for i in range(n):
        ifirst = first + i * totsamples
        ilast = ifirst + dursamples - 1
        # The time span between interval starts (the first sample of one
        # interval to the first sample of the next) includes the one extra
        # sample time.
        istart = start + i * (totsamples * invrate)
        # The stop time is the timestamp of the last valid sample (thus the -1).
        istop = istart + ((dursamples - 1) * invrate)
        intervals.append((istart, istop, ifirst, ilast))

    return np.array(intervals, dtype=interval_dtype).view(np.recarray)
