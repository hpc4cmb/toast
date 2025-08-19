# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
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
        samplespans (list):  A list of tuples containing first and last (exclusive)
            sample ranges.

    """

    def __init__(self, timestamps, intervals=None, timespans=None, samplespans=None):
        super().__init__()
        self.timestamps = timestamps
        if intervals is not None:
            # Construct intervals using timespans from the provided intervals
            if timespans is not None or samplespans is not None:
                raise RuntimeError(
                    "If constructing from intervals, other spans should be None"
                )
            if len(intervals) == 0:
                self.data = np.zeros(0, dtype=interval_dtype).view(np.recarray)
            else:
                timespans = [(x.start, x.stop) for x in intervals]
                indices, times = self._find_indices(timespans)
                self.data = np.array(
                    [
                        (time[0], time[1], ind[0], ind[1])
                        for (time, ind) in zip(times, indices)
                    ],
                    dtype=interval_dtype,
                ).view(np.recarray)
        elif timespans is not None:
            # Construct intervals using provided timespans
            if samplespans is not None:
                raise RuntimeError("Cannot construct from both time and sample spans")
            if len(timespans) == 0:
                self.data = np.zeros(0, dtype=interval_dtype).view(np.recarray)
            else:
                timespans = np.vstack(timespans)
                for i in range(len(timespans) - 1):
                    if np.isclose(timespans[i][1], timespans[i + 1][0], rtol=1e-12):
                        # Force nearly equal timestamps to match
                        timespans[i][1] = timespans[i + 1][0]
                    # Check that the intervals are sorted and disjoint
                    if timespans[i][1] > timespans[i + 1][0]:
                        t1 = timespans[i][1]
                        t2 = timespans[i + 1][0]
                        dt = t1 - t2
                        ts = np.median(np.diff(timestamps))
                        msg = f"Timespans must be sorted and disjoint"
                        msg += f" but {t1} - {t2} = {dt} s = {dt / ts} samples)"
                        raise RuntimeError(msg)
                #  Map interval times into sample indices
                indices, times = self._find_indices(timespans)
                self.data = np.array(
                    [
                        (time[0], time[1], ind[0], ind[1])
                        for (time, ind) in zip(times, indices)
                    ],
                    dtype=interval_dtype,
                ).view(np.recarray)
        elif samplespans is not None:
            # Construct intervals from sample ranges
            if len(samplespans) == 0:
                self.data = np.zeros(0, dtype=interval_dtype).view(np.recarray)
            else:
                for i in range(len(samplespans) - 1):
                    if samplespans[i][1] > samplespans[i + 1][0]:
                        raise RuntimeError("Sample spans must be sorted and disjoint")
                builder = list()
                for first, last in samplespans:
                    if last < 0 or first >= len(self.timestamps):
                        continue
                    if first < 0:
                        first = 0
                    if last > len(self.timestamps):
                        last = len(self.timestamps)
                    builder.append(
                        (self._sample_time(first), self._sample_time(last), first, last)
                    )
                self.data = np.array(builder, dtype=interval_dtype).view(np.recarray)
        else:
            # No data yet
            self.data = np.zeros(0, dtype=interval_dtype).view(np.recarray)

    def _sample_time(self, sample):
        nsample = len(self.timestamps)
        if sample < 0 or sample > nsample:
            msg = f"Invalid sample index: {sample} not in [0, {nsample}]"
            raise RuntimeError(msg)
        if sample == nsample:
            # Handle the end of the timestamps differently
            return self.timestamps[sample - 1]
        else:
            return self.timestamps[sample]

    def _find_indices(self, timespans):
        # Each interval covers all samples where the sample time meets:
        # interval.start <= self.timestamps AND self.timestamps < interval.stop
        # (open-ended interval)
        # with one exception: if the interval ends at the last timestamp, the
        # corresponding sample is included (closed interval)
        start_time, stop_time = np.vstack(timespans).T
        # Cut out timespans that do not overlap with the available timestamps
        good = np.logical_and(
            start_time < self.timestamps[-1], stop_time > self.timestamps[0]
        )
        start_time = start_time[good]
        stop_time = stop_time[good]
        start_indx = np.searchsorted(self.timestamps, start_time, side="left")
        stop_indx = np.searchsorted(self.timestamps, stop_time, side="left")
        # Include the last sample where the stop time matches the last time stamp
        nsample = len(self.timestamps)
        stop_indx[stop_indx == nsample - 1] = nsample
        times = list()
        samples = list()
        for start, stop, first, last in zip(
            start_time, stop_time, start_indx, stop_indx
        ):
            times.append((start, stop))
            samples.append((first, last))
        return samples, times

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
        s = "<IntervalList [\n"
        for ival in self.data:
            s += f" {ival.start:15.3f} - {ival.stop:15.3f} ({ival.first:9} - {ival.last:9}),\n"
        s += "]>"
        # return self.data.__repr__()
        return s

    def __eq__(self, other):
        if len(self.data) != len(other):
            return False
        if len(self.timestamps) != len(other.timestamps):
            return False
        # Comparing timestamps with default tolerances to np.isclose
        # is always True.  Must use sufficiently tight tolerances
        if not np.isclose(
            self.timestamps[0], other.timestamps[0], rtol=1e-12
        ) or not np.isclose(self.timestamps[-1], other.timestamps[-1], rtol=1e-12):
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
        start = self.data[0].start
        stop = self.data[0].stop
        for i in range(1, len(self.data)):
            cur_first = self.data[i].first
            cur_last = self.data[i].last
            cur_start = self.data[i].start
            cur_stop = self.data[i].stop
            if cur_first == last:
                # This interval is contiguous with the previous one
                last = cur_last
                stop = cur_stop
            else:
                # There is a gap
                propose.append((start, stop, first, last))
                first = cur_first
                last = cur_last
                start = cur_start
                stop = cur_stop
        propose.append((start, stop, first, last))
        if len(propose) < len(self.data):
            # Need to update
            self.data = np.array(propose, dtype=interval_dtype).view(np.recarray)

    def __invert__(self):
        if len(self.data) == 0:
            return
        neg = list()
        # Handle range before first interval
        if not np.isclose(self.timestamps[0], self.data[0].start, rtol=1e-12):
            neg.append((self.timestamps[0], self.data[0].start, 0, self.data[0].first))
        for i in range(len(self.data) - 1):
            # Handle gaps between intervals
            cur_last = self.data[i].last
            cur_stop = self.data[i].stop
            next_first = self.data[i + 1].first
            next_start = self.data[i + 1].start
            if next_first != cur_last + 1:
                # There are some samples in between
                neg.append((cur_stop, next_start, cur_last, next_first))
        # Handle range after last interval
        if not np.isclose(self.timestamps[-1], self.data[-1].stop, rtol=1e-12):
            neg.append(
                (
                    self.data[-1].stop,
                    self.timestamps[-1],
                    self.data[-1].last,
                    len(self.timestamps),
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
        if not np.isclose(
            self.timestamps[0], other.timestamps[0], rtol=1e-12
        ) or not np.isclose(self.timestamps[-1], other.timestamps[-1], rtol=1e-12):
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
            start = max(self.data[curself].start, other[curother].start)
            stop = min(self.data[curself].stop, other[curother].stop)
            if start < stop:
                low = max(self.data[curself].first, other[curother].first)
                high = min(self.data[curself].last, other[curother].last)
                result.append((start, stop, low, high))
            if self.data[curself].stop < other[curother].stop:
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
        if not np.isclose(
            self.timestamps[0], other.timestamps[0], rtol=1e-12
        ) or not np.isclose(self.timestamps[-1], other.timestamps[-1], rtol=1e-12):
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
        res_start = None
        res_stop = None
        curself = 0
        curother = 0

        # Walk both sequences.
        done_self = False
        done_other = False
        while (not done_self) or (not done_other):
            next_ = None
            if done_self:
                next_ = other[curother]
                curother += 1
            elif done_other:
                next_ = self.data[curself]
                curself += 1
            else:
                if self.data[curself].first < other[curother].first:
                    next_ = self.data[curself]
                    curself += 1
                else:
                    next_ = other[curother]
                    curother += 1
            if curself >= len(self.data):
                done_self = True
            if curother >= len(other):
                done_other = True

            if res_first is None:
                res_first = next_.first
                res_last = next_.last
                res_start = next_.start
                res_stop = next_.stop
            else:
                # We use '<' here instead of '<=', so that intervals which are next to
                # each other (but not overlapping) are not combined.  If the combination
                # is desired, the simplify() method can be used.
                if next_.first < res_last:
                    # We overlap last interval
                    if next_.last > res_last:
                        # This interval extends beyond the last interval
                        res_last = next_.last
                        res_stop = next_.stop
                else:
                    # We have a break, close out previous interval and start a new one
                    result.append(
                        (
                            res_start,
                            res_stop,
                            res_first,
                            res_last,
                        )
                    )
                    res_first = next_.first
                    res_last = next_.last
                    res_start = next_.start
                    res_stop = next_.stop
        # Close out final interval
        result.append((res_start, res_stop, res_first, res_last))

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
        ilast = ifirst + dursamples
        # The time span between interval starts (the first sample of one
        # interval to the first sample of the next) includes the one extra
        # sample time.
        istart = start + i * (totsamples * invrate)
        # The stop time is the timestamp of the last sample
        istop = istart + (dursamples * invrate)
        intervals.append((istart, istop, ifirst, ilast))

    return np.array(intervals, dtype=interval_dtype).view(np.recarray)
