# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import numpy.testing as nt

from ..intervals import IntervalList, interval_dtype
from .mpi import MPITestCase


class IntervalTest(MPITestCase):
    def setUp(self):
        self.rate = 123.456
        self.duration = 24 * 3601.23
        self.gap = 3600.0
        self.start = 5432.1
        self.first = 10
        self.nint = 3

    def test_list(self):
        stamps = np.arange(100, dtype=np.float64)
        timespans = [(10.0 * x + 2.0, 10.0 * x + 5.0) for x in range(10)]
        sampspans = [(10 * x + 2, 10 * x + 5) for x in range(10)]
        check = np.array(
            [
                (float(10.0 * x + 2), float(10.0 * x + 5), (10 * x + 2), (10 * x + 5))
                for x in range(10)
            ],
            dtype=interval_dtype,
        ).view(np.recarray)
        check_neg = [
            (0.0, 2.0, 0, 2),
        ]
        check_neg.extend(
            [
                (float(10.0 * x + 5), float(10.0 * x + 12), (10 * x + 5), (10 * x + 12))
                for x in range(9)
            ]
        )
        check_neg.append((95.0, 99.0, 95, 100))
        check_neg = np.array(check_neg, dtype=interval_dtype).view(np.recarray)

        itime = IntervalList(stamps, timespans=timespans)

        for it, chk in zip(itime, check):
            self.assertTrue(it == chk)

        isamp = IntervalList(stamps, samplespans=sampspans)
        for it, chk in zip(isamp, check):
            self.assertTrue(it == chk)

        negated = ~isamp
        for it, chk in zip(negated, check_neg):
            self.assertTrue(it == chk)

    def test_simplify(self):
        stamps = np.arange(100, dtype=np.float64)
        boundaries = [10 * x for x in range(1, 9)]
        ranges = [(x, x + 10) for x in boundaries]
        check = np.array([(stamps[10], stamps[90], 10, 90)], dtype=interval_dtype).view(
            np.recarray
        )
        ival = IntervalList(stamps, samplespans=ranges)
        ival.simplify()
        self.assertTrue(ival[0] == check[0])

    def test_bitwise(self):
        nsample = 100
        stamps = np.arange(nsample, dtype=np.float64)
        raw = np.array(
            [
                (float(10.0 * x + 2), float(10.0 * x + 5), (10 * x + 2), (10 * x + 5))
                for x in range(10)
            ],
            dtype=interval_dtype,
        ).view(np.recarray)
        ival = IntervalList(stamps, intervals=raw)
        neg = ~ival

        full = ival | neg
        full.simplify()
        check = np.array(
            [(stamps[0], stamps[-1], 0, nsample)], dtype=interval_dtype
        ).view(np.recarray)
        self.assertTrue(full[0] == check)

        empty = ival & neg

        rawshift = np.array(
            [
                (float(10.0 * x + 3), float(10.0 * x + 6), (10 * x + 3), (10 * x + 6))
                for x in range(10)
            ],
            dtype=interval_dtype,
        ).view(np.recarray)
        shifted = IntervalList(stamps, intervals=rawshift)

        and_check = IntervalList(
            stamps,
            intervals=np.array(
                [
                    (
                        float(10.0 * x + 3),
                        float(10.0 * x + 5),
                        (10 * x + 3),
                        (10 * x + 5),
                    )
                    for x in range(10)
                ],
                dtype=interval_dtype,
            ).view(np.recarray),
        )

        or_check = IntervalList(
            stamps,
            intervals=np.array(
                [
                    (
                        float(10.0 * x + 2),
                        float(10.0 * x + 6),
                        (10 * x + 2),
                        (10 * x + 6),
                    )
                    for x in range(10)
                ],
                dtype=interval_dtype,
            ).view(np.recarray),
        )

        test = ival & shifted
        self.assertTrue(test == and_check)

        test = ival | shifted
        self.assertTrue(test == or_check)

    def test_union(self):
        # Construct two disjoint interval lists that together for a
        # continuous span.  Then form a union between them and confirm
        # that the total number of intervals is the sum of the two lists
        # (no intervals were merged)
        stamps = np.arange(100, dtype=np.float64)
        breaks = stamps[::10]
        nbreak = len(breaks)
        times1 = [(breaks[2 * i], breaks[2 * i + 1]) for i in range(nbreak // 2)]
        times2 = [
            (breaks[2 * i + 1], breaks[2 * i + 2]) for i in range(nbreak // 2 - 1)
        ]
        intervals1 = IntervalList(stamps, timespans=times1)
        intervals2 = IntervalList(stamps, timespans=times2)
        ninterval1 = len(intervals1)
        ninterval2 = len(intervals2)
        intervals12 = intervals1 | intervals2
        ninterval12 = len(intervals12)
        assert ninterval1 + ninterval2 == ninterval12

    def test_inverse(self):
        # Form an interval list and its inverse and confirm that together
        # they cover the entire observation
        n = 100
        stamps = np.arange(n, dtype=np.float64)
        breaks = stamps[::10]
        nbreak = len(breaks)
        times1 = [(breaks[2 * i], breaks[2 * i + 1]) for i in range(nbreak // 2)]
        intervals0 = IntervalList(stamps, timespans=times1)
        intervals1 = ~intervals0
        intervals2 = ~intervals1
        included = np.zeros(n, dtype=int)
        for ival in intervals1:
            included[ival.first : ival.last] += 1
        assert not np.all(included)
        for ival in intervals2:
            included[ival.first : ival.last] += 1
        assert np.all(included == 1)

    def test_closed_time_spans(self):
        n_samp = 100
        n_intr = 10
        stamps = 1000.0 * np.arange(n_samp, dtype=np.float64)

        # Sample ranges are open ended
        samplespans = [(x * n_intr, x * n_intr + n_intr) for x in range(n_intr)]
        intr_samples = IntervalList(stamps, samplespans=samplespans)

        # Time ranges are open ended *unless* the end time coincides
        # with the observation end time.
        timespans = [
            (stamps[x[0]], stamps[min(x[1], stamps.size - 1)]) for x in samplespans
        ]
        intr_times = IntervalList(stamps, timespans=timespans)

        self.assertTrue(intr_samples == intr_times)

    # def test_tochunks(self):
    #     intrvls = regular_intervals(
    #         self.nint, self.start, self.first, self.rate, self.duration, self.gap
    #     )
    #     totsamp = self.nint * (intrvls[0].last - intrvls[0].first)
    #     totsamp += self.nint * (intrvls[1].first - intrvls[0].last - 1)
    #     sizes = intervals_to_chunklist(intrvls, totsamp, startsamp=self.first + 10)
    #     # for it in intrvls:
    #     #     print(it.first," ",it.last," ",it.start," ",it.stop)
    #     # print(sizes)
    #     nt.assert_equal(np.sum(sizes), totsamp)
    #
    # def test_regular(self):
    #     intrvls = regular_intervals(
    #         self.nint, self.start, self.first, self.rate, self.duration, self.gap
    #     )
    #
    #     goodsamp = self.nint * (int(self.duration * self.rate) + 1)
    #
    #     check = 0
    #
    #     for it in intrvls:
    #         # print(it.first," ",it.last," ",it.start," ",it.stop)
    #         check += it.last - it.first
    #
    #     nt.assert_equal(check, goodsamp)
    #     return
