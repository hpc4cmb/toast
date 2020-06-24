# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .mpi import MPITestCase

import numpy.testing as nt

from ..intervals import Interval, IntervalList

# from ..tod.interval import intervals_to_chunklist
#
# from ..tod.sim_interval import regular_intervals


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
        check = [
            Interval(
                start=float(10.0 * x + 2),
                stop=float(10.0 * x + 5),
                first=(10 * x + 2),
                last=(10 * x + 5),
            )
            for x in range(10)
        ]
        check_neg = [Interval(start=0.0, stop=1.0, first=0, last=1)]
        check_neg.extend(
            [
                Interval(
                    start=float(10.0 * x + 6),
                    stop=float(10.0 * x + 11),
                    first=(10 * x + 6),
                    last=(10 * x + 11),
                )
                for x in range(9)
            ]
        )
        check_neg.append(Interval(start=96.0, stop=99.0, first=96, last=99))
        # print("check = ", check)
        # print("check_neg = ", check_neg)

        itime = IntervalList(stamps, timespans=timespans)
        # print("itime = ", itime)

        for it, chk in zip(itime, check):
            self.assertTrue(it == chk)

        isamp = IntervalList(stamps, samplespans=sampspans)
        # print("isamp = ", isamp)
        for it, chk in zip(isamp, check):
            self.assertTrue(it == chk)

        negated = ~isamp
        for it, chk in zip(negated, check_neg):
            self.assertTrue(it == chk)

    def test_simplify(self):
        stamps = np.arange(100, dtype=np.float64)
        boundaries = [10 * x for x in range(1, 9)]
        ranges = [(x, x + 9) for x in boundaries]
        check = Interval(first=10, last=89, start=stamps[10], stop=stamps[89])
        ival = IntervalList(stamps, samplespans=ranges)
        # print("ival = ", ival)
        ival.simplify()
        # print("simple ival = ", ival)
        self.assertTrue(ival[0] == check)

    def test_bitwise(self):
        stamps = np.arange(100, dtype=np.float64)
        raw = [
            Interval(
                start=float(10.0 * x + 2),
                stop=float(10.0 * x + 5),
                first=(10 * x + 2),
                last=(10 * x + 5),
            )
            for x in range(10)
        ]
        ival = IntervalList(stamps, intervals=raw)
        neg = ~ival

        full = ival | neg
        # print("full = ", full)
        self.assertTrue(
            full[0] == Interval(start=stamps[0], stop=stamps[-1], first=0, last=99)
        )

        empty = ival & neg
        # print("empty = ", empty)

        rawshift = [
            Interval(
                start=float(10.0 * x + 3),
                stop=float(10.0 * x + 6),
                first=(10 * x + 3),
                last=(10 * x + 6),
            )
            for x in range(10)
        ]
        shifted = IntervalList(stamps, intervals=rawshift)

        and_check = IntervalList(
            stamps,
            intervals=[
                Interval(
                    start=float(10.0 * x + 3),
                    stop=float(10.0 * x + 5),
                    first=(10 * x + 3),
                    last=(10 * x + 5),
                )
                for x in range(10)
            ],
        )

        or_check = IntervalList(
            stamps,
            intervals=[
                Interval(
                    start=float(10.0 * x + 2),
                    stop=float(10.0 * x + 6),
                    first=(10 * x + 2),
                    last=(10 * x + 6),
                )
                for x in range(10)
            ],
        )

        test = ival & shifted
        # print("bit and = ", test)
        self.assertTrue(test == and_check)

        test = ival | shifted
        # print("bit or = ", test)
        self.assertTrue(test == or_check)

    # def test_tochunks(self):
    #     intrvls = regular_intervals(
    #         self.nint, self.start, self.first, self.rate, self.duration, self.gap
    #     )
    #     totsamp = self.nint * (intrvls[0].last - intrvls[0].first + 1)
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
    #         check += it.last - it.first + 1
    #
    #     nt.assert_equal(check, goodsamp)
    #     return
