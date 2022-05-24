# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import numpy.testing as nt

from ..intervals import IntervalList, interval_dtype
from .mpi import MPITestCase

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
        check = np.array(
            [
                (float(10.0 * x + 2), float(10.0 * x + 5), (10 * x + 2), (10 * x + 5))
                for x in range(10)
            ],
            dtype=interval_dtype,
        ).view(np.recarray)
        check_neg = [
            (0.0, 1.0, 0, 1),
        ]
        check_neg.extend(
            [
                (float(10.0 * x + 6), float(10.0 * x + 11), (10 * x + 6), (10 * x + 11))
                for x in range(9)
            ]
        )
        check_neg.append((96.0, 99.0, 96, 99))
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
        ranges = [(x, x + 9) for x in boundaries]
        check = np.array([(stamps[10], stamps[89], 10, 89)], dtype=interval_dtype).view(
            np.recarray
        )
        ival = IntervalList(stamps, samplespans=ranges)
        # print("ival = ", ival)
        ival.simplify()
        # print("simple ival = ", ival)
        self.assertTrue(ival[0] == check[0])

    def test_bitwise(self):
        stamps = np.arange(100, dtype=np.float64)
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
        # print("full = ", full)
        check = np.array([(stamps[0], stamps[-1], 0, 99)], dtype=interval_dtype).view(
            np.recarray
        )
        # print(f"check = {check}")
        self.assertTrue(full[0] == check)

        empty = ival & neg
        # print("empty = ", empty)

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
