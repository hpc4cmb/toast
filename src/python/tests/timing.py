# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys
import os
from ..mpi import MPI
from .mpi import MPITestCase

from .. import timing as timing

class TimingTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        self.tm = timing.timing_manager()
        self.tm.set_output_files("timing_report_tot.out", "timing_report_avg.out", self.outdir)

    def test_timing(self):

        def fibonacci(n):
            if n < 2:
                return n;
            return fibonacci(n-1) + fibonacci(n-2);

        def time_fibonacci(n):
            key = ('fibonacci(%i)' % n)
            timer = timing.timer(key)
            print (type(timer))
            timer.start()
            val = fibonacci(n)
            timer.stop()

        t = timing.timer("tmanager test");
        t.start()

        for i in [ 39, 35, 43, 39 ]:
            # python is too slow with these values that run in a couple
            # seconds in C/C++
            n = i - 12;
            time_fibonacci(n-2)
            time_fibonacci(n-1)
            time_fibonacci(n)
            time_fibonacci(n+1)

        self.tm.report()

        self.assertEqual(self.tm.size(), 13)

        for i in range(0, self.tm.size()):
            t = self.tm.at(i)
            print (type(t))
            self.assertFalse(t.real_elapsed() < 0.0)
            self.assertFalse(t.user_elapsed() < 0.0)
