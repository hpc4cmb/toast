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

    def test_timing(self):

        tman = timing.timing_manager()
        tman.set_output_file("timing_report.out",
                              self.outdir)

        def fibonacci(n):
            if n < 2:
                return n;
            return fibonacci(n-1) + fibonacci(n-2);

        def time_fibonacci(n):
            atimer = timing.auto_timer('(%i)@%s' % (n, timing.FILE(use_dirname=True)))
            key = ('fibonacci(%i)' % n)
            timer = timing.timer(key)
            timer.start()
            val = fibonacci(n)
            timer.stop()

        tman.clear()
        t = timing.timer("tmanager_test");
        t.start()

        for i in [ 39, 35, 43, 39 ]:
            # python is too slow with these values that run in a couple
            # seconds in C/C++
            n = i - 12;
            time_fibonacci(n-2)
            time_fibonacci(n-1)
            time_fibonacci(n)
            time_fibonacci(n+1)

        tman.report()

        self.assertEqual(tman.size(), 25)

        for i in range(0, tman.size()):
            t = tman.at(i)
            print (type(t))
            self.assertFalse(t.real_elapsed() < 0.0)
            self.assertFalse(t.user_elapsed() < 0.0)
