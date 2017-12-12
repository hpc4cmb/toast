# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys
import os
import time

from ..mpi import MPI
from .mpi import MPITestCase
from .. import timing as timing


class TimingTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        timing.timing_manager.use_timers = True
        timing.timing_manager.serial_report = True

    # Test if the timers are working if not disabled at compilation
    def test_timing(self):
        if timing.enabled() is False:
            return

        tman = timing.timing_manager()
        tman.set_output_file("timing_report.out", self.outdir, "timing_report.json")

        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        def time_fibonacci(n):
            atimer = timing.auto_timer('({})@{}'.format(n, timing.FILE(use_dirname=True)))
            key = ('fibonacci(%i)' % n)
            timer = timing.timer(key)
            timer.start()
            fibonacci(n)
            timer.stop()

        tman.clear()
        t = timing.timer("tmanager_test")
        t.start()

        for i in [39, 35, 43, 39]:
            # python is too slow with these values that run in a couple
            # seconds in C/C++
            n = i - 12
            time_fibonacci(n - 2)
            time_fibonacci(n - 1)
            time_fibonacci(n)
            time_fibonacci(n + 1)

        tman.report()

        self.assertEqual(tman.size(), 25)

        for i in range(0, tman.size()):
            t = tman.at(i)
            self.assertFalse(t.real_elapsed() < 0.0)
            self.assertFalse(t.user_elapsed() < 0.0)
        timing.toggle(True)


    # Test the timing on/off toggle functionalities
    def test_toggle(self):
        # if compiled with DISABLE_TIMERS
        if timing.enabled() is False:
            return

        tman = timing.timing_manager()
        timing.toggle(True)
        tman.clear()

        timing.toggle(True)
        if True:
            autotimer = timing.auto_timer("on")
            time.sleep(1)
        self.assertEqual(tman.size(), 1)

        tman.clear()
        timing.toggle(False)
        if True:
            autotimer = timing.auto_timer("off")
            time.sleep(1)
        self.assertEqual(tman.size(), 0)

        tman.clear()
        timing.toggle(True)
        if True:
            autotimer_on = timing.auto_timer("on")
            timing.toggle(False)
            autotimer_off = timing.auto_timer("off")
            time.sleep(1)
        self.assertEqual(tman.size(), 1)

        tman.set_output_file("timing_toggle.out", self.outdir,
                             "timing_toggle.json")
        tman.report()


    # Test the timing on/off toggle functionalities
    def test_max_depth(self):
        # if compiled with DISABLE_TIMERS
        if timing.enabled() is False:
            return

        tman = timing.timing_manager()
        timing.toggle(True)
        tman.clear()

        def create_timer(n):
            autotimer = timing.auto_timer('{}'.format(n))
            time.sleep(0.25)
            if n < 8:
                create_timer(n + 1)

        ntimers = 4
        timing.timing_manager.max_timer_depth = ntimers
        create_timer(0)

        self.assertEqual(tman.size(), ntimers)

        tman.set_output_file("timing_depth.out", self.outdir,
                             "timing_depth.json")
        tman.report()
