# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
import time

import numpy as np

from ..timing import GlobalTimers, Timer, dump, function_timer, gather_timers
from ._helpers import create_outdir
from .mpi import MPITestCase


@function_timer
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class TimingTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_single(self):
        incr = 200
        dincr = float(incr) / 1000.0
        prec = 1
        tm = Timer()
        self.assertFalse(tm.is_running())
        tm.start()
        time.sleep(dincr)
        elapsed = tm.elapsed_seconds()
        tm.report_elapsed("Test timer elapsed")
        tm.stop()
        try:
            elapsed = tm.elapsed_seconds("This should raise since timer is stopped...")
        except:
            print("Successful exception:  elapsed_seconds() from a stopped timer")
        try:
            tm.report_elapsed("This should raise since timer is stopped...")
        except:
            print("Successful exception:  report_elapsed() for a stopped timer")
        np.testing.assert_almost_equal(tm.seconds(), elapsed, decimal=prec)
        np.testing.assert_almost_equal(tm.seconds(), dincr, decimal=prec)
        tm.report("Test timer stopped")
        tm.clear()
        tm.start()
        time.sleep(dincr)
        try:
            tm.report("This should raise since timer not stopped...")
        except:
            print("Successful exception:  report running timer")
        self.assertTrue(tm.is_running())
        tm.stop()
        # Test pickling
        pkl = pickle.dumps(tm, 2)
        newtm = pickle.loads(pkl)
        tm.report("original")
        newtm.report("pickle roundtrip")

    def test_global(self):
        incr = 200
        dincr = float(incr) / 1000.0
        prec = 1
        gt = GlobalTimers.get()
        tnames = ["timer1", "timer2", "timer3"]
        for nm in tnames:
            try:
                gt.stop(nm)
                tm.report("This should raise since timer not started...")
            except:
                print("Successful exception:  stop an already stopped timer")
        for nm in tnames:
            gt.start(nm)
        for nm in tnames:
            self.assertTrue(gt.is_running(nm))
            try:
                s = gt.seconds(nm)
            except:
                print("Successful exception:  seconds() on running timer")
        gt.stop_all()
        gt.clear_all()
        for nm in tnames:
            gt.start(nm)
        for nm in tnames:
            time.sleep(dincr)
            gt.stop(nm)
        offset = 1
        for nm in tnames:
            np.testing.assert_almost_equal(gt.seconds(nm), offset * dincr, decimal=prec)
            offset += 1
        gt.report()

    def test_comm(self):
        gt = GlobalTimers.get()
        for i in [10, 7, 11]:
            fibonacci(i - 2)
            fibonacci(i - 1)
            fibonacci(i)
        gt.stop_all()
        np = 1
        proc = 0
        if self.comm is not None:
            np = self.comm.size
            proc = self.comm.rank
        for p in range(np):
            if proc == p:
                print("--- Rank {}".format(p), flush=True)
                gt.report()
                print("---", flush=True)
            if self.comm is not None:
                self.comm.barrier()

        result = gather_timers(comm=self.comm)
        if proc == 0:
            for nm in sorted(result.keys()):
                print("{} timing:".format(nm), flush=True)
                props = result[nm]
                for k in sorted(props.keys()):
                    print("  {} = {}".format(k, props[k]), flush=True)
            out = os.path.join(self.outdir, "test_dump")
            dump(result, out)
