# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Thin wrapper around TiMemory

https://github.com/jrmadsen/TiMemory
"""

from functools import wraps

use_timemory = None
try:
    import timemory
    use_timemory = True
except ImportError:
    use_timemory = False

if use_timemory:
    # Use the timemory package
    import timemory
    from timemory.util import timer
    from timemory.util import auto_timer as tm_auto_timer
    from timemory.util import rss_usage as tm_rss_usage
    from timemory import (timing_manager, FILE, enable_signal_detection,
        set_exit_action)

    # We wrap the decorators so that they work both with normal functions
    # and methods of a class instance (we do not have any class or static
    # methods to profile at this time).

    import inspect

    def auto_timer(f):
        if inspect.ismethod(f):
            ti = tm_auto_timer(key=f.__self__.__name__)
            return ti(f)
        else:
            # must be a normal function...
            ti = tm_auto_timer(key=f.__name__)
            return ti(f)

    def rss_usage(f):
        if inspect.ismethod(f):
            ti = tm_rss_usage(key=f.__self__.__name__)
            return ti(f)
        else:
            # must be a normal function...
            ti = tm_rss_usage(key=f.__name__)
            return ti(f)

else:
    # Define equivalent functions which are no-ops
    import time

    def FILE():
        return ""

    def enable_signal_detection(*args, **kwargs):
        pass

    def report(*args, **kwargs):
        pass

    def set_exit_action(*args, **kwargs):
        pass

    def auto_timer(f):
        @wraps(f)
        def df(*args, **kwargs):
            return f(*args, **kwargs)
        return df

    def rss_usage(f):
        @wraps(f)
        def df(*args, **kwargs):
            return f(*args, **kwargs)
        return df

    class timing_manager:
        def __init__(self, *args, **kwargs):
            pass
        def report(self, *args, **kwargs):
            pass
        def clear(self, *args, **kwargs):
            pass


    class timer(object):

        def __init__(self, name):
            self._name = name
            self._running = False
            self._timestart = 0
            self._timestop = 0
            self._elapsed = 0
            print("DBG: created timer {}".format(self._name), flush=True)

        def start(self):
            if not self._running:
                self._timestart = time.time()
                self._running = True
                print("DBG: started timer {} at {}".format(self._name, self._timestart), flush=True)
            else:
                print("DBG: timer {} already started".format(self._name), flush=True)
            return

        def stop(self):
            if self._running:
                self._timestop = time.time()
                self._elapsed = self._timestop - self._timestart
                self._running = False
                print("DBG: stopped timer {} at {}".format(self._name, self._timestop), flush=True)
                print("DBG: timer {} elapsed {}".format(self._name, self._elapsed), flush=True)
            else:
                print("DBG: timer {} already stopped".format(self._name), flush=True)
            return

        def report(self):
            print("DBG: timer {} reporting".format(self._name), flush=True)
            if self._running:
                raise RuntimeError("stop the timer before calling report()")
            print("{}: {} seconds".format(self._name, self._elapsed),
                flush=True)
