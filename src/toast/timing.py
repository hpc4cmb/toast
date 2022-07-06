# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import csv
import inspect
import os
import re
from collections import OrderedDict
from functools import wraps

import numpy as np

from ._libtoast import GlobalTimers, Timer
from .utils import Environment

stack_toast_file_pat = re.compile(r".*(toast.*)\.py")

# Global list of functions to ignore in our simplified timing stacktrace.
# This can be updated by decorating functions with the function_timer_stackskip
# below.

stack_toast_skip = {
    "df",
    "<module>",
}


def function_timer(f):
    env = Environment.get()
    ft = env.function_timers()
    if ft:
        fname = f"{f.__qualname__}"

        @wraps(f)
        def df(*args, **kwargs):
            global stack_toast_skip
            gt = GlobalTimers.get()
            # Build a name from the current function and the call trace.
            tnm = "(function) "
            fnmlist = list()
            frm = inspect.currentframe().f_back
            while frm:
                mat = stack_toast_file_pat.match(frm.f_code.co_filename)
                if mat is not None:
                    # This is a stack frame that we care about
                    if "self" in frm.f_locals:
                        # this is inside a class instance
                        funcname = f"{frm.f_locals['self'].__class__.__name__}.{frm.f_code.co_name}"
                        if funcname not in stack_toast_skip:
                            found = False
                            for base in frm.f_locals["self"].__class__.__bases__:
                                basename = f"{base.__name__}.{frm.f_code.co_name}"
                                if basename in stack_toast_skip:
                                    found = True
                                    break
                            if not found:
                                fnmlist.append(funcname)
                    else:
                        # this is just a function call
                        if frm.f_code.co_name not in stack_toast_skip:
                            fnmlist.append(frm.f_code.co_name)
                frm = frm.f_back

            if len(fnmlist) > 0:
                tnm += "|".join(reversed(fnmlist))
                tnm += "|"

            # Make sure the final frame handle is released
            del frm
            tnm += fname
            gt.start(tnm)
            result = f(*args, **kwargs)
            gt.stop(tnm)
            return result

    else:

        @wraps(f)
        def df(*args, **kwargs):
            return f(*args, **kwargs)

    return df


def function_timer_stackskip(f):
    env = Environment.get()
    ft = env.function_timers()
    if ft:

        @wraps(f)
        def df(*args, **kwargs):
            global stack_toast_skip
            funcname = None
            if inspect.ismethod(f):
                funcname = f.__self__.__name__
            else:
                funcname = f.__qualname__
            if funcname not in stack_toast_skip:
                stack_toast_skip.add(funcname)
            return f(*args, **kwargs)

    else:

        @wraps(f)
        def df(*args, **kwargs):
            return f(*args, **kwargs)

    return df


def compute_stats(plist, full=False):
    """Compute the global timer properties.

    Given a list of dictionaries (one per process), examine the timers in each
    and compute the min / max / mean of the values and number of calls.

    Args:
        plist (list):  The list of per-process results.
        full (bool):  If True, return the full set of process values for each counter.

    Returns:
        (dict):  The stats for each timer.

    """
    # First build the set of all timers
    nproc = len(plist)
    allnames = set()
    for p in plist:
        for k in p.keys():
            allnames.add(k)
    # Repack the timer values and call count into arrays:
    seconds = dict()
    calls = dict()
    for nm in allnames:
        seconds[nm] = np.empty(nproc, dtype=np.float64)
        calls[nm] = np.empty(nproc, dtype=np.int64)
    idx = 0
    for p in plist:
        for nm in allnames:
            if nm in p:
                seconds[nm][idx] = p[nm].seconds()
                calls[nm][idx] = p[nm].calls()
            else:
                seconds[nm][idx] = -1.0
                calls[nm][idx] = -1
        idx += 1
    result = dict()
    for nm in allnames:
        result[nm] = dict()
        good = calls[nm] >= 0
        result[nm]["participating"] = np.sum(good)
        result[nm]["call_min"] = np.min(calls[nm][good])
        result[nm]["call_max"] = np.max(calls[nm][good])
        result[nm]["call_mean"] = np.mean(calls[nm][good])
        result[nm]["call_median"] = np.median(calls[nm][good])
        result[nm]["time_min"] = np.min(seconds[nm][good])
        result[nm]["time_max"] = np.max(seconds[nm][good])
        result[nm]["time_mean"] = np.mean(seconds[nm][good])
        result[nm]["time_median"] = np.median(seconds[nm][good])
        if full:
            result[nm]["calls"] = calls[nm]
            result[nm]["times"] = seconds[nm]
    return result


def gather_timers(comm=None, root=0):
    """Gather global timer information from across a communicator.

    Args:
        comm (MPI.Comm):  The communicator or None.
        root (int):  The process returning the results.

    Returns:
        (dict):  The timer stats on the root process, otherwise None.

    """
    gt = GlobalTimers.get()
    local = gt.collect()
    all = None
    if comm is None:
        all = [local]
    else:
        all = comm.gather(local, root=root)
    result = None
    if (comm is None) or (comm.rank == root):
        result = compute_stats(all)
    return result


def dump(results, path):
    """Write out timing results to a format suitable for further processing.

    Args:
        results (dict):  The results as returned by compute_stats().
        path (str):  File root name to dump.

    Returns:
        None

    """
    cols = OrderedDict(
        [
            ("Timer", "Name"),
            ("Processes", "participating"),
            ("Minimum Calls", "call_min"),
            ("Maximum Calls", "call_max"),
            ("Minimum Time", "time_min"),
            ("Maximum Time", "time_max"),
            ("Mean Time", "time_mean"),
            ("Median Time", "time_median"),
        ]
    )
    outpath = "{}.csv".format(path)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f, delimiter=",", quotechar="'")
        w.writerow(cols.keys())
        for nm in sorted(results.keys()):
            props = results[nm]
            row = [nm]
            for k, v in cols.items():
                if k == "Timer":
                    continue
                row.append(props[v])
            w.writerow(row)
    return
