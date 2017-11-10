# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from . import ctoast as ctoast
import os

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class timer(object):

    def __init__(self, key, _obj = None):
        if _obj is not None:
            self.ctimer = _obj
        else:
            self.ctimer = ctoast.get_timer(key)

    def start(self):
        ctoast.timer_start(self.ctimer)

    def stop(self):
        ctoast.timer_stop(self.ctimer)

    def real_elapsed(self):
        return ctoast.timer_real_elapsed(self.ctimer)

    def system_elapsed(self):
        return ctoast.timer_system_elapsed(self.ctimer)

    def user_elapsed(self):
        return ctoast.timer_user_elapsed(self.ctimer)

class timing_manager(object):

    def __init__(self):
        self.ctiming_manager = ctoast.get_timing_manager()

    def set_output_files(self, tot_fname, avg_fname, odir = None):
        if odir is not None:
            tot_fname = '/'.join([ odir, tot_fname ])
            avg_fname = '/'.join([ odir, avg_fname ])
        ensure_directory_exists(tot_fname)
        ensure_directory_exists(avg_fname)
        ctoast.set_timing_output_files(tot_fname, avg_fname)

    def report(self):
        ctoast.report_timing()

    def size(self):
        return ctoast.timing_manager_size()

    def at(self, i):
        return timer(None, _obj=ctoast.timer_at(i))
