# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from . import ctoast as ctoast

import sys
import os
import argparse

from .mpi import MPI

from os.path import dirname
from os.path import basename
from os.path import join

rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()


#------------------------------------------------------------------------------#
def default_max_depth():
    return 65536


#------------------------------------------------------------------------------#
def enabled():
    return ctoast.timers_enabled()


#------------------------------------------------------------------------------#
def toggle(on_or_off):
    return ctoast.timers_toggle(on_or_off)


#------------------------------------------------------------------------------#
def LINE(back=1):
    """Function that emulates __LINE__ macro"""
    return int(sys._getframe(back).f_lineno)


#------------------------------------------------------------------------------#
def FUNC(back=1):
    """Function that emulates __FUNCTION__ macro"""
    ret = ("{}".format(sys._getframe(back).f_code.co_name))
    return ret


#------------------------------------------------------------------------------#
def FILE(back=2, only_basename=True, use_dirname=False, noquotes=False):
    """Function that emulates __FILE__ macro"""
    ret = None
    def get_fcode():
        return sys._getframe(back).f_code

    if only_basename:
        if use_dirname:
            ret = ("{}".format(join(basename(dirname(get_fcode().co_filename)),
              basename(get_fcode().co_filename))))
        else:
            ret = ("{}".format(basename(get_fcode().co_filename)))
    else:
        ret = ("{}".format(get_fcode().co_filename))

    if noquotes is False:
        ret = ("'{}'".format(ret))

    return ret


#------------------------------------------------------------------------------#
def ensure_directory_exists(file_path):
    """Function to make a directory if it doesn't exist"""

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)


#------------------------------------------------------------------------------#
class simple_timer(object):
    """Class that provides simple interface to C++ toast::util::timer"""

    def __init__(self, key=None, obj=None):
        if obj is not None:
            self._ctimer = obj
        else:
            self._ctimer = ctoast.get_simple_timer(key)

    def __del__(self):
        if self._ctimer is not None:
            ctoast.delete_simple_timer(self._ctimer)

    def start(self):
        if self._ctimer is not None:
            ctoast.timer_start(self._ctimer)

    def stop(self):
        if self._ctimer is not None:
            ctoast.timer_stop(self._ctimer)

    def report(self):
        if self._ctimer is not None:
            ctoast.timer_report(self._ctimer)

    def real_elapsed(self):
        if self._ctimer is not None:
            return ctoast.timer_real_elapsed(self._ctimer)
        return 0.0

    def system_elapsed(self):
        if self._ctimer is not None:
            return ctoast.timer_system_elapsed(self._ctimer)
        return 0.0

    def user_elapsed(self):
        if self._ctimer is not None:
            return ctoast.timer_user_elapsed(self._ctimer)
        return 0.0


#------------------------------------------------------------------------------#
class timer(object):
    """Class that provides interface to C++ toast::util::timer"""

    @staticmethod
    def enabled():
        return ctoast.timers_enabled()

    def __init__(self, key=None, obj=None):
        if obj is not None:
            self._ctimer = obj
        else:
            if enabled():
                self._ctimer = ctoast.get_timer(key)
            else:
                self._ctimer = None

    def start(self):
        if self._ctimer is not None:
            ctoast.timer_start(self._ctimer)

    def stop(self):
        if self._ctimer is not None:
            ctoast.timer_stop(self._ctimer)

    def report(self):
        if self._ctimer is not None:
            ctoast.timer_report(self._ctimer)

    def real_elapsed(self):
        if self._ctimer is not None:
            return ctoast.timer_real_elapsed(self._ctimer)
        return 0.0

    def system_elapsed(self):
        if self._ctimer is not None:
            return ctoast.timer_system_elapsed(self._ctimer)
        return 0.0

    def user_elapsed(self):
        if self._ctimer is not None:
            return ctoast.timer_user_elapsed(self._ctimer)
        return 0.0


#------------------------------------------------------------------------------#
class timing_manager(object):
    """Class that provides interface to C++ toast::util::timing_manager"""

    report_fname = "timing_report.out"
    output_dir = "./"
    serial_fname = "timing_report.json"
    serial_report = True
    max_timer_depth = ctoast.timing_manager_max_depth()

    @staticmethod
    def enabled():
        return ctoast.timers_enabled()

    def __init__(self):
        self._ctiming_manager = ctoast.get_timing_manager()

    def set_output_file(self, fname, odir=None, serial=None):
        timing_manager.report_fname = fname
        timing_manager.output_dir = odir
        if odir is not None:
            fname = os.path.join(odir, fname)
        ensure_directory_exists(fname)
        if self.size() > 0:
            ctoast.set_timing_output_file(fname)
        if serial is not None:
            timing_manager.serial_fname = serial

    def report(self):
        if self.size() > 0:
            self.set_output_file(timing_manager.report_fname, timing_manager.output_dir)
            ctoast.report_timing()
            if timing_manager.serial_report:
                self.serialize(os.path.join(timing_manager.output_dir,
                                            timing_manager.serial_fname))

    def size(self):
        return ctoast.timing_manager_size()

    def at(self, i):
        return timer(None, obj=ctoast.get_timer_at(i))

    def clear(self):
        ctoast.timing_manager_clear()

    def serialize(self, fname):
        ensure_directory_exists(fname)
        ctoast.serialize_timing_manager(fname)


#------------------------------------------------------------------------------#
class auto_timer(object):
    """Class that provides same utilities as toast::util::auto_timer"""

    @staticmethod
    def enabled():
        return ctoast.timers_enabled()

    def __init__(self, key=""):
        self._toggled = False
        keyfunc = FUNC(2)
        if key != "" and key[0] != '@':
            key = ("@{}".format(key))
        self._op_line = LINE(2)
        # increment the instance count
        ctoast.op_timer_instance_count(1, self._op_line)
        # turn off if exceed count and only when enabled
        if (enabled() and
        ctoast.get_timer_instance_count() + 1 > timing_manager.max_timer_depth):
            toggle(False)
            self._toggled = True
        # timer key
        self.t = timer('{}{}'.format(keyfunc, key))
        self.t.start()

    def __del__(self):
        # stop timer
        self.t.stop()
        # decrement instance count
        ctoast.op_timer_instance_count(-1, -self._op_line)
        if not enabled() and self._toggled:
            toggle(True)
            self._toggled = False


#------------------------------------------------------------------------------#
def set_max_depth(_depth):
    """Function for setting the max depth of timers"""
    ctoast.timing_manager_set_max_depth(_depth)
    timing_manager.max_timer_depth = ctoast.timing_manager_max_depth()


#------------------------------------------------------------------------------#
def max_depth():
    """Function for getting the max depth of timers"""
    return ctoast.timing_manager_max_depth()


#------------------------------------------------------------------------------#
def get_file_tag(fname):
    _l = basename(fname).split('.')
    _l.pop()
    return ("{}".format('_'.join(_l)))


#------------------------------------------------------------------------------#
def add_arguments(parser, fname=None):
    """Function to add default output arguments"""
    def_fname = "timing_report"
    if fname is not None:
        def_fname = '_'.join(["timing_report", get_file_tag(fname)])

    parser.add_argument('--toast-output-dir', required=False,
                        default='./', type=str, help="Output directory")
    parser.add_argument('--toast-timing-fname', required=False,
                        default=def_fname, type=str,
                        help="Filename for timing report w/o directory and w/o suffix")
    parser.add_argument('--disable-timers', required=False, action='store_false',
                        dest='use_timers', help="Disable timers for script")
    parser.add_argument('--enable-timers', required=False, action='store_true',
                        dest='use_timers', help="Enable timers for script")
    parser.add_argument('--disable-timer-serialization',
                        required=False, action='store_false',
                        dest='serial_report', help="Disable serialization for timers")
    parser.add_argument('--enable-timer-serialization',
                        required=False, action='store_true',
                        dest='serial_report', help="Enable serialization for timers")
    parser.add_argument('--max-timer-depth', help="Maximum timer depth", type=int,
                        default=65536)

    parser.set_defaults(use_timers=True)
    parser.set_defaults(serial_report=False)


#------------------------------------------------------------------------------#
def parse_args(args):
    """Function to handle the output arguments"""
    tman = timing_manager()
    timing_manager.report_fname = "{}.{}".format(args.toast_timing_fname, "out")
    timing_manager.serial_fname = "{}.{}".format(args.toast_timing_fname, "json")
    timing_manager.output_dir = args.toast_output_dir
    tman.clear()
    toggle(args.use_timers)
    timing_manager.serial_report = args.serial_report
    set_max_depth(args.max_timer_depth)


#------------------------------------------------------------------------------#
def add_arguments_and_parse(parser, fname=None):
    """Combination of timing.add_arguments and timing.parse_args but returns"""
    add_arguments(parser, fname)
    args = parser.parse_args()
    parse_args(args)
    return args


#------------------------------------------------------------------------------#
