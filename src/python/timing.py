# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
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

def enabled():
    return ctoast.timers_enabled()

#------------------------------------------------------------------------------#

def LINE(back = 1):
    """Function that emulates __LINE__ macro"""
    return int(sys._getframe(back).f_lineno)

#------------------------------------------------------------------------------#

def FUNC(back = 1):
    """Function that emulates __FUNCTION__ macro"""
    ret = ("{}".format(sys._getframe(back).f_code.co_name))
    return ret

#------------------------------------------------------------------------------#

def FILE(back = 2, only_basename = True, use_dirname = False, noquotes = False):
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

class timer(object):
    """Class that provides interface to C++ toast::util::timer"""

    @staticmethod
    def enabled():
        return ctoast.timers_enabled()

    def __init__(self, key, _obj = None):
        if _obj is not None:
            self.ctimer = _obj
        else:
            self.ctimer = ctoast.get_timer(key)

    def start(self):
        ctoast.timer_start(self.ctimer)

    def stop(self):
        ctoast.timer_stop(self.ctimer)

    def report(self):
        ctoast.timer_report(self.ctimer)

    def real_elapsed(self):
        return ctoast.timer_real_elapsed(self.ctimer)

    def system_elapsed(self):
        return ctoast.timer_system_elapsed(self.ctimer)

    def user_elapsed(self):
        return ctoast.timer_user_elapsed(self.ctimer)

#------------------------------------------------------------------------------#

class timing_manager(object):
    """Class that provides interface to C++ toast::util::timing_manager"""

    report_fname = "timing_report.out"
    output_dir = "./"
    serial_fname = "timing_report.json"

    @staticmethod
    def enabled():
        return ctoast.timers_enabled()

    def __init__(self):
        self.ctiming_manager = ctoast.get_timing_manager()

    def set_output_file(self, fname, odir = None):
        timing_manager.report_fname = fname
        timing_manager.output_dir = odir
        if odir is not None:
            fname = os.path.join(odir, fname)
        ensure_directory_exists(fname)
        ctoast.set_timing_output_file(fname)

    def report(self):
        self.set_output_file(timing_manager.report_fname, timing_manager.output_dir)
        ctoast.report_timing()
        self.serialize(os.path.join(timing_manager.output_dir,
                                    timing_manager.serial_fname))

    def size(self):
        return ctoast.timing_manager_size()

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

    def __init__(self, key = ""):
        keyfunc = FUNC(2)
        if key != "" and key[0] != '@':
            key = ("@{}".format(key))
        self.op_line = LINE(2)
        ctoast.op_timer_instance_count(1, self.op_line)
        self.t = timer('{}{}'.format(keyfunc, key))
        self.t.start()

    def __del__(self):
        self.t.stop()
        ctoast.op_timer_instance_count(-1, -self.op_line)

#------------------------------------------------------------------------------#

def get_file_tag(fname):
    _l = basename(fname).split('.')
    _l.pop()
    return ("{}".format('_'.join(_l)))

#------------------------------------------------------------------------------#

def add_arguments(parser, fname = None):
    """Function to add default output arguments"""
    def_fname = "timing_report"
    if fname is not None:
        def_fname = '_'.join(["timing_report", get_file_tag(fname)])

    parser.add_argument('--toast-output-dir', required=False,
                        default='./', type=str, help="Output directory")
    parser.add_argument('--toast-timing-fname', required=False,
                        default=def_fname, type=str,
                        help="Filename for timing reports without directory and without suffix")

#------------------------------------------------------------------------------#

def parse_args(args):
    """Function to handle the output arguments"""
    txt_ext = "out"
    json_ext = "json"
    tman = timing_manager()
    timing_manager.report_fname = "{}.{}".format(args.toast_timing_fname, "out")
    timing_manager.serial_fname = "{}.{}".format(args.toast_timing_fname, "json")
    timing_manager.output_dir = args.toast_output_dir
    tman.clear()

#------------------------------------------------------------------------------#

def add_arguments_and_parse(parser, fname = None):
    """Combination of timing.add_arguments and timing.parse_args but returns"""
    add_arguments(parser, fname)
    args = parser.parse_args()
    parse_args(args)
    return args

#------------------------------------------------------------------------------#
