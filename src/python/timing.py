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

#------------------------------------------------------------------------------#

def FUNC(back = 1):
    """Function that emulates __FUNCTION__ macro"""
    ret = ("%s" % (sys._getframe(back).f_code.co_name))
    return ret

#------------------------------------------------------------------------------#

def FILE(back = 2, only_basename = True, use_dirname = False, noquotes = False):
    """Function that emulates __FILE__ macro"""
    ret = None
    def get_fcode():
        return sys._getframe(back).f_code

    if only_basename:
        if use_dirname:
            ret = ("%s" % (join(basename(dirname(get_fcode().co_filename)),
                             basename(get_fcode().co_filename))))
        else:
            ret = ("%s" % (basename(get_fcode().co_filename)))
    else:
        ret = ("%s" % (get_fcode().co_filename))

    if noquotes is False:
        ret = ("'%s'" % (ret))

    return ret

#------------------------------------------------------------------------------#

def ensure_directory_exists(file_path):
    """Function to make a directory if it doesn't exist"""

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#------------------------------------------------------------------------------#

class timer(object):
    """Class that provides interface to C++ toast::util::timer"""

    def __init__(self, key, _obj = None):
        if _obj is not None:
            self.ctimer = _obj
        else:
            self.ctimer = ctoast.get_timer(("[pyc] %s" % key))

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
        if rank == 0:
            ctoast.report_timing()

    def size(self):
        return ctoast.timing_manager_size()

    def at(self, i):
        return timer(None, _obj=ctoast.timer_at(i))

    def clear(self):
        ctoast.timing_manager_clear()

#------------------------------------------------------------------------------#

class auto_timer(object):
    """Class that provides same utilities as toast::util::auto_timer"""

    def __init__(self, key = ""):
        keyfunc = FUNC(2)
        if key != "" and key[0] != '@':
            key = ("@%s" % (key))
        self.t = timer('%s%s' % (keyfunc, key))
        self.t.start()

    def __del__(self):
        self.t.stop()

#------------------------------------------------------------------------------#

def get_file_tag(fname):
    _l = basename(fname).split('.')
    _l.pop()
    return ("%s" % ('_'.join(_l)))

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
    if rank == 0:
        tot_fname = ("%s_tot.out" % args.toast_timing_fname)
        avg_fname = ("%s_avg.out" % args.toast_timing_fname)
        tman = timing_manager()
        tman.set_output_files(tot_fname, avg_fname, args.toast_output_dir)

#------------------------------------------------------------------------------#

def add_arguments_and_parse(parser, fname = None):
    """Combination of timing.add_arguments and timing.parse_args but returns"""
    add_arguments(parser, fname)
    args = parser.parse_args()
    parse_args(args)
    return args

#------------------------------------------------------------------------------#
