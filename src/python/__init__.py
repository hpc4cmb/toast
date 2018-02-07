##
##  Time Ordered Astrophysics Scalable Tools (TOAST)
##
## Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
## All rights reserved.  Use of this source code is governed by
## a BSD-style license that can be found in the LICENSE file.
##
"""
Time Ordered Astrophysics Scalable Tools (TOAST) is a software package
designed to allow the processing of data from telescopes that acquire
data as timestreams (rather than images).
"""

# Import MPI as early as possible, to prevent timeouts from the host system.
from . import mpi

from ._version import __version__

from .dist import (Comm, Data, distribute_uniform, distribute_discrete,
                   distribute_samples)

from .op import Operator

from .weather import Weather

from . import timing

# enable TiMemory signal detection
timing.enable_signal_detection()

# create an exit action function for TiMemory signal detection
def exit_action(errcode):
    tman = timing.timing_manager()
    timing.report(no_min=True)
    fname = 'toast_error_{}.out'.format(errcode)
    f = open(fname, 'w')
    f.write('{}\n'.format(tman))
    f.close()

# set the exit action function
timing.set_exit_action(exit_action)
