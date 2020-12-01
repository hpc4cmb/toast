#
#  Time Ordered Astrophysics Scalable Tools (TOAST)
#
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
#
"""
Time Ordered Astrophysics Scalable Tools (TOAST) is a software package
designed to allow the processing of data from telescopes that acquire
data as timestreams (rather than images).

Runtime behavior of this package can be controlled by setting some
environment variables (before importing the package):

TOAST_LOGLEVEL=<value>
    * Possible values are "VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
      Default is "INFO".
    * Controls logging of both C++ and Python code.

TOAST_FUNCTIME=<value>
    * Any non-empty value will enable python function timers in many parts of the code

TOAST_TOD_BUFFER=<integer>
    * Number of elements to buffer in code where many intermediate timestream
      products are created.  Default is 1048576.

OMP_NUM_THREADS=<integer>
    * Toast uses OpenMP threading in several places and the concurrency is set by the
      usual environment variable.

MPI_DISABLE=<value>
    * Any non-empty value will disable a try block that looks for mpi4py.  Needed on
      some systems where mpi4py is available but does not work.
    * The same variable also controls the `pshmem` package used by toast.

"""
import sys
import os

# Get the package version from the libtoast environment if possible.  If this
# import fails, it is likely due to the toast package being imported prior to
# the build by setuptools (for example).
__version__ = None
try:
    from ._libtoast import Environment

    env = Environment.get()
    __version__ = env.version()
except ImportError:
    # import traceback
    # exc_type, exc_value, exc_traceback = sys.exc_info()
    # lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # print("".join(lines), flush=True)
    #
    # Just manually read the release file.
    thisdir = os.path.abspath(os.path.dirname(__file__))
    relfile = os.path.join(thisdir, "RELEASE")
    try:
        with open(relfile, "r") as rel:
            if __version__ is None:
                __version__ = rel.readline().rstrip()
    except:
        raise ImportError("Cannot read RELEASE file")

# Namespace imports
from .mpi import Comm

from .timing import Timer, GlobalTimers

from .intervals import Interval

from .observation import Observation

from .data import Data

from .config import load_config

from .instrument import Telescope, Focalplane, Site

from .instrument_sim import fake_hexagon_focalplane

from .weather import Weather

from .pixels import PixelDistribution, PixelData
