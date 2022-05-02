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

CUDA_MEMPOOL_FRACTION=<float>
    * If compiled with CUDA support (-DUSE_CUDA), create a memory pool that
      pre-allocates this fraction of the device memory allocated to each process.
      Note that if compiled with -DUSE_OPENACC_MEMPOOL, then OpenACC code will
      use this memory pool.  If you set this fraction to be very large, then you
      should enable this option so that regular OpenACC allocations do not
      exhaust the device memory.

"""
import os
import sys

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

from .config import create_from_config, load_config, parse_config
from .data import Data
from .instrument import Focalplane, GroundSite, SpaceSite, Telescope
from .instrument_sim import fake_hexagon_focalplane
from .intervals import Interval
from .job import job_group_size

# Namespace imports
from .mpi import Comm, get_world
from .observation import Observation
from .pixels import PixelData, PixelDistribution
from .timing import GlobalTimers, Timer
from .weather import Weather
