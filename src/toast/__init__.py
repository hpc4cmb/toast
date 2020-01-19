#
#  Time Ordered Astrophysics Scalable Tools (TOAST)
#
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
#
"""Time Ordered Astrophysics Scalable Tools (TOAST) is a software package
designed to allow the processing of data from telescopes that acquire
data as timestreams (rather than images).
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
    relfile = os.path.join(thisdir, "..", "..", "RELEASE")
    with open(relfile, "r") as rel:
        if __version__ is None:
            __version__ = rel.readline().rstrip()

# Namespace imports
from .mpi import Comm

from .dist import Data, distribute_uniform, distribute_discrete, distribute_samples

from .op import Operator

from .weather import Weather
