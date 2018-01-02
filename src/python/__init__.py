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

import sys

# ensure mpi4py hasn't been imported yet
if 'mpi4py' in sys.modules:
    print ('Error! mpi4py module must be imported after TOAST. Exiting...')
    sys.exit(1)

# import MPI if not done already
# (should be) backwards compatible with old recommendation of:
#   from toast.mpi import MPI
# as the first line
if not 'toast.mpi' in sys.modules:
    from . import mpi

from .tests import test

from ._version import __version__

from .dist import (Comm, Data, distribute_uniform, distribute_discrete,
                   distribute_samples)

from .op import Operator

from .weather import Weather

from .ctoast import raise_error
