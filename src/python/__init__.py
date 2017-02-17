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

from .tests import test

from ._version import __version__

from .dist import (Comm, Data, distribute_uniform, distribute_discrete, 
    distribute_samples)

from .op import Operator
