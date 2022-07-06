# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Unit tests for the toast package.
"""

# If toast has not yet been imported, make sure we initialize MPI
from ..mpi import MPI
from .runner import test as run
