##
# Copyright (c) 2017-2020, all rights reserved.  Use of this source code
# is governed by a BSD license that can be found in the top-level
# LICENSE file.
##
"""Parallel shared memory tools.

This package contains tools for using synchronized shared memory across nodes
and implementing communicator-wide MUTEX locks.

"""

__version__ = "0.1.0"

# Namespace imports

from .shmem import MPIShared
from .locking import MPILock
from .test import run as test
