# Test whether we can import mpi4py
from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import sys

try:
    import mpi4py
    print(mpi4py.get_include())
    sys.exit(0)
except ImportError:
    print("")
    sys.exit(1)
