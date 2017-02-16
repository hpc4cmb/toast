# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os
import shutil
import tempfile

from .._version import __version__

from .ctoast import test_ctoast



def test():
    # We run tests with COMM_WORLD
    comm = MPI.COMM_WORLD
    
    outdir = "toast_test_output"

    if comm.rank == 0:
        outdir = os.path.abspath(outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    outdir = comm.bcast(outdir, root=0)

    # Run tests from the compiled library.  This separately uses
    # MPI_COMM_WORLD.
    test_ctoast()

    # Run tests on the ToastBuffer CPython class.



    return

