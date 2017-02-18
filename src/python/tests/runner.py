# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os
import unittest

from .._version import __version__

from .mpi import MPITestRunner

from .ctoast import test_ctoast

from . import cbuffer as testcbuffer
from . import cache as testcache
from . import rng as testrng
from . import dist as testdist
from . import qarray as testqarray


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

    # Run python tests.

    loader = unittest.TestLoader()
    mpirunner = MPITestRunner(verbosity=2)
    suite = unittest.TestSuite()

    suite.addTest( loader.loadTestsFromModule(testcbuffer) )
    suite.addTest( loader.loadTestsFromModule(testcache) )
    suite.addTest( loader.loadTestsFromModule(testrng) )
    suite.addTest( loader.loadTestsFromModule(testdist) )
    suite.addTest( loader.loadTestsFromModule(testqarray) )


    mpirunner.run(suite)


    return

