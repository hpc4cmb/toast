# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, use_mpi

import os
import sys
import unittest

from .mpi import MPITestRunner

from ..vis import set_backend

from .._libtoast import libtoast_tests

from . import cache as testcache
from . import timing as testtiming
from . import rng as testrng
from . import fft as testfft
from . import dist as testdist
from . import qarray as testqarray
from . import tod as testtod

from . import psd_math as testpsdmath
from . import intervals as testintervals

from . import cov as testcov

from . import ops_pmat as testopspmat

from . import ops_dipole as testopsdipole
from . import ops_simnoise as testopssimnoise

from . import ops_polyfilter as testopspolyfilter
from . import ops_groundfilter as testopsgroundfilter

from . import ops_gainscrambler as testopsgainscrambler
from . import ops_applygain as testopsapplygain

from . import ops_memorycounter as testopsmemorycounter

from . import ops_madam as testopsmadam

from . import map_satellite as testmapsatellite

from . import map_ground as testmapground

from . import binned as testbinned

from . import sim_focalplane as testsimfocalplane
from . import tod_satellite as testtodsat

from . import ops_sim_pysm as testopspysm

from ..tod import atm_available

if atm_available:
    from . import ops_sim_atm as testopsatm

from ..tod import tidas_available

if tidas_available:
    from . import tidas as testtidas

# from ..tod import spt3g_available
# if spt3g_available:
#     from . import spt3g as testspt3g
testspt3g = None
spt3g_available = False


def test(name=None, verbosity=2):
    # We run tests with COMM_WORLD if available
    comm = None
    rank = 0
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.rank

    set_backend()

    outdir = "toast_test_output"

    if rank == 0:
        outdir = os.path.abspath(outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    if comm is not None:
        outdir = comm.bcast(outdir, root=0)

    if (name is None) or (name == "libtoast"):
        # Run tests from the serial compiled library.
        libtoast_tests(list(sys.argv))

    # Run python tests.

    loader = unittest.TestLoader()
    mpirunner = MPITestRunner(comm, verbosity=verbosity, warnings="ignore")
    suite = unittest.TestSuite()

    if name is None:
        suite.addTest(loader.loadTestsFromModule(testcache))
        suite.addTest(loader.loadTestsFromModule(testtiming))
        suite.addTest(loader.loadTestsFromModule(testrng))
        suite.addTest(loader.loadTestsFromModule(testfft))
        suite.addTest(loader.loadTestsFromModule(testdist))
        suite.addTest(loader.loadTestsFromModule(testqarray))
        suite.addTest(loader.loadTestsFromModule(testtod))
        suite.addTest(loader.loadTestsFromModule(testtodsat))
        suite.addTest(loader.loadTestsFromModule(testintervals))
        suite.addTest(loader.loadTestsFromModule(testopssimnoise))
        suite.addTest(loader.loadTestsFromModule(testopsapplygain))
        suite.addTest(loader.loadTestsFromModule(testopspmat))
        suite.addTest(loader.loadTestsFromModule(testcov))
        suite.addTest(loader.loadTestsFromModule(testopsdipole))
        suite.addTest(loader.loadTestsFromModule(testopsgroundfilter))
        suite.addTest(loader.loadTestsFromModule(testsimfocalplane))
        suite.addTest(loader.loadTestsFromModule(testopspolyfilter))
        suite.addTest(loader.loadTestsFromModule(testopsmemorycounter))
        suite.addTest(loader.loadTestsFromModule(testopsgainscrambler))
        suite.addTest(loader.loadTestsFromModule(testpsdmath))
        suite.addTest(loader.loadTestsFromModule(testopsmadam))
        suite.addTest(loader.loadTestsFromModule(testmapsatellite))
        suite.addTest(loader.loadTestsFromModule(testmapground))
        suite.addTest(loader.loadTestsFromModule(testbinned))
        suite.addTest(loader.loadTestsFromModule(testopspysm))
        suite.addTest(loader.loadTestsFromModule(testopsatm))
        if tidas_available:
            suite.addTest(loader.loadTestsFromModule(testtidas))
        if spt3g_available:
            suite.addTest(loader.loadTestsFromModule(testspt3g))
    elif name != "libtoast":
        if (name == "tidas") and (not tidas_available):
            print("Cannot run TIDAS tests- package not available")
            return
        elif (name == "spt3g") and (not spt3g_available):
            print("Cannot run SPT3G tests- package not available")
            return
        else:
            modname = "toast.tests.{}".format(name)
            suite.addTest(loader.loadTestsFromModule(sys.modules[modname]))

    ret = 0
    _ret = mpirunner.run(suite)
    if not _ret.wasSuccessful():
        ret += 1

    if ret > 0:
        sys.exit(ret)

    return ret
