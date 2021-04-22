# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, use_mpi

import os
import sys
import shutil

import unittest

from .mpi import MPITestRunner

from ..vis import set_matplotlib_backend

from .._libtoast import libtoast_tests

from . import env as test_env
from . import timing as test_timing
from . import rng as test_rng
from . import fft as test_fft
from . import healpix as test_healpix
from . import qarray as test_qarray
from . import instrument as test_instrument
from . import intervals as test_intervals
from . import pixels as test_pixels
from . import weather as test_weather

from . import observation as test_observation

from . import dist as test_dist

from . import config as test_config

from . import ops_sim_satellite as test_ops_sim_satellite
from . import ops_memory_counter as test_ops_memory_counter
from . import ops_pointing_healpix as test_ops_pointing_healpix
from . import ops_sim_tod_noise as test_ops_sim_tod_noise
from . import ops_sim_tod_dipole as test_ops_sim_tod_dipole
from . import ops_sim_tod_conviqt as test_ops_sim_tod_conviqt
from . import ops_sim_tod_totalconvolve as test_ops_sim_tod_totalconvolve
from . import ops_mapmaker_utils as test_ops_mapmaker_utils
from . import ops_mapmaker_binning as test_ops_mapmaker_binning
from . import ops_mapmaker_solve as test_ops_mapmaker_solve
from . import ops_mapmaker as test_ops_mapmaker
from . import ops_scan_map as test_ops_scan_map
from . import ops_scan_healpix as test_ops_scan_healpix
from . import ops_madam as test_ops_madam
from . import ops_gainscrambler as test_ops_grainscrambler


from . import covariance as test_covariance

from . import template_amplitudes as test_template_amplitudes
from . import template_offset as test_template_offset
from . import template_fourier2d as test_template_fourier2d
from . import template_subharmonic as test_template_subharmonic


#
# from . import psd_math as testpsdmath
#
# from . import ops_dipole as testopsdipole
# from . import ops_sim_sss as testopssimsss
#
# from . import ops_polyfilter as testopspolyfilter
# from . import ops_groundfilter as testopsgroundfilter
#
# from . import ops_gainscrambler as testopsgainscrambler
# from . import ops_applygain as testopsapplygain
#
# from . import map_satellite as testmapsatellite
#
# from . import map_ground as testmapground
#
# from . import sim_focalplane as testsimfocalplane
# from . import tod_satellite as testtodsat
#
# from ..todmap import pysm
#
# if pysm is not None:
#     from . import ops_sim_pysm as testopspysm
#
# from . import ops_sim_atm as testopsatm
#
# from ..tod import tidas_available
#
# # if tidas_available:
# #     from . import tidas as testtidas
# testtidas = None
# tidas_available = False
#
# # from ..tod import spt3g_available
# # if spt3g_available:
# #     from . import spt3g as testspt3g
# testspt3g = None
# spt3g_available = False


def test(name=None, verbosity=2):
    # We run tests with COMM_WORLD if available
    comm = None
    rank = 0
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.rank

    # set_matplotlib_backend(backend="agg")

    outdir = "toast_test_output"

    if rank == 0:
        outdir = os.path.abspath(outdir)
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)

    if comm is not None:
        outdir = comm.bcast(outdir, root=0)

    if (name is None) or (name == "libtoast"):
        # Run tests from the serial compiled library.
        libtoast_tests(list(sys.argv))

    # Run python tests.

    loader = unittest.TestLoader()
    mpirunner = MPITestRunner(verbosity=verbosity, warnings="ignore")
    suite = unittest.TestSuite()

    if name is None:
        suite.addTest(loader.loadTestsFromModule(test_env))
        if not (("CONDA_BUILD" in os.environ) or ("CIBUILDWHEEL" in os.environ)):
            # When doing a conda build on CI services in containers
            # the timing information is not accurate and these tests
            # fail.
            suite.addTest(loader.loadTestsFromModule(test_timing))
        suite.addTest(loader.loadTestsFromModule(test_rng))
        suite.addTest(loader.loadTestsFromModule(test_fft))
        suite.addTest(loader.loadTestsFromModule(test_healpix))
        suite.addTest(loader.loadTestsFromModule(test_qarray))
        suite.addTest(loader.loadTestsFromModule(test_intervals))
        suite.addTest(loader.loadTestsFromModule(test_instrument))
        suite.addTest(loader.loadTestsFromModule(test_pixels))
        suite.addTest(loader.loadTestsFromModule(test_weather))

        suite.addTest(loader.loadTestsFromModule(test_observation))
        suite.addTest(loader.loadTestsFromModule(test_dist))
        suite.addTest(loader.loadTestsFromModule(test_config))

        suite.addTest(loader.loadTestsFromModule(test_ops_sim_satellite))
        suite.addTest(loader.loadTestsFromModule(test_ops_memory_counter))
        suite.addTest(loader.loadTestsFromModule(test_ops_pointing_healpix))
        suite.addTest(loader.loadTestsFromModule(test_ops_sim_tod_noise))
        suite.addTest(loader.loadTestsFromModule(test_ops_sim_tod_dipole))
        suite.addTest(loader.loadTestsFromModule(test_ops_sim_tod_conviqt))
        suite.addTest(loader.loadTestsFromModule(test_ops_sim_tod_totalconvolve))
        suite.addTest(loader.loadTestsFromModule(test_ops_mapmaker_utils))
        suite.addTest(loader.loadTestsFromModule(test_ops_mapmaker_binning))
        suite.addTest(loader.loadTestsFromModule(test_ops_mapmaker_solve))
        suite.addTest(loader.loadTestsFromModule(test_ops_mapmaker))
        suite.addTest(loader.loadTestsFromModule(test_ops_scan_map))
        suite.addTest(loader.loadTestsFromModule(test_ops_scan_healpix))
        suite.addTest(loader.loadTestsFromModule(test_ops_madam))
        # suite.addTest(loader.loadTestsFromModule(test_ops_gainscrambler))

        suite.addTest(loader.loadTestsFromModule(test_covariance))

        suite.addTest(loader.loadTestsFromModule(test_template_amplitudes))
        suite.addTest(loader.loadTestsFromModule(test_template_offset))
        suite.addTest(loader.loadTestsFromModule(test_template_fourier2d))
        suite.addTest(loader.loadTestsFromModule(test_template_subharmonic))

        #
        # suite.addTest(loader.loadTestsFromModule(testopssimsss))

        # suite.addTest(loader.loadTestsFromModule(testopsapplygain))

        # suite.addTest(loader.loadTestsFromModule(testopsdipole))
        # suite.addTest(loader.loadTestsFromModule(testopsgroundfilter))
        # suite.addTest(loader.loadTestsFromModule(testsimfocalplane))
        # suite.addTest(loader.loadTestsFromModule(testopspolyfilter))
        # suite.addTest(loader.loadTestsFromModule(testopsgainscrambler))

        # suite.addTest(loader.loadTestsFromModule(testpsdmath))

        # suite.addTest(loader.loadTestsFromModule(testmapsatellite))
        # suite.addTest(loader.loadTestsFromModule(testmapground))

        # suite.addTest(loader.loadTestsFromModule(testopsatm))
        #
        # # These tests segfault locally.  Re-enable once we are doing bandpass
        # # integration on on the fly.
        # # if pysm is not None:
        # #     suite.addTest(loader.loadTestsFromModule(testopspysm))
        #
        # if tidas_available:
        #     suite.addTest(loader.loadTestsFromModule(testtidas))
        # if spt3g_available:
        #     suite.addTest(loader.loadTestsFromModule(testspt3g))
    elif name != "libtoast":
        # if (name == "tidas") and (not tidas_available):
        #     print("Cannot run TIDAS tests- package not available")
        #     return
        # elif (name == "spt3g") and (not spt3g_available):
        #     print("Cannot run SPT3G tests- package not available")
        #     return
        # else:
        modname = "toast.tests.{}".format(name)
        if modname not in sys.modules:
            result = '"{}" is not a valid test.  Try'.format(name)
            for name in sys.modules:
                if name.startswith("toast.tests."):
                    result += '\n  - "{}"'.format(name.replace("toast.tests.", ""))
            result += "\n"
            raise RuntimeError(result)
        suite.addTest(loader.loadTestsFromModule(sys.modules[modname]))

    ret = 0
    _ret = mpirunner.run(suite)
    if not _ret.wasSuccessful():
        ret += 1

    if ret > 0:
        sys.exit(ret)

    return ret
