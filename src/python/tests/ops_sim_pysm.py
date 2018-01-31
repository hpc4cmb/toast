# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import os
import numpy as np

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_map import *
from ..map.pixels import *
from ..map.rings import DistRings
from ..map import PySMSky
from ..todmap.pysm_operator import OpSimPySM


def setup_toast_observations(self, nside):
    self.outdir = "toast_test_output"
    if self.comm.rank == 0:
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

    # Note: self.comm is set by the test infrastructure
    self.worldsize = self.comm.size
    if (self.worldsize >= 2):
        self.groupsize = int( self.worldsize / 2 )
        self.ngroup = 2
    else:
        self.groupsize = 1
        self.ngroup = 1
    self.toastcomm = Comm(world=self.comm, groupsize=self.groupsize)
    self.data = Data(self.toastcomm)

    spread = 0 * np.pi / 180.0
    angterm = np.cos(spread / 2.0)
    axiscoef = np.sin(spread / 2.0)

    self.dets = {
        'fake_0A' : np.array([axiscoef, 0.0, 0.0, angterm]),
    }

    self.nside = nside
    self.totsamp = 12 * self.nside**2

    # Every process group creates a single observation that will
    # have the same data.
    nobs = 1

    for i in range(nobs):
        # create the TOD for this observation.  This fake TOD just has
        # boresight pointing that spirals around the healpix rings.

        tod = TODHpixSpiral(
            self.toastcomm.comm_group,
            self.dets,
            self.totsamp,
            rate=1.0,
            nside=self.nside,
            detranks=self.toastcomm.group_size,
        )

        ob = {}
        ob['name'] = 'test'
        ob['id'] = 0
        ob['tod'] = tod
        ob['intervals'] = None
        ob['baselines'] = None
        ob['noise'] = None

        self.data.obs.append(ob)


class OpSimPySMTest(MPITestCase):

    def setUp(self):
        setup_toast_observations(self, nside=8)

    def tearDown(self):
        del self.data


    def test_pysm_local_pix(self):
        start = MPI.Wtime()

        npix = 12 * self.nside * self.nside

        npix_local = int(np.ceil(npix / float(self.toastcomm.comm_world.size)))
        local_pixels = np.arange(
            self.toastcomm.comm_world.rank * npix_local,
            min(self.toastcomm.comm_world.rank*npix_local + npix_local, npix)
        )

        # construct the PySM operator.  Pass in information needed by PySM...

        pysm_sky_config = {
            'synchrotron': "s1",
            'dust': "d8",
            'freefree': "f1",
            'cmb': "c1",
            'ame': "a1",
        }
        bandpasses = {
                "1a": (np.linspace(20, 25, 10), np.ones(10)),
                "1b": (np.linspace(21, 26, 10), np.ones(10)),
                "2a": (np.linspace(18, 23, 10), np.ones(10)),
                "2b": (np.linspace(19, 24, 10), np.ones(10)),
        }
        op = PySMSky(local_pixels=local_pixels, nside=self.nside,
                     pysm_sky_config=pysm_sky_config, units='uK_RJ')
        local_map = {} # it should be Cache in production
        op.exec(local_map, out="sky", bandpasses=bandpasses)

        # Now we have timestreams in the cache.  We could compare the
        # timestream values or we could make a binned map and look at those
        # values.

        np.testing.assert_almost_equal(
            local_map["sky_1a"][0, :3],
            np.array([121.40114346, 79.86737489, 77.23336053]))

        np.testing.assert_almost_equal(
            local_map["sky_1b"][2, -3:],
            np.array([1.57564944, -0.22345616, -3.55604102]))

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("test_pysm took {:.3f} s".format(elapsed))

    def test_pysm_distrings(self):
        start = MPI.Wtime()

        dist_rings = DistRings(self.toastcomm.comm_world,
                               nside = self.nside,
                               nnz = 3)

        # construct the PySM operator.  Pass in information needed by PySM...

        pysm_sky_config = {
            'synchrotron': "s1",
            'dust': "d8",
            'freefree': "f1",
            'cmb': "c1",
            'ame': "a1",
        }
        bandpasses = {
                "1a": (np.linspace(20, 25, 10), np.ones(10)),
                "1b": (np.linspace(21, 26, 10), np.ones(10)),
                "2a": (np.linspace(18, 23, 10), np.ones(10)),
                "2b": (np.linspace(19, 24, 10), np.ones(10)),
        }
        op = PySMSky(local_pixels=dist_rings.local_pixels, nside=self.nside,
                     pysm_sky_config=pysm_sky_config, units='uK_RJ')
        local_map = {} # it should be Cache in production
        op.exec(local_map, out="sky", bandpasses=bandpasses)

        # Now we have timestreams in the cache.  We could compare the
        # timestream values or we could make a binned map and look at those
        # values.

        np.testing.assert_almost_equal(
            local_map["sky_1a"][0, :3],
            np.array([121.40114346, 79.86737489, 77.23336053]))

        np.testing.assert_almost_equal(
            local_map["sky_1b"][2, -3:],
            np.array([1.57564944, -0.22345616, -3.55604102]))

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("test_pysm took {:.3f} s".format(elapsed))

    def test_op_pysm_nosmooth(self):
        # expand the pointing into a low-res pointing matrix
        pointing = OpPointingHpix(nside=self.nside, nest=False, mode="IQU")
        pointing.exec(self.data)

        # Get locally hit pixels.  Only do this if the PySM operator
        # needs local pixels...
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)
        #subnpix = np.floor_divide(self.nside, 4)
        #localsm = np.unique(np.floor_divide(localpix, subnpix))

        subnpix, localsm = 64, np.arange(12)
        focalplane = {'fake_0A': {
            'bandcenter_ghz': 22.5,
            'bandwidth_ghz': 5,
            'fmin': 1e-05,
            'fwhm': 0.16666666666666666}}
        op_sim_pysm = OpSimPySM(comm=self.toastcomm.comm_world,
                                out='signal',
                                pysm_model="a2,d7,f1,s3,c1",
                                focalplanes=[focalplane],
                                nside=self.nside,
                                subnpix=subnpix, localsm=localsm,
                                apply_beam=False, nest=False, units='uK_RJ')

        op_sim_pysm.exec(self.data)
        tod = self.data.obs[0]["tod"]
        rescanned_tod = tod.cache.reference("signal_fake_0A")
        pix = tod.cache.reference("pixels_fake_0A")
        weights = tod.cache.reference("weights_fake_0A")

        # compare with maps computes with PySM standalone running
        # the test_mpi.py script from the PySM repository

        I = np.array([121.76090504, 80.14861367, 77.58032105])
        Q = np.array([-7.91582211, 1.03253469, -5.49429043])
        U = np.array([-6.20504973, 7.71503699, -6.17427374])
        expected = []
        for i, (qw, uw) in enumerate([(-1, 1), (1, 1), (1, -1)]):
            expected.append(1.*I[i] + qw*0.70710678*Q[i] + uw*0.70710678*U[i])
        np.testing.assert_array_almost_equal(rescanned_tod[:3], expected,
                                             decimal=4)

class OpSimPySMTestSmooth(MPITestCase):

    def setUp(self):
        setup_toast_observations(self, nside=64)

    def test_op_pysm_smooth(self):
        # expand the pointing into a low-res pointing matrix
        pointing = OpPointingHpix(nside=self.nside, nest=False, mode="IQU")
        pointing.exec(self.data)

        subnpix, localsm = self.nside**2, np.arange(12)
        focalplane = {'fake_0A': {
            'bandcenter_ghz': 22.5,
            'bandwidth_ghz': 5,
            #'fwhm': 30*0.16666666666666666}}
            'fwhm': 10}}
        op_sim_pysm = OpSimPySM(comm=self.toastcomm.comm_world,
                                out='signal',
                                pysm_model="a2,d7,f1,s3,c1",
                                focalplanes=[focalplane],
                                nside=self.nside,
                                subnpix=subnpix, localsm=localsm,
                                apply_beam=True, nest=False, units='uK_RJ')

        op_sim_pysm.exec(self.data)
        rescanned_tod = self.data.obs[0]["tod"].cache.reference("signal_fake_0A")

        I = np.array([94.77166125, 91.80626154, 95.08816366])
        Q = np.array([-5.23292836,  4.85879899, -4.80883829])
        U = np.array([-8.72476727,  8.57532387, -8.52495214])
        expected = []
        for i, (qw, uw) in enumerate([(-1, 1), (1, 1), (1, -1)]):
            expected.append(1.*I[i] + qw*0.70710678*Q[i] + uw*0.70710678*U[i])
        np.testing.assert_array_almost_equal(rescanned_tod[:3], expected,
                                             decimal=1)
