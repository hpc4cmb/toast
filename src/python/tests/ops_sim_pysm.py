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

from ..dist import distribute_uniform

from ._helpers import (create_outdir, create_distdata, boresight_focalplane,
    uniform_chunks)


def setup_toast_observations(self, nside):
    fixture_name = os.path.splitext(os.path.basename(__file__))[0]
    self.outdir = create_outdir(self.comm, fixture_name)

    # Create one observation per group, and each observation will have
    # a fixed number of detectors and one chunk per process.
    self.data = create_distdata(self.comm, obs_per_group=1)

    # Create a single detector.  We could use more, but that would change
    # the results compared to previous versions of these unit tests.
    self.ndet = 1
    self.dets = {
        'fake_0A' : np.array([0.0, 0.0, 0.0, 1.0]),
    }

    # Use one TOD sample per healpix pixel with the "spiral" pointing.
    self.nside = nside
    self.totsamp = 12 * self.nside**2
    self.rate = 1.0

    # Chunks - one per process.
    chunks = uniform_chunks(self.totsamp, nchunk=self.data.comm.group_size)

    # Populate the observations (one per group)

    tod = TODHpixSpiral(
        self.data.comm.comm_group,
        self.dets,
        self.totsamp,
        detranks=1,
        firsttime=0.0,
        rate=self.rate,
        nside=self.nside,
        sampsizes=chunks)

    self.data.obs[0]["tod"] = tod

    return


class OpSimPySMTest(MPITestCase):

    def setUp(self):
        setup_toast_observations(self, nside=8)

    def tearDown(self):
        del self.data


    def test_pysm_local_pix(self):
        npix = 12 * self.nside * self.nside

        local_start, nlocal = \
            distribute_uniform(npix, self.comm.size)[self.comm.rank]
        local_pixels = np.arange(nlocal, dtype=np.int64)
        local_pixels += local_start

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

        if self.comm.rank == 0:
            # The first process has the first few pixels
            np.testing.assert_almost_equal(
                local_map["sky_1a"][0, :3],
                np.array([121.40114346, 79.86737489, 77.23336053]))

        if self.comm.rank == self.comm.size - 1:
            # The last process has the last few pixels
            np.testing.assert_almost_equal(
                local_map["sky_1b"][2, -3:],
                np.array([1.57564944, -0.22345616, -3.55604102]))

        return


    def test_pysm_distrings(self):
        dist_rings = DistRings(self.data.comm.comm_world,
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

        if self.comm.rank == 0:
            # The first process has the first few pixels
            np.testing.assert_almost_equal(
                local_map["sky_1a"][0, :3],
                np.array([121.40114346, 79.86737489, 77.23336053]))

        if self.comm.rank == 0:
            # The first process has the symmetric rings at the end of the map
            np.testing.assert_almost_equal(
                local_map["sky_1b"][2, -3:],
                np.array([1.57564944, -0.22345616, -3.55604102]))

        return


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
        op_sim_pysm = OpSimPySM(comm=self.data.comm.comm_world,
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

        if tod.mpicomm.rank == 0:
            np.testing.assert_array_almost_equal(rescanned_tod[:3], expected,
                decimal=4)

        return


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
            'fwhm': 600}} # fwhm is in arcmin
        op_sim_pysm = OpSimPySM(comm=self.data.comm.comm_world,
                                out='signal',
                                pysm_model="a2,d7,f1,s3,c1",
                                focalplanes=[focalplane],
                                nside=self.nside,
                                subnpix=subnpix, localsm=localsm,
                                apply_beam=True, nest=False, units='uK_RJ')

        op_sim_pysm.exec(self.data)

        tod = self.data.obs[0]["tod"]
        rescanned_tod = tod.cache.reference("signal_fake_0A")

        I = np.array([94.77166125, 91.80626154, 95.08816366])
        Q = np.array([-5.23292836,  4.85879899, -4.80883829])
        U = np.array([-8.72476727,  8.57532387, -8.52495214])
        expected = []
        for i, (qw, uw) in enumerate([(-1, 1), (1, 1), (1, -1)]):
            expected.append(1.*I[i] + qw*0.70710678*Q[i] + uw*0.70710678*U[i])

        if tod.mpicomm.rank == 0:
            np.testing.assert_array_almost_equal(rescanned_tod[:3], expected,
                decimal=1)
        return
