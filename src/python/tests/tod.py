# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

from ..dist import *
from ..tod.tod import *

from ..tod.applygain import *
from ._helpers import create_outdir, create_distdata, boresight_focalplane

class TODTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

        # Note: self.comm is set by the test infrastructure
        self.dets = ["1a", "1b", "2a", "2b"]
        self.mynsamp = 10
        self.myoff = self.mynsamp * self.comm.rank
        self.totsamp = self.mynsamp * self.comm.size
        self.tod = TODCache(self.comm, self.dets, self.totsamp)
        self.rms = 10.0
        self.pntgvec = np.ravel(
            np.random.random((self.mynsamp, 4))).reshape(-1,4)
        self.pflagvec = np.random.uniform(
            low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)

        self.datavec = np.random.normal(loc=0.0, scale=self.rms,
                                        size=self.mynsamp)
        self.flagvec = np.random.uniform(
            low=0, high=1, size=self.mynsamp).astype(np.uint8, copy=True)
        self.tod.write_times(stamps=np.arange(self.mynsamp))
        for d in self.dets:
            self.tod.write_common_flags(local_start=0, flags=self.pflagvec)
            self.tod.write_flags(detector=d, local_start=0,
                                     flags=self.flagvec)
            self.tod.write(detector=d, local_start=0, data=self.datavec)
            self.tod.write_pntg(detector=d, local_start=0, data=self.pntgvec)

    def tearDown(self):
        pass


    def test_props(self):
        self.assertEqual(sorted(self.tod.detectors), sorted(self.dets))
        self.assertEqual(sorted(self.tod.local_dets), sorted(self.dets))
        self.assertEqual(self.tod.total_samples, self.totsamp)
        self.assertEqual(self.tod.local_samples[0], self.myoff)
        self.assertEqual(self.tod.local_samples[1], self.mynsamp)
        return


    def test_read(self):
        common = self.tod.read_common_flags(local_start=0, n=self.mynsamp)
        for d in self.dets:
            data = self.tod.read(detector=d, local_start=0, n=self.mynsamp)
            flags = self.tod.read_flags(detector=d, local_start=0,
                                        n=self.mynsamp)
            np.testing.assert_equal(flags, self.flagvec)
            np.testing.assert_equal(common, self.pflagvec)
            np.testing.assert_almost_equal(data, self.datavec)
        return


    def test_cached_read(self):
        common = self.tod.local_common_flags()
        for d in self.dets:
            data = self.tod.local_signal(d)
            flags = self.tod.local_flags(d)
            np.testing.assert_equal(flags, self.flagvec)
            np.testing.assert_equal(common, self.pflagvec)
            np.testing.assert_almost_equal(data, self.datavec)

        # Then we can use the cached TOD

        common = self.tod.cache.reference('common_flags')
        for d in self.dets:
            data = self.tod.cache.reference('signal_'+d)
            flags = self.tod.cache.reference('flags_'+d)
            np.testing.assert_equal(flags, self.flagvec)
            np.testing.assert_equal(common, self.pflagvec)
            np.testing.assert_almost_equal(data, self.datavec)

        return


    def test_read_pntg(self):
        for d in self.dets:
            pntg = self.tod.read_pntg(detector=d, local_start=0, n=self.mynsamp)
            np.testing.assert_almost_equal(pntg, self.pntgvec)
        return


    def test_local_intervals(self):
        local_intervals = self.tod.local_intervals(None)
        self.assertEqual(self.mynsamp,
                         local_intervals[0].last - local_intervals[0].first + 1)
        return


    def test_local_signal(self):
        for d in self.dets:
            data = self.tod.local_signal(d)
            np.testing.assert_almost_equal(data, self.datavec)

class TestApplyGain(MPITestCase):

    def setUp(self):
        self.gain = {
                "TIME":np.arange(10),
                "1a":2*np.ones(10, dtype=np.float32),
                "1b":3*np.ones(10, dtype=np.float32)
                }
        self.gain["1b"][5:] = 0
    
    def test_write_calibration_file(self):
        write_calibration_file("test_cal.fits", self.gain)

    def test_op_applygain(self):

        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "dipole")
        if self.comm.rank == 0:
            if not os.path.isdir(self.mapdir):
                os.mkdir(self.mapdir)

        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = create_distdata(self.comm, obs_per_group=1)

        # Two detectors, default properties
        self.ndet = 2
        self.dets = ["1a", "1b"]
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = \
            boresight_focalplane(self.ndet)

        # Pick some number of samples per observation
        self.samples_per_obs = 10
        time_stamps = np.arange(self.samples_per_obs)

        # (there is only one observation per group- see above)
        self.data.obs[0]['tod'] = TODCache(self.data.comm.comm_group,
            self.dets, self.samples_per_obs)

        # generate input data of ones for each of the 2 channels and 10 timestamps
        # from 1 to 10, no need for pointing

        for obs in self.data.obs:
            tod = obs['tod']
            tod.write_times(stamps=time_stamps)
            for det in tod.local_dets:
                tod.write(detector=det, data=np.ones(tod.local_samples[1]))


        op_apply_gain = OpApplyGain(self.gain, name="toast_tod_detdata")
        op_apply_gain.exec(self.data)

        # compare calibrated timelines

        for obs in self.data.obs:
            tod = obs['tod']
            for det in tod.local_dets:
                tod.read(detector=det, name="toast_tod_detdata")
