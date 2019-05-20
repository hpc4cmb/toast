# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .mpi import MPITestCase

from ..tod import TODCache

from ._helpers import create_outdir


class TODTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.dets = ["1a", "1b", "2a", "2b"]
        self.mynsamp = 10
        self.myoff = self.mynsamp * self.comm.rank
        self.totsamp = self.mynsamp * self.comm.size
        self.tod = TODCache(self.comm, self.dets, self.totsamp)
        self.rms = 10.0
        self.pntgvec = np.ravel(np.random.random((self.mynsamp, 4))).reshape(-1, 4)
        self.pflagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(
            np.uint8, copy=True
        )

        self.datavec = np.random.normal(loc=0.0, scale=self.rms, size=self.mynsamp)
        self.flagvec = np.random.uniform(low=0, high=1, size=self.mynsamp).astype(
            np.uint8, copy=True
        )
        self.tod.write_times(stamps=np.arange(self.mynsamp))
        for d in self.dets:
            self.tod.write_common_flags(local_start=0, flags=self.pflagvec)
            self.tod.write_flags(detector=d, local_start=0, flags=self.flagvec)
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
            flags = self.tod.read_flags(detector=d, local_start=0, n=self.mynsamp)
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

        common = self.tod.cache.reference("common_flags")
        for d in self.dets:
            data = self.tod.cache.reference("signal_" + d)
            flags = self.tod.cache.reference("flags_" + d)
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
        self.assertEqual(
            self.mynsamp, local_intervals[0].last - local_intervals[0].first + 1
        )
        return

    def test_local_signal(self):
        for d in self.dets:
            data = self.tod.local_signal(d)
            np.testing.assert_almost_equal(data, self.datavec)
