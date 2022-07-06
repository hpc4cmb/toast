# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..rng import random, random_multi
from ..utils import AlignedU64
from .mpi import MPITestCase


class RNGTest(MPITestCase):
    def setUp(self):
        # data
        self.size = 11
        self.counter = [1357111317, 888118218888]
        self.key = [3405692589, 3131965165]
        self.counter00 = [0, 0]
        self.key00 = [0, 0]
        self.nstream = 100

        # C test output with counter=[1357111317,888118218888] and
        # key=[3405692589,3131965165]
        # self.array_gaussian = np.array([-0.602799, 2.141513, -0.433604,
        # 0.493275, -0.037459, -0.926340, -0.536562, -0.064849, -0.662582,
        # -1.024292, -0.170119], np.float64)

        self.array_m11 = np.array(
            [
                -0.951008,
                0.112014,
                -0.391117,
                0.858437,
                -0.232332,
                -0.929797,
                0.513278,
                -0.722889,
                -0.439833,
                0.814677,
                0.466897,
            ],
            np.float64,
        )
        self.array_01 = np.array(
            [
                0.524496,
                0.056007,
                0.804442,
                0.429218,
                0.883834,
                0.535102,
                0.256639,
                0.638556,
                0.780084,
                0.407338,
                0.233448,
            ],
            np.float64,
        )
        self.array_uint64 = np.array(
            [
                9675248043493244317,
                1033143684219887964,
                14839328367301273822,
                7917682351778602270,
                16303863741333868668,
                9870884412429777903,
                4734154306332135586,
                11779270208507399991,
                14390002533568630569,
                7514066637753215609,
                4306362335420736255,
            ],
            np.uint64,
        )

        # C test output with counter=[0,0] and key=[0,0]
        # self.array00_gaussian = np.array([-0.680004, -0.633214, -1.523790,
        # -1.847484, -0.427139, 0.991348, 0.601200, 0.481707, -0.085967,
        # 0.110980, -1.220734], np.float64)

        self.array00_m11 = np.array(
            [
                -0.478794,
                -0.704256,
                0.533997,
                0.004571,
                0.392376,
                -0.785938,
                -0.373569,
                0.866371,
                0.325575,
                -0.266422,
                0.937621,
            ],
            np.float64,
        )
        self.array00_01 = np.array(
            [
                0.760603,
                0.647872,
                0.266998,
                0.002285,
                0.196188,
                0.607031,
                0.813215,
                0.433185,
                0.162788,
                0.866789,
                0.468810,
            ],
            np.float64,
        )
        self.array00_uint64 = np.array(
            [
                14030652003081164901,
                11951131804325250240,
                4925249918008276254,
                42156276261651215,
                3619028682724454876,
                11197741606642300638,
                15001177968947004470,
                7990859118804543502,
                3002902877118036975,
                15989435820833075781,
                8648023362736035120,
            ],
            np.uint64,
        )

    def test_rng_gaussian(self):
        # # Testing with any counter and any key
        # result = random(self.size, counter=self.counter, key=self.key)
        # self.assertTrue((result > -10).all() and (result < 10).all())
        # np.testing.assert_array_almost_equal(result, self.array_gaussian)

        # # Testing with counter=[0,0] and key=[0,0]
        # result = random(self.size, counter=self.counter00, key=self.key00)
        # self.assertTrue((result > -10).all() and (result < 10).all())
        # np.testing.assert_array_almost_equal(result, self.array00_gaussian)

        # Test reproducibility
        result1 = random(20, key=[0, 0], counter=[0, 0], sampler="gaussian")
        result2 = random(20, key=[0, 0], counter=[0, 5], sampler="gaussian")
        np.testing.assert_array_almost_equal(result1[5:], result2[:-5])

        # ...And with threads
        result1 = random(
            20, key=[0, 0], counter=[0, 0], sampler="gaussian", threads=True
        )
        result2 = random(
            20, key=[0, 0], counter=[0, 5], sampler="gaussian", threads=True
        )
        np.testing.assert_array_almost_equal(result1[5:], result2[:-5])
        return

    def test_rng_m11(self):
        # Testing with any counter and any key
        result = random(
            self.size, key=self.key, counter=self.counter, sampler="uniform_m11"
        )
        self.assertTrue((result > -1.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(result, self.array_m11)

        # Testing with counter=[0,0] and key=[0,0]
        result = random(
            self.size, key=self.key00, counter=self.counter00, sampler="uniform_m11"
        )
        self.assertTrue((result > -1.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(result, self.array00_m11)

        # ...And with threads
        result = random(
            self.size,
            key=self.key,
            counter=self.counter,
            sampler="uniform_m11",
            threads=True,
        )
        self.assertTrue((result > -1.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(result, self.array_m11)

        result = random(
            self.size,
            key=self.key00,
            counter=self.counter00,
            sampler="uniform_m11",
            threads=True,
        )
        self.assertTrue((result > -1.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(result, self.array00_m11)

        return

    def test_rng_01(self):
        # Testing with any counter and any key
        result = random(
            self.size, key=self.key, counter=self.counter, sampler="uniform_01"
        )
        self.assertTrue((result > 0.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(np.array(result), self.array_01)

        # Testing with counter=[0,0] and key=[0,0]
        result = random(
            self.size, key=self.key00, counter=self.counter00, sampler="uniform_01"
        )
        self.assertTrue((result > 0.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(result, self.array00_01)

        # ...And with threads
        result = random(
            self.size,
            key=self.key,
            counter=self.counter,
            sampler="uniform_01",
            threads=True,
        )
        self.assertTrue((result > 0.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(result, self.array_01)

        result = random(
            self.size,
            key=self.key00,
            counter=self.counter00,
            sampler="uniform_01",
            threads=True,
        )
        self.assertTrue((result > 0.0).all() and (result < 1.0).all())
        np.testing.assert_array_almost_equal(result, self.array00_01)

        return

    def test_rng_uint64(self):
        # Testing with any counter and any key
        result = random(
            self.size, key=self.key, counter=self.counter, sampler="uniform_uint64"
        )
        self.assertTrue(type(result) == AlignedU64)
        np.testing.assert_array_equal(result, self.array_uint64)

        # Testing with counter=[0,0] and key=[0,0]
        result = random(
            self.size, key=self.key00, counter=self.counter00, sampler="uniform_uint64"
        )
        self.assertTrue(type(result) == AlignedU64)
        np.testing.assert_array_equal(result, self.array00_uint64)

        # ...And with threads
        result = random(
            self.size,
            key=self.key,
            counter=self.counter,
            sampler="uniform_uint64",
            threads=True,
        )
        self.assertTrue(type(result) == AlignedU64)
        np.testing.assert_array_equal(result, self.array_uint64)

        result = random(
            self.size,
            key=self.key00,
            counter=self.counter00,
            sampler="uniform_uint64",
            threads=True,
        )
        self.assertTrue(type(result) == AlignedU64)
        np.testing.assert_array_equal(result, self.array00_uint64)

        return

    # Multiple stream testing.

    def test_rng_gaussian_multi(self):
        # Test reproducibility
        samples = [20 for x in range(self.nstream)]
        keys = [(0, 0) for x in range(self.nstream)]
        counters = [(0, 0) for x in range(self.nstream)]

        result1 = random_multi(samples, keys, counters, sampler="gaussian")

        counters = [(0, 5) for x in range(self.nstream)]
        result2 = random_multi(samples, keys, counters, sampler="gaussian")

        for i in range(self.nstream):
            np.testing.assert_array_almost_equal(result1[i][5:], result2[i][:-5])
        return

    def test_rng_m11_multi(self):
        # Testing with any counter and any key
        samples = [self.size for x in range(self.nstream)]
        keys = [self.key for x in range(self.nstream)]
        counters = [self.counter for x in range(self.nstream)]
        result = random_multi(samples, keys, counters, sampler="uniform_m11")
        for i in range(self.nstream):
            self.assertTrue((result[i] > -1.0).all() and (result[i] < 1.0).all())
            np.testing.assert_array_almost_equal(result[i], self.array_m11)

        # Testing with counter=[0,0] and key=[0,0]
        samples = [self.size for x in range(self.nstream)]
        keys = [self.key00 for x in range(self.nstream)]
        counters = [self.counter00 for x in range(self.nstream)]
        result = random_multi(samples, keys, counters, sampler="uniform_m11")
        for i in range(self.nstream):
            self.assertTrue((result[i] > -1.0).all() and (result[i] < 1.0).all())
            np.testing.assert_array_almost_equal(result[i], self.array00_m11)
        return

    def test_rng_01_multi(self):
        # Testing with any counter and any key
        samples = [self.size for x in range(self.nstream)]
        keys = [self.key for x in range(self.nstream)]
        counters = [self.counter for x in range(self.nstream)]
        result = random_multi(samples, keys, counters, sampler="uniform_01")
        for i in range(self.nstream):
            self.assertTrue((result[i] > 0.0).all() and (result[i] < 1.0).all())
            np.testing.assert_array_almost_equal(result[i], self.array_01)

        # Testing with counter=[0,0] and key=[0,0]
        samples = [self.size for x in range(self.nstream)]
        keys = [self.key00 for x in range(self.nstream)]
        counters = [self.counter00 for x in range(self.nstream)]
        result = random_multi(samples, keys, counters, sampler="uniform_01")
        for i in range(self.nstream):
            self.assertTrue((result[i] > 0.0).all() and (result[i] < 1.0).all())
            np.testing.assert_array_almost_equal(result[i], self.array00_01)
        return

    def test_rng_uint64_multi(self):
        # Testing with any counter and any key
        samples = [self.size for x in range(self.nstream)]
        keys = [self.key for x in range(self.nstream)]
        counters = [self.counter for x in range(self.nstream)]
        result = random_multi(samples, keys, counters, sampler="uniform_uint64")
        for i in range(self.nstream):
            self.assertTrue(type(result[i]) == AlignedU64)
            np.testing.assert_array_equal(result[i], self.array_uint64)

        # Testing with counter=[0,0] and key=[0,0]
        samples = [self.size for x in range(self.nstream)]
        keys = [self.key00 for x in range(self.nstream)]
        counters = [self.counter00 for x in range(self.nstream)]
        result = random_multi(samples, keys, counters, sampler="uniform_uint64")
        for i in range(self.nstream):
            self.assertTrue(type(result[i]) == AlignedU64)
            np.testing.assert_array_equal(result[i], self.array00_uint64)
        return
