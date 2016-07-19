# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

import numpy as np

from toast.mpirunner import MPITestCase

from toast.rng import *


class RNGTest(MPITestCase):

    def setUp(self):
        #data
        self.size = 6
        self.counter = [1357111317,888118218888]
        self.key = [0xfeedbead,0xbaadcafe]
        self.counter00 = [0,0]
        self.key00 = [0,0]

        # C test output with counter=[1357111317,888118218888] and key=[0xfeedbead,0xbaadcafe]
        self.array_gaussian = np.array([ 2.275196 , -1.671555 , -0.389303 , -1.649345 , -0.123563 , -0.675699 ], np.float64)
        self.array_m11 = np.array([ 0.701690 , 0.037174 , -0.926218 , 0.475780 , -0.942428 , -0.420310 ], np.float64)
        self.array_01 = np.array([ 0.350845 , 0.018587 , 0.536891 , 0.237890 , 0.528786 , 0.789845 ], np.float64)
        self.array_uint64 = np.array([ 6471948635099375789 , 342865335310108017 , 9903888746739875372 , 4388295497737174248 , 9754383911267064809 , 14570067135682199833 ], np.uint64)

        # C test output with counter=[0,0] and key=[0,0]
        self.array00_gaussian = np.array([ -1.286392 , 0.085829 , -1.131298 , -0.845273 , 1.076501 , -0.115413 ], np.float64)
        self.array00_m11 = np.array([ -0.478794 , 0.871153 , -0.704256 , 0.737851 , 0.533997 , -0.886999 ], np.float64)
        self.array00_01 = np.array([ 0.760603 , 0.435576 , 0.647872 , 0.368925 , 0.266998 , 0.556500 ], np.float64)
        self.array00_uint64 = np.array([ 14030652003081164901 , 8034964082011408461 , 11951131804325250240 , 6805473726779904618 , 4925249918008276254 , 10265621268231006908 ], np.uint64)


    def test_rng_gaussian(self):
        # Testing with any counter and any key
        result = random(self.size, counter=self.counter, key=self.key)
        self.assertTrue((result > -10).all() and (result < 10).all())
        np.testing.assert_array_almost_equal(result, self.array_gaussian)

        # Testing with counter=[0,0] and key=[0,0]
        result = random(self.size, counter=self.counter00, key=self.key00)
        self.assertTrue((result > -10).all() and (result < 10).all())
        np.testing.assert_array_almost_equal(result, self.array00_gaussian)

    def test_rng_m11(self):
        # Testing with any counter and any key
        result = random(self.size, counter=self.counter, sampler="uniform_m11", key=self.key)
        self.assertTrue((result > -1).all() and (result < 1).all())
        np.testing.assert_array_almost_equal(result, self.array_m11)

        # Testing with counter=[0,0] and key=[0,0]
        result = random(self.size, counter=self.counter00, sampler="uniform_m11", key=self.key00)
        self.assertTrue((result > -1).all() and (result < 1).all())
        np.testing.assert_array_almost_equal(result, self.array00_m11)

    def test_rng_01(self):
        # Testing with any counter and any key
        result = random(self.size, counter=self.counter, sampler="uniform_01", key=self.key)
        self.assertTrue((result > 0).all() and (result < 1).all())
        np.testing.assert_array_almost_equal(result, self.array_01)

        # Testing with counter=[0,0] and key=[0,0]
        result = random(self.size, counter=self.counter00, sampler="uniform_01", key=self.key00)
        self.assertTrue((result > 0).all() and (result < 1).all())
        np.testing.assert_array_almost_equal(result, self.array00_01)

    def test_rng_uint64(self):
        # Testing with any counter and any key
        result = random(self.size, counter=self.counter, sampler="uniform_uint64", key=self.key)
        self.assertTrue(type(result[0]) == np.uint64)
        np.testing.assert_array_equal(result, self.array_uint64)

        # Testing with counter=[0,0] and key=[0,0]
        result = random(self.size, counter=self.counter00, sampler="uniform_uint64", key=self.key00)
        self.assertTrue(type(result[0]) == np.uint64)
        np.testing.assert_array_equal(result, self.array00_uint64)
