# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

import numpy as np

import unittest

from toast.rng import *


class RNGTest(unittest.TestCase):

    def setUp(self):
        #data
        self.size = 6
        self.array = np.zeros(self.size)
        self.counter = [0,0]
        self.key = [0,0]

        # self.array_gaussian = np.array([ -1.286392 , 0.085829 , -1.131298 , -0.845273 , 1.076501 , -0.115413 ], np.float64)
        # self.array_m11 = np.array([ -0.478794 , 0.871153 , -0.704256 , 0.737851 , 0.533997 , -0.886999 ], np.float64)
        # self.array_01 = np.array([ 0.760603 , 0.435576 , 0.647872 , 0.368925 , 0.266998 , 0.556500 ], np.float64)
        # self.array_uint64 = np.array([ 14030652003081164901 , 8034964082011408461 , 11951131804325250240 , 6805473726779904618 , 4925249918008276254 , 10265621268231006908 ], np.uint64)

    def test_rng_gaussian(self):
        CBRNG.random(self.array,self.counter)
        self.assertTrue((self.array > -10).all() and (self.array < 10).all())

    def test_rng_m11(self):
        CBRNG.random(self.array,self.counter,"uniform_m11")
        self.assertTrue((self.array > -1).all() and (self.array < 1).all())

    def test_rng_01(self):
        CBRNG.random(self.array,self.counter,"uniform_01")
        self.assertTrue((self.array > 0).all() and (self.array < 1).all())

    def test_rng_uint64(self):
        self.array = np.zeros(self.size,dtype=np.uint64)
        CBRNG.random(self.array,self.counter,"uniform_uint64")
        self.assertTrue(type(self.array[0]) == np.uint64)
