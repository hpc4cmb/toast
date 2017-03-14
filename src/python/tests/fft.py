# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

from ..fft import r1d_forward, r1d_backward

from ..rng import random

import numpy as np


class FFTTest(MPITestCase):

    def setUp(self):
        self.length = 65536
        self.input = random(self.length, counter=[0,0], key=[0,0])
        self.compare = np.copy(self.input)


    def test_roundtrip(self):
        output = r1d_forward(self.input)
        check = r1d_backward(output)
        np.testing.assert_array_almost_equal(check, self.compare)

