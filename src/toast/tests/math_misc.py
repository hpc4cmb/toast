# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

import numpy as np

from .._libtoast import integrate_simpson
from ..rng import random, random_multi
from ..utils import AlignedU64
from .mpi import MPITestCase


class MathMiscTest(MPITestCase):
    def setUp(self):
        pass

    def test_integrate_simpson_odd(self):
        x = np.arange(101)
        f = np.arange(101) * 1e-6

        val1 = integrate_simpson(x, f)
        val2 = simpson(f, x=x)

        assert np.abs((val1 - val2) / val2) < 1e-4

    def test_integrate_simpson_even(self):
        x = np.arange(100)
        f = np.arange(100) * 1e-6

        val1 = integrate_simpson(x, f)
        val2 = simpson(f, x=x)

        assert np.abs((val1 - val2) / val2) < 1e-4
