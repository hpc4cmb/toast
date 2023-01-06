# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
import os

import numpy as np

from .. import ops as ops
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class MemoryCounterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_counter(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Set up a pipeline that generates some data
        pipe_ops = [
            ops.DefaultNoiseModel(),
            ops.SimNoise(),
        ]

        pipe = ops.Pipeline()
        pipe.operators = pipe_ops
        pipe.apply(data)

        # Get the memory used
        mcount = ops.MemoryCounter()
        bytes = mcount.apply(data)

        close_data(data)
