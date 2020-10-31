# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os
import copy

import numpy as np

from .. import future_ops as ops

from .. import config as tc

from ._helpers import create_outdir, create_distdata, create_telescope


class MemoryCounterTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # One observation per group
        self.data = create_distdata(self.comm, obs_per_group=1)

        # Make a fake telescope for every observation

        tele = create_telescope(self.data.comm.group_size)

        # Set up a pipeline that generates some data

        pipe_ops = [
            ops.SimSatellite(
                name="sim_satellite",
                telescope=tele,
                n_observation=self.data.comm.ngroups,
            ),
            ops.DefaultNoiseModel(name="noise_model"),
            ops.SimNoise(name="sim_noise"),
        ]

        self.pipe = ops.Pipeline(name="sim_pipe")
        self.pipe.operators = pipe_ops

    def test_counter(self):
        # Start with empty data
        self.data.clear()

        # Run a standard pipeline to simulate some data
        self.pipe.apply(self.data)

        # Get the memory used
        mcount = ops.MemoryCounter()
        bytes = mcount.apply(self.data)

        return
