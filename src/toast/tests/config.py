# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import copy

import numpy as np
import numpy.testing as nt

from tomlkit import comment, document, nl, table, dumps, loads

from ..utils import Environment

from ..config import load_config, dump_toml, build_config, create

from ..instrument import Telescope, Focalplane

from ..ops import SimSatellite, Pipeline, SimNoise, DefaultNoiseModel

from ..data import Data

from ._helpers import create_outdir, create_comm, create_space_telescope


class ConfigTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.toastcomm = create_comm(self.comm)

        env = Environment.get()

        # Create some example configs to load

        ops = [
            SimSatellite(name="sim_satellite"),
            DefaultNoiseModel(name="noise_model"),
            SimNoise(name="sim_noise"),
        ]

        self.doc1_file = os.path.join(self.outdir, "doc1.toml")
        self.doc2_file = os.path.join(self.outdir, "doc2.toml")

        conf = build_config(ops)

        if self.toastcomm.world_rank == 0:
            dump_toml(self.doc1_file, conf)

        pipe = Pipeline(name="sim_pipe")
        pipe.operators = ops
        conf_pipe = pipe.get_config()

        if self.toastcomm.world_rank == 0:
            dump_toml(self.doc2_file, conf_pipe)

    def test_load(self):
        conf = None
        if self.toastcomm.world_rank == 0:
            conf = load_config(self.doc1_file)
            conf = load_config(self.doc2_file)
        if self.toastcomm.comm_world is not None:
            conf = self.toastcomm.comm_world.bcast(conf, root=0)

    def test_roundtrip(self):
        conf = None
        if self.toastcomm.world_rank == 0:
            conf = load_config(self.doc2_file)
        if self.toastcomm.comm_world is not None:
            conf = self.toastcomm.comm_world.bcast(conf, root=0)

        check_file = os.path.join(self.outdir, "check.toml")
        check = None
        if self.toastcomm.world_rank == 0:
            dump_toml(check_file, conf)
            check = load_config(check_file)
        if self.toastcomm.comm_world is not None:
            check = self.toastcomm.comm_world.bcast(check, root=0)
        self.assertTrue(conf == check)

    def test_run(self):
        conf = None
        if self.toastcomm.world_rank == 0:
            conf = load_config(self.doc2_file)
            print("", flush=True)

        if self.toastcomm.comm_world is not None:
            conf = self.toastcomm.comm_world.bcast(conf, root=0)

        run = create(conf)

        data = Data(self.toastcomm)
        tele = create_space_telescope(self.toastcomm.group_size)

        # Add our fake telescope
        run["operators"]["sim_satellite"].telescope = tele

        # Generate one observation per group
        run["operators"]["sim_satellite"].n_observation = self.toastcomm.ngroups

        run["operators"]["sim_pipe"].apply(data)

        for obs in data.obs:
            for d in obs.local_detectors:
                print(
                    "proc {}, det {}: {}".format(
                        self.toastcomm.world_rank, d, obs.detdata["noise"][d][:5]
                    ),
                    flush=True,
                )
