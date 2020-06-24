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

from ..future_ops import SimSatellite  # , Pipeline, SimNoise, DefaultNoiseModel

from ..data import Data

from ._helpers import create_outdir, create_distdata, boresight_focalplane, create_comm


class ConfigTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.data = create_distdata(
            self.comm, obs_per_group=1, future_obs=True, samples=10
        )

        env = Environment.get()

        # Make a fake focalplane for pipeline tests

        self.ndet = 4
        (
            dnames,
            dquat,
            depsilon,
            drate,
            dnet,
            dfmin,
            dfknee,
            dalpha,
        ) = boresight_focalplane(self.ndet)

        detdata = {}
        for d in dnames:
            detdata[d] = {
                "fsample": drate[d],
                "NET": dnet[d],
                "quat": dquat[d],
                "fmin": dfmin[d],
                "fknee": dfknee[d],
                "alpha": dalpha[d],
                "pol_leakage": depsilon[d],
            }

        self.focalplane = Focalplane(
            detector_data=detdata, sample_rate=drate[dnames[0]]
        )

        # Create some example configs to load

        ops = [SimSatellite(name="sim_satellite")]
        #
        # ops = {
        #     "sim_satellite": SimSatellite,
        #     "noise_model": DefaultNoiseModel,
        #     "sim_noise": SimNoise,
        # }

        conf = build_config(ops)

        self.doc1_file = os.path.join(self.outdir, "doc1.toml")
        dump_toml(self.doc1_file, conf)
        #
        # ops = {"sim_pipe": Pipeline}
        #
        # conf = default_config(operators=ops)
        # conf["operators"]["sim_pipe"]["operators"] = [
        #     "@config:/operators/sim_satellite",
        #     "@config:/operators/noise_model",
        #     "@config:/operators/sim_noise",
        # ]
        #
        # self.doc2_file = os.path.join(self.outdir, "doc2.toml")
        # dump_config(self.doc2_file, conf)

    def test_load(self):
        conf = load_config(self.doc1_file)
        # conf = load_config(self.doc2_file, input=conf)

    def test_roundtrip(self):
        conf = load_config(self.doc1_file)
        # conf = load_config(self.doc2_file, input=conf)
        check_file = os.path.join(self.outdir, "check.toml")
        dump_toml(check_file, conf)
        check = load_config(check_file)
        self.assertTrue(conf == check)

    def test_create(self):
        conf = load_config(self.doc1_file)
        # conf = load_config(self.doc2_file, input=conf)

        run = create(conf)

        # Add our fake telescope
        run["operators"]["sim_satellite"].telescope = Telescope(
            name="fake", focalplane=self.focalplane
        )

        # print(run)

        # print(run["operators"]["sim_pipe"].config["operators"])

    # def test_run(self):
    #     conf = load_config(self.doc1_file)
    #     conf = load_config(self.doc2_file, input=conf)
    #
    #     # Add a fake telescope for testing
    #     conf["operators"]["sim_satellite"]["telescope"] = Telescope(
    #         name="fake", focalplane=self.focalplane
    #     )
    #
    #     run = create(conf)
    #
    #     toastcomm = create_comm(self.comm)
    #     data = Data(toastcomm)
    #
    #     run["operators"]["sim_pipe"].exec(data)
    #     # for obs in data.obs:
    #     #     for d in obs.signal().detectors:
    #     #         print(d, obs.signal()[d][:5])
