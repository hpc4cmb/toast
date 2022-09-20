# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import copy
import os
import types
from datetime import datetime

import numpy as np
import numpy.testing as nt
from astropy import units as u
from tomlkit import comment, document, dumps, loads, nl, table

from .. import ops
from ..config import (
    build_config,
    create_from_config,
    dump_toml,
    load_config,
    parse_config,
)
from ..data import Data
from ..instrument import Focalplane, Telescope
from ..schedule_sim_satellite import create_satellite_schedule
from ..templates import Offset, SubHarmonic
from ..traits import (
    Bool,
    Dict,
    Float,
    Instance,
    Int,
    List,
    Quantity,
    Set,
    Tuple,
    Unicode,
    trait_docs,
)
from ..utils import Environment, Logger
from ._helpers import create_comm, create_outdir, create_space_telescope
from .mpi import MPITestCase


class FakeClass:
    def __init__(self, other="blah"):
        self.foo = "bar"
        self.other = other


@trait_docs
class ConfigOperator(ops.Operator):
    """Dummy class to test all the different trait types."""

    # Class traits
    API = Int(0, help="Internal interface version for this operator")

    unicode_default = Unicode("str", help="String default")
    unicode_empty = Unicode("", help="String empty")
    unicode_none = Unicode(None, allow_none=True, help="String none")

    int_default = Int(123456, help="Int default")
    int_none = Int(None, allow_none=True, help="Int None")

    float_default = Float(1.2345, help="Float default")
    float_none = Float(None, allow_none=True, help="Float none")

    bool_default = Bool(False, help="Bool default")
    bool_none = Bool(None, allow_none=True, help="Bool none")

    quantity_default = Quantity(1.2345 * u.second, help="Quantity default")
    quantity_none = Quantity(None, allow_none=True, help="Quantity none")

    # NOTE:  Our config system does not currently support Instance traits
    # with allow_none=False.
    instance_none = Instance(allow_none=True, klass=FakeClass, help="Instance none")

    list_none = List(None, allow_none=True, help="List none")
    list_string = List(["foo", "bar", "blat"], help="List string default")
    list_string_empty = List(["", "", ""], help="List string empty")
    list_float = List([1.23, 4.56, 7.89], help="List float default")
    list_quant = List([1.23 * u.meter, 4.56 * u.K], help="List Quantity default")
    list_mixed = List(
        [None, True, "", "foo", 1.23, 4.56, 7.89 * u.meter], help="list mixed"
    )

    dict_none = Dict(None, allow_none=True, help="Dict none")
    dict_string = Dict(
        {"a": "foo", "b": "bar", "c": "blat"}, help="Dict string default"
    )
    dict_string_empty = Dict({"a": "", "b": "", "c": ""}, help="Dict string empty")
    dict_float = Dict({"a": 1.23, "b": 4.56, "c": 7.89}, help="Dict float default")
    dict_float = Dict(
        {"a": 1.23 * u.meter, "b": 4.56 * u.K}, help="Dict Quantity default"
    )
    dict_mixed = Dict(
        {"a": None, "b": True, "c": "", "d": 4.56, "e": 7.89 * u.meter},
        help="Dict mixed",
    )

    set_none = Set(None, allow_none=True, help="Set none")
    set_string = Set({"a", "b", "c"}, help="Set string default")
    set_string_empty = Set({"", "", ""}, help="Set string empty")
    set_float = Set({1.23, 4.56, 7.89}, help="Set float default")
    set_quant = Set({1.23 * u.meter, 4.56 * u.meter}, help="Set Quantity default")
    set_mixed = Set({None, "", "foo", True, 4.56, 7.89 * u.meter}, help="Set mixed")

    tuple_none = Tuple(None, allow_none=True, help="Tuple string default")
    tuple_string = Tuple(["foo", "bar", "blat"], help="Tuple string default")
    tuple_string_empty = Tuple(["", "", ""], help="Tuple string empty")
    tuple_float = Tuple([1.23, 4.56, 7.89], help="Tuple float")
    tuple_float = Tuple([1.23 * u.meter, 4.56 * u.meter], help="Tuple Quantity")
    tuple_mixed = Tuple(
        [None, True, "", "foo", 4.56, 7.89 * u.meter], help="Tuple mixed"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm

    def _finalize(self, data, **kwargs):
        pass

    def _requires(self):
        return dict()

    def _provides(self):
        return dict()

    def _accelerators(self):
        return list()


class ConfigTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.toastcomm = create_comm(self.comm)

    def create_operators(self):
        oplist = [
            ops.SimSatellite(
                name="sim_satellite",
                hwp_angle="hwp",
                hwp_rpm=1.0,
            ),
            ops.DefaultNoiseModel(name="noise_model"),
            ops.SimNoise(name="sim_noise"),
            ops.MemoryCounter(name="mem_count"),
        ]
        return {x.name: x for x in oplist}

    def create_templates(self):
        tmpls = [Offset(name="baselines"), SubHarmonic(name="subharmonic")]
        return {x.name: x for x in tmpls}

    def test_trait_types(self):
        fake = ConfigOperator(name="fake")

        fakeconf = fake.get_config()

        fakefile = os.path.join(self.outdir, "types_fake.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(fakefile, fakeconf)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        loadconf = None
        if self.toastcomm.world_rank == 0:
            loadconf = load_config(fakefile)
        if self.toastcomm.comm_world is not None:
            loadconf = self.toastcomm.comm_world.bcast(loadconf, root=0)

        run = create_from_config(loadconf)

        if run.operators.fake != fake:
            print(
                f" Trait type round trip failed, {run.operators.fake} != {fake}",
                flush=True,
            )
        self.assertTrue(run.operators.fake == fake)

    def test_config_multi(self):
        testops = self.create_operators()
        objs = [y for x, y in testops.items()]
        defaults = build_config(objs)

        conf_file = os.path.join(self.outdir, "multi_defaults.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(conf_file, defaults)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        # Now change some values
        testops["mem_count"].prefix = "newpref"
        testops["mem_count"].enabled = False
        testops["sim_noise"].serial = False
        testops["sim_satellite"].hwp_rpm = 8.0
        testops["sim_satellite"].distribute_time = True

        # Dump this config
        objs = [y for x, y in testops.items()]
        conf = build_config(objs)
        conf2_file = os.path.join(self.outdir, "multi_conf.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(conf2_file, conf)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        # Options for testing
        arg_opts = [
            "--mem_count.prefix",
            "altpref",
            "--mem_count.enable",
            "--sim_noise.serial",
            "--sim_satellite.hwp_rpm",
            "3.0",
            "--sim_satellite.no_distribute_time",
            "--config",
            conf_file,
            conf2_file,
        ]

        parser = argparse.ArgumentParser(description="Test")
        config, remaining, jobargs = parse_config(
            parser,
            operators=[y for x, y in testops.items()],
            templates=list(),
            prefix="",
            opts=arg_opts,
        )

        # Instantiate
        run = create_from_config(config)
        runops = run.operators

        # Check
        self.assertTrue(runops.mem_count.prefix == "altpref")
        self.assertTrue(runops.mem_count.enabled == True)
        self.assertTrue(runops.sim_noise.serial == True)
        self.assertTrue(runops.sim_satellite.distribute_time == False)
        self.assertTrue(runops.sim_satellite.hwp_rpm == 3.0)

    def test_roundtrip(self):
        testops = self.create_operators()
        tmpls = self.create_templates()

        objs = [y for x, y in testops.items()]
        objs.extend([y for x, y in tmpls.items()])
        conf = build_config(objs)

        conf_file = os.path.join(self.outdir, "roundtrip_conf.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(conf_file, conf)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        new_conf = None
        if self.toastcomm.world_rank == 0:
            new_conf = load_config(conf_file)
        if self.toastcomm.comm_world is not None:
            new_conf = self.toastcomm.comm_world.bcast(new_conf, root=0)

        check_file = os.path.join(self.outdir, "roundtrip_check.toml")
        check = None
        if self.toastcomm.world_rank == 0:
            dump_toml(check_file, new_conf)
            check = load_config(check_file)
        if self.toastcomm.comm_world is not None:
            check = self.toastcomm.comm_world.bcast(check, root=0)
        run = create_from_config(check)

        for opname, op in testops.items():
            other = getattr(run.operators, opname)
            self.assertTrue(other == op)

        for tname, tmpl in tmpls.items():
            other = getattr(run.templates, tname)
            self.assertTrue(other == tmpl)

    def test_run(self):
        testops = self.create_operators()

        pipe = ops.Pipeline(name="sim_pipe")
        pipe.operators = [y for x, y in testops.items()]
        conf_pipe = pipe.get_config()

        conf_file = os.path.join(self.outdir, "run_conf.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(conf_file, conf_pipe)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        run = create_from_config(conf_pipe)

        data = Data(self.toastcomm)

        tele = create_space_telescope(self.toastcomm.group_size)
        sch = create_satellite_schedule(
            mission_start=datetime(2023, 2, 23), num_observations=self.toastcomm.ngroups
        )

        # Add our fake telescope and schedule
        run.operators.sim_satellite.telescope = tele
        run.operators.sim_satellite.schedule = sch

        run.operators.sim_pipe.apply(data)
