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
    Unit,
    trait_docs,
    trait_scalar_to_string,
)
from ..utils import Environment, Logger
from ._helpers import close_data, create_comm, create_outdir, create_space_telescope
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

    quantity_default = Quantity(1.2345 * u.meter / u.second, help="Quantity default")
    quantity_none = Quantity(None, allow_none=True, help="Quantity none")

    unit_default = Unit(u.meter / u.second, help="Unit default")
    unit_none = Unit(None, allow_none=True, help="Unit none")

    # NOTE:  Our config system does not currently support Instance traits
    # with allow_none=False.
    instance_none = Instance(allow_none=True, klass=FakeClass, help="Instance none")

    list_empty = List(list(), help="List empty")
    list_string = List(["foo", "bar", "blat"], help="List string default")
    list_string_empty = List(["", "", ""], help="List string empty")
    list_float = List([1.23, 4.56, 7.89], help="List float default")
    list_quant = List(
        [1.23 * u.meter / u.second, 4.56 * u.K], help="List Quantity default"
    )
    list_mixed = List(
        [None, True, "", "foo", 1.23, 4.56, 7.89 * u.meter], help="list mixed"
    )

    dict_empty = Dict(dict(), help="Dict empty")
    dict_string = Dict(
        {"a": "foo", "b": "bar", "c": "blat"}, help="Dict string default"
    )
    dict_string_empty = Dict({"a": "", "b": "", "c": ""}, help="Dict string empty")
    dict_float = Dict({"a": 1.23, "b": 4.56, "c": 7.89}, help="Dict float default")
    dict_quant = Dict(
        {"a": 1.23 * u.meter / u.second, "b": 4.56 * u.K}, help="Dict Quantity default"
    )
    dict_mixed = Dict(
        {"a": None, "b": True, "c": "", "d": 4.56, "e": 7.89 * u.meter},
        help="Dict mixed",
    )

    set_empty = Set(set(), help="Set empty")
    set_string = Set({"a", "b", "c"}, help="Set string default")
    set_string_empty = Set({"", "", ""}, help="Set string empty")
    set_float = Set({1.23, 4.56, 7.89}, help="Set float default")
    set_quant = Set(
        {1.23 * u.meter / u.second, 4.56 * u.meter}, help="Set Quantity default"
    )
    set_mixed = Set({None, "", "foo", True, 4.56, 7.89 * u.meter}, help="Set mixed")

    tuple_empty = Tuple(tuple(), help="Tuple empty")
    tuple_string = Tuple(("foo", "bar", "blat"), help="Tuple string default")
    tuple_string_empty = Tuple(("", "", ""), help="Tuple string empty")
    tuple_float = Tuple((1.23, 4.56, 7.89), help="Tuple float")
    tuple_quant = Tuple(
        (1.23 * u.meter / u.second, 4.56 * u.meter), help="Tuple Quantity"
    )
    tuple_mixed = Tuple(
        (None, True, "", "foo", 4.56, 7.89 * u.meter), help="Tuple mixed"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def modified_traits(self):
        """Construct a modified set of trait values."""
        tmod = dict()
        for tname, trait in self.traits().items():
            if tname == "enabled":
                continue
            if tname == "API":
                continue
            if trait.get(self) is None:
                continue
            if isinstance(trait, Bool):
                # toggle value
                if trait.get(self):
                    tmod[tname] = False
                else:
                    tmod[tname] = True
            elif trait.get(self) is None:
                tmod[tname] = None
            elif isinstance(trait, Unit):
                tmod[tname] = u.km / u.s
            elif isinstance(trait, Quantity):
                tmod[tname] = 5.0 * u.km / u.s
            elif isinstance(trait, (Int, Float)):
                tmod[tname] = 1 + trait.get(self)
            elif isinstance(trait, Unicode):
                tmod[tname] = f"modified_{trait.get(self)}"
            elif isinstance(trait, Set):
                # Remove one element
                s = set(trait.get(self))
                if len(s) > 0:
                    _ = s.pop()
                tmod[tname] = s
            elif isinstance(trait, Tuple):
                # Reverse it
                temp = list(trait.get(self))
                temp.reverse()
                tmod[tname] = tuple(temp)
            elif isinstance(trait, List):
                # Reverse it
                temp = list(trait.get(self))
                temp.reverse()
                tmod[tname] = temp
            elif isinstance(trait, Dict):
                # Remove one element
                d = dict(trait.get(self))
                if len(d) > 0:
                    k = list(d.keys())[0]
                    del d[k]
                tmod[tname] = d
            else:
                # This is some other type
                pass
        return tmod

    def args_test(self):
        """Build an argparse list for testing."""
        # Get a modified set of trait values
        tmod = self.modified_traits()

        # Make an argparse list
        targs = list()
        for k, v in tmod.items():
            if v is None:
                # Skip this
                continue
            if isinstance(v, bool):
                # Special case...
                if v:
                    targs.append(f"--{self.name}.{k}")
                else:
                    targs.append(f"--{self.name}.no_{k}")
            else:
                targs.append(f"--{self.name}.{k}")
                if isinstance(v, set):
                    if len(v) == 0:
                        targs.append("{}")
                    else:
                        formatted = set([trait_scalar_to_string(x) for x in v])
                        targs.append(str(formatted))
                elif isinstance(v, tuple):
                    if len(v) == 0:
                        targs.append("()")
                    else:
                        formatted = tuple([trait_scalar_to_string(x) for x in v])
                        targs.append(str(formatted))
                elif isinstance(v, dict):
                    if len(v) == 0:
                        targs.append("{}")
                    else:
                        formatted = {x: trait_scalar_to_string(y) for x, y in v.items()}
                        targs.append(str(formatted))
                elif isinstance(v, list):
                    if len(v) == 0:
                        targs.append("[]")
                    else:
                        formatted = [trait_scalar_to_string(x) for x in v]
                        targs.append(str(formatted))
                elif isinstance(v, u.Unit):
                    targs.append(str(v))
                elif isinstance(v, u.Quantity):
                    targs.append(f"{v.value:0.14e} {v.unit}")
                else:
                    targs.append(str(v))
        return targs

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm

    def _finalize(self, data, **kwargs):
        pass

    def _requires(self):
        return dict()

    def _provides(self):
        return dict()


class ConfigTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.toastcomm = create_comm(self.comm)

    def compare_trait(self, check, expected):
        def _compare_element(chk, expt):
            if isinstance(chk, float) and isinstance(expt, float):
                return np.allclose(chk, expt)
            elif isinstance(chk, Quantity) and isinstance(expt, Quantity):
                return np.allclose(chk.value, expt.value) and chk.unit == expt.unit
            else:
                return chk == expt

        if isinstance(check, (list, tuple)) and isinstance(expected, (list, tuple)):
            result = True
            for a, b in zip(check, expected):
                if not _compare_element(a, b):
                    result = False
            return result
        if isinstance(check, set) and isinstance(expected, set):
            # Jump through some hoops here...
            if len(check) != len(expected):
                return False
            scheck = set([str(x) for x in check])
            sexpected = set([str(x) for x in expected])
            for a, b in zip(sorted(scheck), sorted(sexpected)):
                if a != b:
                    return False
            return True
        elif isinstance(check, dict) and isinstance(expected, dict):
            result = True
            if check.keys() != expected.keys():
                return False
            for k in check.keys():
                if not _compare_element(check[k], expected[k]):
                    result = False
            return result
        else:
            return _compare_element(check, expected)

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
            ops.PointingDetectorSimple(name="det_pointing"),
            ops.PixelsWCS(name="pixels"),
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

    def test_trait_types_argparse(self):
        fake = ConfigOperator(name="fake")

        test_args = fake.args_test()

        parser = argparse.ArgumentParser(description="Test")
        config, remaining, jobargs = parse_config(
            parser,
            operators=[fake],
            templates=list(),
            prefix="",
            opts=test_args,
        )

        check_fake = ConfigOperator.from_config("fake", config["operators"]["fake"])
        # run = create_from_config(config)

        # Modified values that we expect
        check = fake.modified_traits()

        # Compare
        # op = run.operators.fake
        for tname, trait in check_fake.traits().items():
            if tname in check:
                tval = trait.get(check_fake)
                if not self.compare_trait(tval, check[tname]):
                    print(f"{tval} != {check[tname]}")
                    self.assertTrue(False)

    def test_config_multi(self):
        testops = self.create_operators()
        testops["fake"] = ConfigOperator(name="fake")
        testops["scan_map"] = ops.ScanHealpixMap(name="scan_map")

        objs = [y for x, y in testops.items()]
        defaults = build_config(objs)

        conf_defaults_file = os.path.join(self.outdir, "multi_defaults.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(conf_defaults_file, defaults)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        # Now change some values
        testops["mem_count"].prefix = "newpref"
        testops["mem_count"].enabled = False
        testops["sim_noise"].serial = False
        testops["sim_satellite"].hwp_rpm = 8.0
        testops["sim_satellite"].distribute_time = True
        testops["sim_satellite"].shared_flags = None
        testops["scan_map"].file = "blat"
        testops["fake"].unicode_none = "foo"

        # Dump these to 2 disjoint configs
        testops_fake = {"fake": testops["fake"]}
        testops_notfake = dict(testops)
        del testops_notfake["fake"]

        conf_fake = build_config([y for x, y in testops_fake.items()])
        conf_notfake = build_config([y for x, y in testops_notfake.items()])

        conf_notfake_file = os.path.join(self.outdir, "multi_notfake.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(conf_notfake_file, conf_notfake)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        conf_fake_file = os.path.join(self.outdir, "multi_fake.toml")
        if self.toastcomm.world_rank == 0:
            dump_toml(conf_fake_file, conf_fake)
        if self.toastcomm.comm_world is not None:
            self.toastcomm.comm_world.barrier()

        # Load the configs in either order (since they are disjoint)
        # and verify that the final result is the same

        iter = 1
        for conf_order in (
            [conf_notfake_file, conf_fake_file],
            [conf_fake_file, conf_notfake_file],
        ):
            # Options for testing
            arg_opts = [
                "--mem_count.prefix",
                "altpref",
                "--mem_count.enable",
                "--sim_noise.serial",
                "--sim_satellite.hwp_rpm",
                "3.0",
                "--sim_satellite.no_distribute_time",
                "--pixels.resolution",
                "(0.05 deg, 0.05 deg)",
                "--scan_map.file",
                "foobar",
                "--fake.unicode_none",
                "None",
                "--config",
            ]
            arg_opts.extend(conf_order)

            parser = argparse.ArgumentParser(description="Test")
            config, remaining, jobargs = parse_config(
                parser,
                operators=[y for x, y in testops.items()],
                templates=list(),
                prefix="",
                opts=arg_opts,
            )
            debug_file = os.path.join(self.outdir, f"debug_{iter}.toml")
            if self.toastcomm.world_rank == 0:
                dump_toml(debug_file, config)
            if self.toastcomm.comm_world is not None:
                self.toastcomm.comm_world.barrier()

            iter += 1

            # Instantiate
            run = create_from_config(config)
            runops = run.operators

            # Check
            self.assertTrue(runops.mem_count.prefix == "altpref")
            self.assertTrue(runops.mem_count.enabled == True)
            self.assertTrue(runops.sim_noise.serial == True)
            self.assertTrue(runops.sim_satellite.distribute_time == False)
            self.assertTrue(runops.sim_satellite.hwp_rpm == 3.0)
            self.assertTrue(runops.sim_satellite.shared_flags is None)
            self.assertTrue(runops.fake.unicode_none is None)
            self.assertTrue(runops.scan_map.file == "foobar")

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
        conf_pipe = dict()
        for op_name, op in testops.items():
            conf_pipe = op.get_config(input=conf_pipe)

        pipe = ops.Pipeline(name="sim_pipe")
        pipe.operators = [y for x, y in testops.items() if x != "det_pointing"]
        conf_pipe = pipe.get_config(input=conf_pipe)

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

        # Set up detector pointing
        run.operators.pixels.detector_pointing = run.operators.det_pointing

        # Run it
        run.operators.sim_pipe.apply(data)

        close_data(data)
