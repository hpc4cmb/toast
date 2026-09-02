# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .. import ops
from .helpers import (
    close_data,
    create_outdir,
    create_satellite_empty,
    create_satellite_data,
)
from .mpi import MPITestCase


class ArithmeticTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.shapes = [(1,), (4,), (3, 2)]
        self.types = {
            "f64": np.float64,
            "f32": np.float32,
            "i64": np.int64,
            "u64": np.uint64,
            "i32": np.int32,
            "u32": np.uint32,
            "i16": np.int16,
            "u16": np.uint16,
            "i8": np.int8,
            "u8": np.uint8,
        }

    def field_list(self, root):
        fields = []
        for ishp, shp in enumerate(self.shapes):
            for tstr, dtype in self.types.items():
                fields.append(f"{root}_{ishp}_{tstr}")
        return fields

    def create_data(self):
        rng = np.random.default_rng(seed=12345)
        data = create_satellite_empty(self.comm, obs_per_group=1, samples=10)
        for obs in data.obs:
            n_samp = obs.n_local_samples
            dets = obs.local_detectors
            for ishp, shp in enumerate(self.shapes):
                smplen = np.prod(shp)
                if smplen == 1:
                    fullshp = (n_samp,)
                else:
                    fullshp = (n_samp,) + shp
                flatlen = np.prod(fullshp)
                for tstr, dtype in self.types.items():
                    dname = f"signal_{ishp}_{tstr}"
                    obs.detdata.create(
                        dname, sample_shape=shp, dtype=dtype, detectors=None
                    )
                    for d in dets:
                        if tstr == "f64" or tstr == "f32":
                            flat = rng.random(size=flatlen, dtype=dtype)
                        else:
                            flat = rng.integers(
                                low=0, high=64, size=flatlen, dtype=dtype
                            )
                        obs.detdata[dname][d] = flat.reshape(fullshp)
        return data

    def test_movement(self):
        data = self.create_data()

        orig_fields = self.field_list("signal")
        cp_fields = self.field_list("copy")

        # Store a separate copy of all arrays for comparison
        input = {}
        for obs in data.obs:
            input[obs.uid] = {}
            for fld in orig_fields:
                input[obs.uid][fld] = {}
                for d in obs.local_detectors:
                    input[obs.uid][fld][d] = np.copy(obs.detdata[fld][d])

        ops.Copy(detdata=[(x, y) for x, y in zip(orig_fields, cp_fields)]).apply(data)

        ops.Reset(detdata=orig_fields).apply(data)

        for obs in data.obs:
            for fld in orig_fields:
                for d in obs.local_detectors:
                    if np.any(obs.detdata[fld][d]):
                        print(f"Failed to reset {obs.name}:{fld}:{d}", flush=True)
                        self.assertTrue(False)

        ops.Copy(detdata=[(x, y) for x, y in zip(cp_fields, orig_fields)]).apply(data)

        for obs in data.obs:
            for fld in orig_fields:
                for d in obs.local_detectors:
                    if not np.array_equal(obs.detdata[fld][d], input[obs.uid][fld][d]):
                        print(f"Failed to restore {obs.name}:{fld}:{d}", flush=True)
                        self.assertTrue(False)

        close_data(data)

    def test_math(self):
        data = self.create_data()

        orig_fields = self.field_list("signal")
        temp1_fields = self.field_list("temp1")
        temp2_fields = self.field_list("temp2")

        # Store a separate copy of all arrays for comparison
        input = {}
        for obs in data.obs:
            input[obs.uid] = {}
            for fld in orig_fields:
                input[obs.uid][fld] = {}
                for d in obs.local_detectors:
                    input[obs.uid][fld][d] = np.copy(obs.detdata[fld][d])

        # Test out-of-place operations

        # temp1 = signal
        ops.Copy(detdata=[(x, y) for x, y in zip(orig_fields, temp1_fields)]).apply(
            data
        )

        # temp2 = signal + temp1
        for ffirst, fsecond, fresult in zip(orig_fields, temp1_fields, temp2_fields):
            ops.Combine(op="add", first=ffirst, second=fsecond, result=fresult).apply(
                data
            )

        for obs in data.obs:
            for fld, ofld in zip(temp2_fields, orig_fields):
                for d in obs.local_detectors:
                    dvals = obs.detdata[fld][d]
                    if not np.allclose(dvals, 2 * input[obs.uid][ofld][d]):
                        print(
                            f"Failed out-of-place sum {obs.name}:{fld}:{d}", flush=True
                        )
                        bad = dvals != 0
                        indx = np.arange(obs.n_local_samples, dtype=np.int64)[bad]
                        for isamp, badval in zip(indx, dvals[bad]):
                            print(f"  {isamp}:  {badval}", flush=True)
                        self.assertTrue(False)

        # signal = 0
        ops.Reset(detdata=orig_fields).apply(data)

        # signal = temp2 - temp1
        for ffirst, fsecond, fresult in zip(temp2_fields, temp1_fields, orig_fields):
            ops.Combine(
                op="subtract", first=ffirst, second=fsecond, result=fresult
            ).apply(data)

        for obs in data.obs:
            for fld in orig_fields:
                for d in obs.local_detectors:
                    dvals = obs.detdata[fld][d]
                    if not np.allclose(dvals, input[obs.uid][fld][d]):
                        print(
                            f"Failed out-of-place diff {obs.name}:{fld}:{d}", flush=True
                        )
                        bad = dvals != 0
                        indx = np.arange(obs.n_local_samples, dtype=np.int64)[bad]
                        for isamp, badval in zip(indx, dvals[bad]):
                            print(f"  {isamp}:  {badval}", flush=True)
                        self.assertTrue(False)

        # Restore signal

        for obs in data.obs:
            for fld in orig_fields:
                for d in obs.local_detectors:
                    obs.detdata[fld][d] = input[obs.uid][fld][d]

        # Test in-place operations

        # temp1 = signal
        ops.Copy(detdata=[(x, y) for x, y in zip(orig_fields, temp1_fields)]).apply(
            data
        )

        # signal += temp1
        for ffirst, fsecond, fresult in zip(orig_fields, temp1_fields, orig_fields):
            ops.Combine(op="add", first=ffirst, second=fsecond, result=fresult).apply(
                data
            )

        for obs in data.obs:
            for fld in orig_fields:
                for d in obs.local_detectors:
                    dvals = obs.detdata[fld][d]
                    if not np.allclose(dvals, 2 * input[obs.uid][fld][d]):
                        print(
                            f"Failed in-place sum {obs.name}:{fld}:{d}", flush=True
                        )
                        bad = dvals != 0
                        indx = np.arange(obs.n_local_samples, dtype=np.int64)[bad]
                        for isamp, badval in zip(indx, dvals[bad]):
                            print(f"  {isamp}:  {badval}", flush=True)
                        self.assertTrue(False)

        # signal -= temp1
        for ffirst, fsecond, fresult in zip(orig_fields, temp1_fields, orig_fields):
            ops.Combine(
                op="subtract", first=ffirst, second=fsecond, result=fresult
            ).apply(data)

        for obs in data.obs:
            for fld in orig_fields:
                for d in obs.local_detectors:
                    dvals = obs.detdata[fld][d]
                    if not np.allclose(dvals, input[obs.uid][fld][d]):
                        print(
                            f"Failed in-place diff {obs.name}:{fld}:{d}", flush=True
                        )
                        bad = dvals != 0
                        indx = np.arange(obs.n_local_samples, dtype=np.int64)[bad]
                        for isamp, badval in zip(indx, dvals[bad]):
                            print(f"  {isamp}:  {badval}", flush=True)
                        self.assertTrue(False)

        close_data(data)

    def test_pipeline(self):
        data = create_satellite_data(self.comm, obs_per_group=1)

        model = ops.DefaultNoiseModel()
        sim_noise1 = ops.SimNoise()
        copy1 = ops.Copy(detdata=[("signal", "nse1")])
        sim_noise2 = ops.SimNoise()
        copy2 = ops.Copy(detdata=[("signal", "nse2")])
        diff1 = ops.Combine(op="subtract", first="nse2", second="nse1", result="diff")
        check = ops.Combine(op="add", first="nse1", second="diff", result="signal")

        ops.Pipeline(
            operators=[
                model,
                sim_noise1,
                copy1,
                sim_noise2,
                copy2,
                diff1,
                check,
            ]
        ).apply(data)

        for obs in data.obs:
            for d in obs.local_detectors:
                if not np.allclose(obs.detdata["signal"][d], obs.detdata["nse2"][d]):
                    print(
                        f"Failed pipeline check {obs.name}:{d}", flush=True
                    )
                    self.assertTrue(False)

        close_data(data)
