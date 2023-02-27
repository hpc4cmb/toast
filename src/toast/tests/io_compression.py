# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..data import Data
from ..io import compress_detdata, decompress_detdata, load_hdf5, save_hdf5
from ..io.compression_flac import (
    compress_detdata_flac,
    compress_flac,
    compress_flac_2D,
    decompress_detdata_flac,
    decompress_flac,
    decompress_flac_2D,
    float2int,
    have_flac_support,
    int2float,
    int64to32,
)
from ..observation import default_values as defaults
from ..observation_data import DetectorData
from ..timing import Timer
from ..utils import AlignedI32, AlignedU8
from ._helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class IoCompressionTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.types = {
            "f64": np.float64,
            "f32": np.float32,
            "i64": np.int64,
            "i32": np.int32,
        }
        self.fakedets = ["D00A", "D00B", "D01A", "D01B"]

    def test_type_conversion(self):
        rng = np.random.default_rng(12345)

        n_test = 10000

        off = 1.0e6
        scale = 100.0
        data = scale * rng.random(size=n_test, dtype=np.float64) + off
        idata, doff, dgain = float2int(data)
        check = int2float(idata, doff, dgain)
        self.assertTrue(np.allclose(check, data, rtol=1.0e-6, atol=1.0e-5))

        rng_max = np.iinfo(np.int32).max // 2
        i64data = rng.integers(
            -rng_max,
            rng_max,
            size=n_test,
            dtype=np.int64,
        )
        idata, ioff = int64to32(i64data)
        check = np.array(idata, dtype=np.int64) + ioff
        self.assertTrue(np.array_equal(check, i64data))

    def test_flac_lowlevel(self):
        if not have_flac_support():
            print("FLAC disabled, skipping...")
            return

        timer1 = Timer()
        timer2 = Timer()

        n_det = 20
        n_samp = 100000

        rng = np.random.default_rng(12345)

        rng_max = np.iinfo(np.int32).max // 2
        input = rng.integers(
            -rng_max,
            rng_max,
            size=(n_det * n_samp),
            dtype=np.int32,
        ).reshape((n_det, n_samp))

        # Compare results of 1D and 2D compression

        timer2.start()
        fbytes2, foffs2 = compress_flac_2D(input, 5)
        timer2.stop()

        # print(f"Compress 2D one shot in {timer2.seconds()} s")
        timer2.clear()

        fbytes1 = AlignedU8()
        foffs1 = np.zeros(n_det, dtype=np.int64)
        timer1.start()
        for d in range(n_det):
            cur = fbytes1.size()
            foffs1[d] = cur
            dbytes = compress_flac(input[d], 5)
            ext = len(dbytes)
            fbytes1.resize(cur + ext)
            fbytes1[cur : cur + ext] = dbytes
        timer1.stop()

        # print(f"Compress {n_det} dets with 1D in {timer1.seconds()} s")
        timer1.clear()

        self.assertTrue(len(fbytes1) == len(fbytes2))
        self.assertTrue(np.array_equal(fbytes1, fbytes2))
        self.assertTrue(np.array_equal(foffs1, foffs2))

        timer2.start()
        output2 = decompress_flac_2D(fbytes2, foffs2)
        timer2.stop()

        # print(f"Decompress 2D one shot in {timer2.seconds()} s")
        timer2.clear()

        output1 = AlignedI32()
        timer1.start()
        for d in range(n_det):
            cur = output1.size()
            if d == n_det - 1:
                slc = slice(foffs1[d], len(fbytes1), 1)
            else:
                slc = slice(foffs1[d], foffs1[d + 1], 1)
            dout = decompress_flac(fbytes1[slc])
            ext = len(dout)
            output1.resize(cur + ext)
            output1[cur : cur + ext] = dout
        timer1.stop()

        # print(f"Decompress {n_det} dets with 1D in {timer1.seconds()} s")
        timer1.clear()

        self.assertTrue(np.array_equal(output1, output2))

    def test_roundtrip_detdata(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        n_samp = 100000

        rng = np.random.default_rng(12345)

        comp_types = ["none", "gzip"]
        if have_flac_support():
            comp_types.append("flac")
        else:
            print("FLAC disabled, skipping 'flac' compression type")

        for comp_type in comp_types:
            for dtname, dt in self.types.items():
                # Use fake data with multiple elements per sample, just to test
                # correct handling.
                detdata = DetectorData(
                    self.fakedets,
                    (n_samp, 4),
                    dtype=dt,
                    units=u.K,
                )
                if dtname == "f32" or dtname == "f64":
                    detdata.flatdata[:] = rng.random(
                        size=(4 * n_samp * len(self.fakedets)), dtype=dt
                    )
                else:
                    rng_max = np.iinfo(np.int32).max // 2
                    detdata.flatdata[:] = rng.integers(
                        0,
                        rng_max,
                        size=(4 * n_samp * len(self.fakedets)),
                        dtype=dt,
                    )

                # print(
                #     f"Uncompressed {comp_type}:{dtname} is {detdata.memory_use()} bytes"
                # )
                comp_data = compress_detdata(detdata, {"type": comp_type})
                # print(f"  Compressed {comp_type}:{dtname} is {len(comp_data[0])} bytes")
                new_detdata = decompress_detdata(
                    comp_data[0], comp_data[1], comp_data[2]
                )
                check = np.allclose(new_detdata[:], detdata[:], atol=1.0e-5)
                if not check:
                    print(f"Orig:  {detdata}")
                    print(f"New:  {new_detdata}")
                    self.assertTrue(False)
                del new_detdata
                del detdata

    def create_data(self):
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight="boresight_azel", quats="quats_azel"
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model="noise_model",
            out_model="el_weighted",
            detector_pointing=detpointing_azel,
        )
        el_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(noise_model=el_model.out_model)
        sim_noise.apply(data)

        # Simulate atmosphere
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
        )
        sim_atm.apply(data)

        # Delete temporary object.
        for ob in data.obs:
            del ob.detdata["quats_azel"]

        return data

    def test_roundtrip_memory(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        data = self.create_data()

        comp_types = ["none", "gzip"]
        if have_flac_support():
            comp_types.append("flac")
        else:
            print("FLAC disabled, skipping 'flac' compression type")

        for comp_type in comp_types:
            # Extract compressed versions of signal and flags
            for ob in data.obs:
                for key in [defaults.det_data]:
                    # msg = f"{ob.name} uncompressed {comp_type}:{key} is "
                    # msg += f"{ob.detdata[key].memory_use()} bytes"
                    # print(msg)
                    comp_data = compress_detdata(ob.detdata[key], {"type": comp_type})
                    # msg = f"{ob.name}   compressed {comp_type}:{key} is "
                    # msg += f"{len(comp_data[0])} bytes"
                    # print(msg)
                    new_detdata = decompress_detdata(
                        comp_data[0], comp_data[1], comp_data[2]
                    )
                    check = np.allclose(new_detdata[:], ob.detdata[key][:], atol=1.0e-5)
                    if not check:
                        print(f"Orig:  {ob.detdata[key]}")
                        print(f"New:  {new_detdata}")
                        self.assertTrue(False)

                if comp_type == "flac":
                    continue

                comp_data = compress_detdata(
                    ob.detdata[defaults.det_flags], {"type": comp_type}
                )
                new_detdata = decompress_detdata(
                    comp_data[0], comp_data[1], comp_data[2]
                )
                if new_detdata != ob.detdata[defaults.det_flags]:
                    print(f"Orig:  {ob.detdata[defaults.det_flags]}")
                    print(f"New:  {new_detdata}")
                    self.assertTrue(False)

        close_data(data)

    def test_roundtrip_hdf5(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        testdir = os.path.join(self.outdir, "test_hdf5")
        if rank == 0:
            os.makedirs(testdir)
        nocompdir = os.path.join(self.outdir, "test_hdf5_nocomp")
        if rank == 0:
            os.makedirs(nocompdir)

        data = self.create_data()

        obfiles = list()
        for obs in data.obs:
            _ = save_hdf5(
                obs,
                nocompdir,
                meta=None,
                detdata=[
                    defaults.det_data,
                    defaults.det_flags,
                ],
                shared=None,
                intervals=None,
                config=None,
                times=defaults.times,
                force_serial=False,
                detdata_float32=True,
            )
            if have_flac_support():
                dcomp = (defaults.det_data, {"type": "flac"})
            else:
                print("FLAC disabled, default to detdata compression='none'")
                dcomp = (defaults.det_data, {"type": "none"})
            obf = save_hdf5(
                obs,
                testdir,
                meta=None,
                detdata=[
                    dcomp,
                    (defaults.det_flags, {"type": "gzip"}),
                ],
                shared=None,
                intervals=None,
                config=None,
                times=defaults.times,
                force_serial=False,
                detdata_float32=False,
            )
            obfiles.append(obf)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Load the data and check
        check_data = Data(comm=data.comm)

        for hfile in obfiles:
            check_data.obs.append(load_hdf5(hfile, check_data.comm))

        # Verify.  The other unit tests will check general HDF5 I/O in the case without
        # compression.  Here we are testing the round trip of DetectorData objects.
        for ob, orig in zip(check_data.obs, data.obs):
            if ob.detdata[defaults.det_flags] != orig.detdata[defaults.det_flags]:
                msg = f"---- Proc {data.comm.world_rank} flags not equal ---\n"
                msg += f"{orig.detdata[defaults.det_flags]}\n"
                msg += f"{ob.detdata[defaults.det_flags]}"
                print(msg)
                self.assertTrue(False)
            if not np.allclose(
                ob.detdata[defaults.det_data],
                orig.detdata[defaults.det_data],
                atol=1.0e-4,
                rtol=1.0e-5,
            ):
                msg = f"---- Proc {data.comm.world_rank} signal not equal ---\n"
                msg += f"{orig.detdata[defaults.det_data]}\n"
                msg += f"{ob.detdata[defaults.det_data]}"
                print(msg)
                self.assertTrue(False)

        close_data(data)

    def test_hdf5_verify(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        testdir = os.path.join(self.outdir, "verify_hdf5")
        if rank == 0:
            os.makedirs(testdir)

        data = self.create_data()

        if have_flac_support():
            dcomp = (defaults.det_data, {"type": "flac"})
        else:
            print("FLAC disabled, default to detdata compression='none'")
            dcomp = (defaults.det_data, {"type": "none"})

        saver = ops.SaveHDF5(
            volume=testdir,
            detdata=[
                dcomp,
                (defaults.det_flags, {"type": "gzip"}),
            ],
            detdata_float32=True,
            verify=True,
        )
        saver.apply(data)

        close_data(data)
