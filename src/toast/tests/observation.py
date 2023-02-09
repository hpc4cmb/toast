# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys
import traceback

import numpy as np
import numpy.testing as nt
from astropy import units as u
from pshmem import MPIShared

from .. import ops
from ..data import Data
from ..mpi import MPI, Comm
from ..observation import DetectorData, Observation
from ..observation import default_values as defaults
from ..observation import set_default_values
from ._helpers import (
    close_data,
    create_ground_data,
    create_outdir,
    create_satellite_empty,
    fake_flags,
)
from .mpi import MPITestCase


class ObservationTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.dets = ["d00", "d01", "d02", "d03"]
        self.shapes = [(10,), (10, 4), (10, 3, 2)]
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

    def test_detdata(self):
        for shp in self.shapes:
            for tname, tp in self.types.items():
                tdata = DetectorData(self.dets, shp, tp)
                # if self.comm is None or self.comm.rank == 0:
                #     print(tdata)
                gdets = tdata.detectors
                for didx, dname in enumerate(gdets):
                    tdata[didx] = didx * np.ones(shp, dtype=tp)
                    sidata = tdata[didx]
                    sndata = tdata[dname]
                    # print(sidata)
                    np.testing.assert_equal(sidata, sndata)
                sdata = tdata[1:-1]
                sdata = tdata[[gdets[0], gdets[-1]]]

                # Get a view
                view = tdata.view((slice(0, 2, 1), slice(0, 2, 1)))

                np.testing.assert_equal(view[1], tdata[1, 0:2])
                vshp = [2]
                for s in shp[1:]:
                    vshp.append(s)
                vshp = tuple(vshp)
                view[1] = 5 * np.ones(vshp, dtype=tp)

                np.testing.assert_equal(view[1], tdata[1, 0:2])

                tdata.clear()

    def test_observation(self):
        # Populate the observations
        np.random.seed(12345)
        rms = 10.0
        data = create_satellite_empty(self.comm, obs_per_group=1, samples=10)
        for obs in data.obs:
            n_samp = obs.n_local_samples
            dets = obs.local_detectors
            n_det = len(dets)

            # Test all the different ways of assigning to shared objects

            sample_common = np.ravel(np.random.random((n_samp, 3))).reshape(-1, 3)
            flag_common = np.zeros(n_samp, dtype=np.uint8)
            det_common = np.random.random((n_det, 3, 4, 5))
            all_common = np.random.random((2, 3, 4))

            # Explicit create_* functions and then assignment from array on one process

            obs.shared.create_column(
                "samp_A",
                shape=sample_common.shape,
                dtype=sample_common.dtype,
            )
            if obs.comm_col_rank == 0:
                obs.shared["samp_A"][:, :] = sample_common
            else:
                obs.shared["samp_A"][None] = None

            obs.shared.create_row(
                "det_A",
                shape=det_common.shape,
                dtype=det_common.dtype,
            )

            if obs.comm_row_rank == 0:
                obs.shared["det_A"][:, :, :, :] = det_common
            else:
                obs.shared["det_A"][None] = None

            obs.shared.create_group(
                "all_A",
                shape=all_common.shape,
                dtype=all_common.dtype,
            )
            if obs.comm.group_rank == 0:
                obs.shared["all_A"][:, :, :] = all_common
            else:
                obs.shared["all_A"][None] = None

            obs.shared.create_column(
                "flg_A",
                shape=flag_common.shape,
                dtype=flag_common.dtype,
            )
            if obs.comm_col_rank == 0:
                obs.shared["flg_A"][:] = flag_common
            else:
                obs.shared["flg_A"][None] = None

            # Create and assign from MPIShared objects with explicit comm type

            sh = MPIShared(sample_common.shape, sample_common.dtype, obs.comm_col)

            if obs.comm_col_rank == 0:
                sh[:, :] = sample_common
            else:
                sh[None] = None

            obs.shared["samp_B"] = (sh, "column")

            sh = MPIShared(flag_common.shape, flag_common.dtype, obs.comm_col)
            if obs.comm_col_rank == 0:
                sh[:] = flag_common
            else:
                sh[None] = None
            obs.shared["flg_B"] = (sh, "column")

            sh = MPIShared(det_common.shape, det_common.dtype, obs.comm_row)
            if obs.comm_row_rank == 0:
                sh[:, :, :, :] = det_common
            else:
                sh[None] = None
            obs.shared["det_B"] = (sh, "row")

            sh = MPIShared(all_common.shape, all_common.dtype, obs.comm.comm_group)
            if obs.comm.group_rank == 0:
                sh[:, :, :] = all_common
            else:
                sh[None] = None
            obs.shared["all_B"] = (sh, "group")

            # Create and assign from array on one process with explicit comm type

            del obs.shared["samp_B"]
            sh = None
            if obs.comm_col_rank == 0:
                sh = sample_common
            obs.shared["samp_B"] = (sh, "column")

            del obs.shared["flg_B"]
            sh = None
            if obs.comm_col_rank == 0:
                sh = flag_common
            obs.shared["flg_B"] = (sh, "column")

            del obs.shared["det_B"]
            sh = None
            if obs.comm_row_rank == 0:
                sh = det_common
            obs.shared["det_B"] = (sh, "row")

            del obs.shared["all_B"]
            sh = None
            if obs.comm.group_rank == 0:
                sh = all_common
            obs.shared["all_B"] = (sh, "group")

            # Create and assign with assumed group comm type

            sh = MPIShared(all_common.shape, all_common.dtype, obs.comm.comm_group)
            if obs.comm.group_rank == 0:
                sh[:, :, :] = all_common
            else:
                sh[None] = None
            obs.shared["all_C"] = sh

            del obs.shared["all_C"]
            sh = None
            if obs.comm.group_rank == 0:
                sh = all_common
            obs.shared["all_C"] = sh

            np.testing.assert_equal(obs.shared["samp_A"][:], sample_common)
            np.testing.assert_equal(obs.shared["samp_B"][:], sample_common)
            np.testing.assert_equal(obs.shared["det_A"][:], det_common)
            np.testing.assert_equal(obs.shared["det_B"][:], det_common)
            np.testing.assert_equal(obs.shared["all_A"][:], all_common)
            np.testing.assert_equal(obs.shared["all_B"][:], all_common)
            np.testing.assert_equal(obs.shared["all_C"][:], all_common)
            np.testing.assert_equal(obs.shared["flg_A"][:], flag_common)
            np.testing.assert_equal(obs.shared["flg_B"][:], flag_common)

            # Test different assignment methods for detdata

            signal = np.random.random((n_samp,))
            pntg = np.ones((n_samp, 4), dtype=np.float32)
            flg = np.zeros((n_samp,), dtype=np.uint8)

            obs.detdata.create(
                "sig_A", sample_shape=(), dtype=signal.dtype, detectors=None
            )
            obs.detdata["sig_A"][:] = np.tile(signal, n_det).reshape((n_det, -1))

            obs.detdata.create(
                "pntg_A", sample_shape=(4,), dtype=pntg.dtype, detectors=None
            )
            obs.detdata["pntg_A"][:] = np.tile(pntg, n_det).reshape((n_det, -1, 4))

            obs.detdata.create(
                "flg_A", sample_shape=None, dtype=flg.dtype, detectors=None
            )
            obs.detdata["flg_A"][:] = np.tile(flg, n_det).reshape((n_det, -1))

            dsig = DetectorData(obs.local_detectors, (n_samp,), np.float64)
            dsig[:] = np.tile(signal, n_det).reshape((n_det, -1))

            dpntg = DetectorData(obs.local_detectors, (n_samp, 4), np.float32)
            dpntg[:] = np.tile(pntg, n_det).reshape((n_det, -1, 4))

            dflg = DetectorData(obs.local_detectors, (n_samp,), np.uint8)
            dflg[:] = np.tile(flg, n_det).reshape((n_det, -1))

            obs.detdata["sig_B"] = dsig
            obs.detdata["pntg_B"] = dpntg
            obs.detdata["flg_B"] = dflg

            obs.detdata["sig_C"] = {d: signal for d in obs.local_detectors}
            obs.detdata["pntg_C"] = {d: pntg for d in obs.local_detectors}
            obs.detdata["flg_C"] = {d: flg for d in obs.local_detectors}

            obs.detdata["sig_D"] = signal
            obs.detdata["pntg_D"] = pntg
            obs.detdata["flg_D"] = flg

            obs.detdata["sig_E"] = np.tile(signal, n_det).reshape((n_det, -1))
            obs.detdata["pntg_E"] = np.tile(pntg, n_det).reshape((n_det, -1, 4))
            obs.detdata["flg_E"] = np.tile(flg, n_det).reshape((n_det, -1))

            # Now add some more normal data

            fake_bore = np.ravel(np.random.random((n_samp, 4))).reshape(-1, 4)
            fake_flags = np.random.uniform(low=0, high=2, size=n_samp).astype(
                np.uint8, copy=True
            )
            bore = None
            common_flags = None
            times = None
            if obs.comm_col_rank == 0:
                bore = fake_bore
                common_flags = fake_flags
                times = np.arange(n_samp, dtype=np.float64)

            # Construct some default shared objects from local buffers
            obs.shared.create_column("boresight_azel", shape=(n_samp, 4))
            obs.shared["boresight_azel"][:, :] = bore

            obs.shared.create_column("boresight_radec", shape=(n_samp, 4))
            obs.shared["boresight_radec"][:, :] = bore

            obs.shared.create_column("flags", shape=(n_samp,), dtype=np.uint8)
            obs.shared["flags"][:] = common_flags

            obs.shared.create_column("timestamps", shape=(n_samp,), dtype=np.float64)
            obs.shared["timestamps"][:] = times

            # Create some shared objects over the whole comm
            local_array = None
            if obs.comm.group_rank == 0:
                local_array = np.arange(100, dtype=np.int16)
            obs.shared["everywhere"] = local_array

            # Allocate the default detector data and flags
            obs.detdata.create("signal", dtype=np.float64)
            obs.detdata.create("flags", sample_shape=(), dtype=np.uint16)

            # Allocate some other detector data
            obs.detdata["calibration"] = np.ones(
                (len(obs.local_detectors), obs.n_local_samples), dtype=np.float64
            )

            obs.detdata["sim_noise"] = np.zeros(
                (obs.n_local_samples,), dtype=np.float64
            )

            obs.detdata["with_units"] = u.Quantity(
                np.ones(obs.n_local_samples, dtype=np.float64),
                u.K,
            )

            # Store some values for detector data
            for det in dets:
                obs.detdata["signal"][det] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                )
                obs.detdata["calibration"][det] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                ).astype(np.float32)
                obs.detdata["sim_noise"][det] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                )
                obs.detdata["flags"][det] = fake_flags

            # Make some shared objects, one per detector, shared across the process
            # rows.
            obs.shared.create_row(
                "beam_profile",
                shape=(len(dets), 1000, 1000),
                dtype=np.float32,
            )
            for didx, det in enumerate(dets):
                beam_data = None
                if obs.comm_row_rank == 0:
                    beam_data = (
                        np.random.uniform(low=0, high=100, size=(1000 * 1000))
                        .astype(np.float32, copy=True)
                        .reshape(1, 1000, 1000)
                    )
                obs.shared["beam_profile"].set(
                    beam_data, offset=(didx, 0, 0), fromrank=0
                )

            # You can access detector data by index or by name
            for didx, det in enumerate(dets):
                np.testing.assert_equal(
                    obs.detdata["signal"][det], obs.detdata["signal"][didx]
                )

            # ... Or you can access it as one big array (first dimension is detector)
            # print("\n", obs.detdata["signal"].data, "\n")

        close_data(data)

    def test_view(self):
        np.random.seed(12345)
        rms = 1.0
        data = create_satellite_empty(self.comm, obs_per_group=1, samples=10)
        for ob in data.obs:
            n_samp = ob.n_local_samples
            dets = ob.local_detectors
            n_det = len(dets)

            # Create some data

            fake_bore = np.ravel(np.random.random((n_samp, 4))).reshape(-1, 4)
            fake_flags = np.random.uniform(low=0, high=2, size=n_samp).astype(
                np.uint8, copy=True
            )
            bore = None
            common_flags = None
            times = None
            if ob.comm_col_rank == 0:
                bore = fake_bore
                common_flags = fake_flags
                times = np.arange(
                    ob.local_index_offset,
                    ob.local_index_offset + n_samp,
                    dtype=np.float64,
                )

            # Construct some default shared objects from local buffers
            ob.shared.create_column("boresight_azel", shape=(n_samp, 4))
            ob.shared["boresight_azel"][:, :] = bore

            ob.shared.create_column("boresight_radec", shape=(n_samp, 4))
            ob.shared["boresight_radec"][:, :] = bore

            ob.shared.create_column("flags", shape=(n_samp,), dtype=np.uint8)
            ob.shared["flags"][:] = common_flags

            ob.shared.create_column("timestamps", shape=(n_samp,), dtype=np.float64)
            ob.shared["timestamps"][:] = times

            # Allocate the default detector data and flags
            ob.detdata.create("signal", dtype=np.float64)
            ob.detdata.create("flags", dtype=np.uint16)

            # Store some values for detector data
            for det in dets:
                ob.detdata["signal"][det] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                )
                ob.detdata["flags"][det] = fake_flags

            # Make some intervals

            bad = None
            if ob.comm.group_rank == 0:
                all_time = np.arange(ob.n_all_samples, dtype=np.float64)
                incr = ob.n_all_samples // 2
                bad = [(float(x * incr), float(x * incr + 2)) for x in range(2)]
            ob.intervals.create("bad", bad, ob.shared["timestamps"], fromrank=0)
            ob.intervals["good"] = ~ob.intervals["bad"]

            # Test global view
            for dd in ["signal", "flags"]:
                np.testing.assert_equal(
                    ob.detdata[dd]._data, ob.view[None].detdata[dd][0]._data
                )
            for sh in ["boresight_azel", "boresight_radec", "flags", "timestamps"]:
                np.testing.assert_equal(ob.shared[sh], ob.view[None].shared[sh][0])

            # Test named views
            good_slices = [slice(x.first, x.last + 1, 1) for x in ob.intervals["good"]]
            bad_slices = [slice(x.first, x.last + 1, 1) for x in ob.intervals["bad"]]

            for dd in ["signal", "flags"]:
                for vw, slc in zip(ob.view["good"].detdata[dd], good_slices):
                    np.testing.assert_equal(ob.detdata[dd][:, slc], vw[:])
                for vw, slc in zip(ob.view["bad"].detdata[dd], bad_slices):
                    np.testing.assert_equal(ob.detdata[dd][:, slc], vw[:])

            for sh in ["boresight_azel", "boresight_radec", "flags", "timestamps"]:
                for vw, slc in zip(ob.view["good"].shared[sh], good_slices):
                    np.testing.assert_equal(ob.shared[sh][slc], vw)
                for vw, slc in zip(ob.view["bad"].shared[sh], bad_slices):
                    np.testing.assert_equal(ob.shared[sh][slc], vw)

            # Modify the original data and verify

            for dd in ["signal", "flags"]:
                ob.detdata[dd][:, -1] = 200
            for sh in ["boresight_azel", "boresight_radec"]:
                dummy = None
                if ob.comm_col_rank == 0:
                    dummy = np.array(
                        [
                            [5.0, 5.0, 5.0, 5.0],
                        ]
                    )
                ob.shared[sh][-1, :] = dummy

            for dd in ["signal", "flags"]:
                np.testing.assert_equal(
                    ob.detdata[dd][:, -1], ob.view["good"].detdata[dd][-1][:, -1]
                )
            for sh in ["boresight_azel", "boresight_radec"]:
                np.testing.assert_equal(
                    ob.shared[sh][-1], ob.view["good"].shared[sh][-1][-1]
                )

            # Modify the view and verify

            for dd in ["signal", "flags"]:
                ob.view["good"].detdata[dd][-1][:, -1] = 100

            for dd in ["signal", "flags"]:
                np.testing.assert_equal(
                    ob.detdata[dd][:, -1], ob.view["good"].detdata[dd][-1][:, -1]
                )
        close_data(data)

    def create_redist_data(self):
        # Populate the observations
        np.random.seed(12345)
        rms = 10.0
        data = create_ground_data(self.comm, sample_rate=10.0 * u.Hz)
        ops.DefaultNoiseModel().apply(data)

        for obs in data.obs:
            n_samp = obs.n_local_samples
            dets = obs.local_detectors
            n_det = len(dets)

            # Delete some problematic intervals that prevent us from getting a
            # round-trip result that matches the original
            del obs.intervals["throw_leftright"]
            del obs.intervals["throw_rightleft"]
            del obs.intervals["throw"]

            # Create some shared objects over the whole comm
            local_array = None
            if obs.comm.group_rank == 0:
                local_array = np.arange(100, dtype=np.int16)
            obs.shared["everywhere"] = local_array

            # Allocate the default detector data
            obs.detdata.ensure("signal", dtype=np.float64, create_units=u.K)
            # and flags.  Default data type (np.uint8) is incompatible
            if "flags" in obs.detdata:
                del obs.detdata["flags"]
            obs.detdata.ensure("flags", sample_shape=(), dtype=np.uint16)

            # Allocate some other detector data
            obs.detdata["calibration"] = np.ones(
                (len(obs.local_detectors), obs.n_local_samples), dtype=np.float64
            )
            obs.detdata["sim_noise"] = np.zeros(
                (obs.n_local_samples,), dtype=np.float64
            )

            # Store some values for detector data
            for det in dets:
                obs.detdata["signal"][det] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                )
                obs.detdata["calibration"][det] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                ).astype(np.float32)
                obs.detdata["sim_noise"][det] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                )
                obs.detdata["flags"][det] = np.random.uniform(
                    low=0, high=2, size=n_samp
                ).astype(np.uint8, copy=True)

            # Make some shared objects, one per detector, shared across the process
            # rows.
            obs.shared.create_row(
                "beam_profile",
                shape=(len(dets), 1000, 1000),
                dtype=np.float32,
            )
            for didx, det in enumerate(dets):
                beam_data = None
                if obs.comm_row_rank == 0:
                    beam_data = (
                        np.random.uniform(low=0, high=100, size=(1000 * 1000))
                        .astype(np.float32, copy=True)
                        .reshape(1, 1000, 1000)
                    )
                obs.shared["beam_profile"].set(
                    beam_data, offset=(didx, 0, 0), fromrank=0
                )

            # Test the redistribution of intervals that align with scanning pattern
            # obs.intervals["frames"] = (
            #     obs.intervals["scanning"] | obs.intervals["turnaround"]
            # )
            # obs.intervals["frames"] |= obs.intervals["elnod"]
        return data

    def test_redistribute(self):
        data = self.create_redist_data()

        # Redistribute, and make a copy for verification later
        original = list()
        for ob in data.obs:
            original.append(ob.duplicate(times=defaults.times))
            ob.redistribute(1, times=defaults.times)
            self.assertTrue(ob.comm_col_size == 1)
            self.assertTrue(ob.comm_row_size == data.comm.group_size)
            self.assertTrue(ob.local_detectors == ob.all_detectors)

        # Verify that the observations are no longer equal- only if we actually
        # have more than one process per observation.
        if data.comm.group_size > 1:
            for ob, orig in zip(data.obs, original):
                self.assertFalse(ob == orig)

        # Redistribute back and verify
        for ob, orig in zip(data.obs, original):
            ob.redistribute(orig.comm.group_size, times=defaults.times)
            if ob != orig:
                print(f"Rank {self.comm.rank}: {orig} != {ob}", flush=True)
            self.assertTrue(ob == orig)

        close_data(data)

    def test_partial_redist(self):
        data = self.create_redist_data()

        # Zero signal values
        for ob in data.obs:
            ob.detdata[defaults.det_data][:] = 0.0

        # Make a copy without redistribution for later checking
        inplace = Data(data.comm)
        for ob in data.obs:
            new_ob = ob.duplicate(times=defaults.times)
            for idet, det in enumerate(new_ob.all_detectors):
                if det in new_ob.local_detectors:
                    new_vals = (
                        idet * new_ob.n_all_samples
                        + new_ob.local_index_offset
                        + np.arange(new_ob.n_local_samples)
                    )
                    new_ob.detdata[defaults.det_data][det] = new_vals
            inplace.obs.append(new_ob)

        for ob in data.obs:
            # Duplicate a subset of data
            proc_rows = ob.dist.process_rows
            temp_ob = ob.duplicate(
                times=defaults.times,
                meta=list(),
                shared=[defaults.boresight_radec, defaults.boresight_azel],
                detdata=[defaults.det_data],
                intervals=list(),
            )

            # Redistribute
            temp_ob.redistribute(1, times=defaults.times, override_sample_sets=None)
            # temp_ob.redistribute(1, times=defaults.times)

            # Modify the signal
            for idet, det in enumerate(temp_ob.all_detectors):
                if det not in temp_ob.local_detectors:
                    raise RuntimeError("Redistributed obs should have all dets")
                new_vals = (
                    idet * temp_ob.n_all_samples
                    + temp_ob.local_index_offset
                    + np.arange(temp_ob.n_local_samples)
                )
                temp_ob.detdata[defaults.det_data][det] = new_vals

            # Distribute back
            temp_ob.redistribute(
                proc_rows,
                times=defaults.times,
                override_sample_sets=ob.dist.sample_sets,
            )

            # Copy into place
            for det in ob.local_detectors:
                ob.detdata[defaults.det_data][det, :] = temp_ob.detdata[
                    defaults.det_data
                ][det, :]

        # Verify
        for ob, check in zip(data.obs, inplace.obs):
            if ob != check:
                print(f"{ob.name}:  {ob} != {check}")
                self.assertTrue(False)

        del inplace
        close_data(data)

    # The code below is here for reference, but we cannot actually test this
    # within the unit test framework.  The reason is that the operator classes
    # have already been imported by other tests (the trait defaults for those classes
    # are set at first import).  In order to swap default names, it must be done
    # before importing toast.ops

    # def test_default_values(self):
    #     # Change defaults
    #     custom = {
    #         "times": "custom_times",
    #         "shared_flags": "custom_flags",
    #         "det_data": "custom_signal",
    #         "det_flags": "custom_flags",
    #         "hwp_angle": "custom_hwp_angle",
    #         "azimuth": "custom_azimuth",
    #         "elevation": "custom_elevation",
    #         "boresight_azel": "custom_boresight_azel",
    #         "boresight_radec": "custom_boresight_radec",
    #         "position": "custom_position",
    #         "velocity": "custom_velocity",
    #         "pixels": "custom_pixels",
    #         "weights": "custom_weights",
    #         "quats": "custom_quats",
    #     }
    #     set_default_values(custom)

    #     from .. import ops as ops

    #     # Create all the data objects
    #     np.random.seed(12345)
    #     rms = 10.0
    #     data = create_ground_data(self.comm, sample_rate=10 * u.Hz)

    #     detpointing = ops.PointingDetectorSimple()
    #     detpointing.apply(data)

    #     default_model = ops.DefaultNoiseModel()
    #     default_model.apply(data)

    #     sim_noise = ops.SimNoise(noise_model=default_model.noise_model)
    #     sim_noise.apply(data)

    #     pointing = ops.PointingHealpix(
    #         nside=64,
    #         mode="IQU",
    #         hwp_angle=defaults.hwp_angle,
    #         detector_pointing=detpointing,
    #     )
    #     pointing.apply(data)

    #     fake_flags(data)

    #     # Verify names
    #     for ob in data.obs:
    #         print(ob)
    #         self.assertTrue(custom["times"] in ob.shared)
    #         self.assertTrue(custom["shared_flags"] in ob.shared)
    #         self.assertTrue(custom["hwp_angle"] in ob.shared)
    #         self.assertTrue(custom["azimuth"] in ob.shared)
    #         self.assertTrue(custom["elevation"] in ob.shared)
    #         self.assertTrue(custom["boresight_azel"] in ob.shared)
    #         self.assertTrue(custom["boresight_radec"] in ob.shared)
    #         self.assertTrue(custom["position"] in ob.shared)
    #         self.assertTrue(custom["velocity"] in ob.shared)
    #         self.assertTrue(custom["det_data"] in ob.detdata)
    #         self.assertTrue(custom["det_flags"] in ob.detdata)
    #         self.assertTrue(custom["pixels"] in ob.detdata)
    #         self.assertTrue(custom["weights"] in ob.detdata)
    #         self.assertTrue(custom["quats"] in ob.detdata)
