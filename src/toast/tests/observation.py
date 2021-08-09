# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os
import sys
import traceback

import numpy as np
import numpy.testing as nt

from astropy import units as u

from pshmem import MPIShared

from ..observation import DetectorData, Observation

from ..mpi import Comm, MPI

from ._helpers import create_outdir, create_satellite_empty, create_ground_data


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

            obs.shared.create(
                "samp_A",
                shape=sample_common.shape,
                dtype=sample_common.dtype,
                comm=obs.comm_col,
            )
            if obs.comm_col_rank == 0:
                obs.shared["samp_A"][:, :] = sample_common
            else:
                obs.shared["samp_A"][None] = None

            self.comm.barrier()

            obs.shared.create(
                "det_A",
                shape=det_common.shape,
                dtype=det_common.dtype,
                comm=obs.comm_row,
            )
            self.comm.barrier()

            if obs.comm_row_rank == 0:
                obs.shared["det_A"][:, :, :, :] = det_common
            else:
                obs.shared["det_A"][None] = None

            self.comm.barrier()

            obs.shared.create(
                "all_A",
                shape=all_common.shape,
                dtype=all_common.dtype,
                comm=obs.comm,
            )
            if obs.comm_rank == 0:
                obs.shared["all_A"][:, :, :] = all_common
            else:
                obs.shared["all_A"][None] = None

            self.comm.barrier()

            obs.shared.create(
                "flg_A",
                shape=flag_common.shape,
                dtype=flag_common.dtype,
                comm=obs.comm_col,
            )
            if obs.comm_col_rank == 0:
                obs.shared["flg_A"][:] = flag_common
            else:
                obs.shared["flg_A"][None] = None

            self.comm.barrier()

            sh = MPIShared(sample_common.shape, sample_common.dtype, obs.comm_col)

            if obs.comm_col_rank == 0:
                sh[:, :] = sample_common
            else:
                sh[None] = None

            obs.shared["samp_B"] = sh

            sh = MPIShared(flag_common.shape, flag_common.dtype, obs.comm_col)
            if obs.comm_col_rank == 0:
                sh[:] = flag_common
            else:
                sh[None] = None
            obs.shared["flg_B"] = sh

            sh = MPIShared(det_common.shape, det_common.dtype, obs.comm_row)
            if obs.comm_row_rank == 0:
                sh[:, :, :, :] = det_common
            else:
                sh[None] = None
            obs.shared["det_B"] = sh

            sh = MPIShared(all_common.shape, all_common.dtype, obs.comm)
            if obs.comm_rank == 0:
                sh[:, :, :] = all_common
            else:
                sh[None] = None
            obs.shared["all_B"] = sh

            # this style of assignment only works for the default obs.comm
            if obs.comm_rank == 0:
                obs.shared["all_C"] = all_common
            else:
                obs.shared["all_C"] = None

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
            obs.shared.create("boresight_azel", shape=(n_samp, 4), comm=obs.comm_col)
            obs.shared["boresight_azel"][:, :] = bore

            obs.shared.create("boresight_radec", shape=(n_samp, 4), comm=obs.comm_col)
            obs.shared["boresight_radec"][:, :] = bore

            obs.shared.create(
                "flags", shape=(n_samp,), dtype=np.uint8, comm=obs.comm_col
            )
            obs.shared["flags"][:] = common_flags

            obs.shared.create(
                "timestamps", shape=(n_samp,), dtype=np.float64, comm=obs.comm_col
            )
            obs.shared["timestamps"][:] = times

            # Create some shared objects over the whole comm
            local_array = None
            if obs.comm_rank == 0:
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
            obs.shared.create(
                "beam_profile",
                shape=(len(dets), 1000, 1000),
                dtype=np.float32,
                comm=obs.comm_row,
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
            ob.shared.create("boresight_azel", shape=(n_samp, 4), comm=ob.comm_col)
            ob.shared["boresight_azel"][:, :] = bore

            ob.shared.create("boresight_radec", shape=(n_samp, 4), comm=ob.comm_col)
            ob.shared["boresight_radec"][:, :] = bore

            ob.shared.create("flags", shape=(n_samp,), dtype=np.uint8, comm=ob.comm_col)
            ob.shared["flags"][:] = common_flags

            ob.shared.create(
                "timestamps", shape=(n_samp,), dtype=np.float64, comm=ob.comm_col
            )
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
            if ob.comm_rank == 0:
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

    def test_redistribute(self):
        # Populate the observations
        np.random.seed(12345)
        rms = 10.0
        data = create_ground_data(self.comm, sample_rate=10 * u.Hz)
        for obs in data.obs:
            n_samp = obs.n_local_samples
            dets = obs.local_detectors
            n_det = len(dets)

            # Create some shared objects over the whole comm
            local_array = None
            if obs.comm_rank == 0:
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
            obs.shared.create(
                "beam_profile",
                shape=(len(dets), 1000, 1000),
                dtype=np.float32,
                comm=obs.comm_row,
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

        # Redistribute, and make a copy for verification later
        original = list()
        for ob in data.obs:
            original.append(ob.duplicate(times="times"))
            ob.redistribute(1, times="times")

        # Verify that the observations are no longer equal
        for ob, orig in zip(data.obs, original):
            self.assertFalse(ob == orig)

        # Redistribute back and verify
        for ob, orig in zip(data.obs, original):
            ob.redistribute(orig.comm_size, times="times")
            self.assertTrue(ob == orig)
