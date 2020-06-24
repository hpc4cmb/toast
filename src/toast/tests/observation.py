# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import numpy as np
import numpy.testing as nt

from ..instrument import Focalplane, Telescope

from ..observation import DetectorData, Observation

from ..mpi import Comm, MPI

from ._helpers import create_outdir, create_distdata


class ObservationTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.data = create_distdata(self.comm, obs_per_group=1)

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
                tdata.clear()

    def test_observation(self):
        # Populate the observations
        rms = 10.0
        for obs in self.data.obs:
            n_samp = obs.n_local
            dets = obs.local_detectors
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
                times = np.arange(n_samp)

            # Construct some default shared objects from local buffers
            obs.shared.create("boresight_azel", original=bore, comm=obs.comm_col)
            obs.shared.create("boresight_radec", original=bore, comm=obs.comm_col)
            obs.shared.create("flags", original=common_flags, comm=obs.comm_col)
            obs.shared.create("timestamps", original=times, comm=obs.comm_col)

            # Allocate the default detector data and flags
            obs.detdata.create("signal", shape=(n_samp,), dtype=np.float64)
            obs.detdata.create("flags", shape=(n_samp,), dtype=np.uint16)

            # Allocate some other detector data
            obs.detdata.create("calibration", shape=(n_samp,), dtype=np.float32)
            obs.detdata.create("sim_noise", shape=(n_samp,), dtype=np.float64)

            # Store some values for detector data
            for det in dets:
                obs.detdata["signal"][det, :] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                )
                obs.detdata["calibration"][det, :] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                ).astype(np.float32)
                obs.detdata["sim_noise"][det, :] = np.random.normal(
                    loc=0.0, scale=rms, size=n_samp
                )
                obs.detdata["flags"][det, :] = fake_flags

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
            print("\n", obs.detdata["signal"].data, "\n")
