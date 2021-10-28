# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np

from astropy import units as u

from .mpi import MPITestCase

from .. import ops as ops

from ..mpi import MPI

from ..data import Data

from ..io import save_hdf5

from ..config import build_config

from ._helpers import create_outdir, create_ground_data


class IoHdf5Test(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def create_data(self):
        # Create fake observing of a small patch
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

        config = build_config(
            [
                detpointing_azel,
                default_model,
                el_model,
                sim_noise,
            ]
        )

        # Make another detdata object with units for testing
        for ob in data.obs:
            ob.detdata.create("alt_signal", dtype=np.float64, units=u.Kelvin)
            ob.detdata["alt_signal"][:] = 2.725 * np.ones(
                (len(ob.local_detectors), ob.n_local_samples)
            )

        return data, config

    def test_save_load(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        data, config = self.create_data()

        # Export the data, and make a copy for later comparison.
        original = list()
        obfiles = list()
        for ob in data.obs:
            original.append(ob.duplicate(times="times"))
            try:
                obf = save_hdf5(ob, datadir, config=config)
                obfiles.append(obf)
            except:
                import traceback

                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                lines = [f"Proc {self.comm.rank}: {x}" for x in lines]
                msg = "".join(lines)
                print(msg)

        # # Import the data
        # check_data = Data(comm=data.comm)

        # for obframes in g3data:
        #     check_data.obs.append(importer(obframes))

        # for ob in check_data.obs:
        #     ob.redistribute(ob.comm_size, times="times")

        # # Verify
        # for ob, orig in zip(check_data.obs, original):
        #     if ob != orig:
        #         print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
        #     self.assertTrue(ob == orig)
