# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os
import sys

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..config import build_config
from ..data import Data
from ..io import load_hdf5, save_hdf5
from ..mpi import MPI
from ..observation_data import DetectorData
from ._helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class IoHdf5Test(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def create_data(self, split=False):
        # Create fake observing of a small patch.  Use a multifrequency
        # focalplane so we can test split sessions.

        ppp = 10
        freq_list = [(100 + 10 * x) * u.GHz for x in range(3)]
        data = create_ground_data(
            self.comm,
            freqs=freq_list,
            pixel_per_process=ppp,
            split=split,
        )

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

        for droot in ["default", "serial"]:
            datadir = os.path.join(self.outdir, f"save_load_{droot}")
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
                obf = save_hdf5(
                    ob, datadir, config=config, force_serial=(droot == "serial")
                )
                obfiles.append(obf)

            if self.comm is not None:
                self.comm.barrier()

            # Import the data
            check_data = Data(comm=data.comm)

            for hfile in obfiles:
                check_data.obs.append(load_hdf5(hfile, check_data.comm))

            # Verify
            for ob, orig in zip(check_data.obs, original):
                if ob != orig:
                    print(
                        f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}"
                    )
                self.assertTrue(ob == orig)

            close_data(data)

    def test_save_load_float32(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_float32")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        data, config = self.create_data()

        # Make a copy for later comparison.  Convert float64 detdata to
        # float32.
        original = dict()
        for ob in data.obs:
            original[ob.name] = ob.duplicate(times="times")
            for field, ddata in original[ob.name].detdata.items():
                if ddata.dtype.char == "d":
                    # Hack in a replacement
                    new_dd = DetectorData(
                        ddata.detectors,
                        ddata.detector_shape,
                        np.float32,
                        units=ddata.units,
                    )
                    new_dd[:] = original[ob.name].detdata[field][:]
                    original[ob.name].detdata._internal[field] = new_dd

        saver = ops.SaveHDF5(volume=datadir, config=config, detdata_float32=True)
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        check_data = Data(data.comm)
        loader = ops.LoadHDF5(volume=datadir)
        loader.apply(check_data)

        # Verify
        for ob in check_data.obs:
            orig = original[ob.name]
            if ob != orig:
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
            self.assertTrue(ob == orig)

        close_data(data)

    def test_save_load_ops(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_ops")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        data, config = self.create_data(split=True)

        # Make a copy for later comparison.
        original = dict()
        for ob in data.obs:
            original[ob.name] = ob.duplicate(times="times")

        saver = ops.SaveHDF5(volume=datadir, config=config, verify=True)
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        check_data = Data(data.comm)
        loader = ops.LoadHDF5(volume=datadir)
        loader.apply(check_data)

        # Verify
        for ob in check_data.obs:
            orig = original[ob.name]
            if ob != orig:
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
            self.assertTrue(ob == orig)
        del check_data

        # Also test loading explicit files
        check_data = Data(data.comm)
        loader.volume = None
        loader.files = glob.glob(f"{datadir}/*.h5")
        loader.apply(check_data)

        for ob in check_data.obs:
            orig = original[ob.name]
            if ob != orig:
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
            self.assertTrue(ob == orig)
        del check_data

        # Also check loading by regex, in this case only one frequency
        check_data = Data(data.comm)
        loader.volume = datadir
        loader.pattern = ".*100\.0-GHz.*\.h5"
        loader.apply(check_data)

        for ob in check_data.obs:
            orig = original[ob.name]
            if ob != orig:
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
            self.assertTrue(ob == orig)
        del check_data

        close_data(data)

    def test_save_load_ops_f32(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_ops_f32")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        data, config = self.create_data(split=True)

        # Make a copy for later comparison.
        original = dict()
        for ob in data.obs:
            original[ob.name] = ob.duplicate(times="times")

        saver = ops.SaveHDF5(
            volume=datadir, config=config, detdata_float32=True, verify=True
        )
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        close_data(data)
