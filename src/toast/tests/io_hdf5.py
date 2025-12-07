# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..config import build_config
from ..data import Data
from ..io import load_hdf5, save_hdf5
from ..ops.save_hdf5 import obs_approx_equal
from ..weather import Weather
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class ExtraMeta(object):
    """Class to test Observation attribute save / load."""

    def __init__(self):
        self._data = np.random.normal(size=100)

    def save_hdf5(self, group, obs):
        if group is not None:
            hdata = group.create_dataset(
                "ExtraMeta", self._data.shape, dtype=self._data.dtype
            )
        if obs.comm.group_rank == 0:
            hdata.write_direct(self._data, (slice(0, 100, 1),), (slice(0, 100, 1),))

    def load_hdf5(self, group, obs):
        if group is not None:
            ds = group["ExtraMeta"]
            if obs.comm.group_rank == 0:
                self._data = np.empty(ds.shape, dtype=ds.dtype)
                hslc = tuple([slice(0, x, 1) for x in ds.shape])
                ds.read_direct(self._data, hslc, hslc)
        if obs.comm.comm_group is not None:
            self._data = obs.comm.comm_group.bcast(self._data, root=0)

    def __eq__(self, other):
        if np.allclose(self._data, other._data):
            return True
        else:
            return False


def create_other_meta():
    """Helper function to generate python containers of metadata for testing"""
    # Create nested containers of all types, with scalars and arrays
    scalar = 1.234
    qscalar = 1.234 * u.second
    arr = np.arange(10, dtype=np.float64)
    qarr = arr * u.meter

    def _leaf_dict():
        return {
            "scalar": scalar,
            "qscalar": qscalar,
            "arr": arr,
            "qarr": qarr,
        }

    def _leaf_list():
        return [
            scalar,
            qscalar,
            arr,
            qarr,
        ]

    def _leaf_tuple():
        return (
            scalar,
            qscalar,
            arr,
            qarr,
        )

    def _node_dict():
        return {
            "dict": _leaf_dict(),
            "list": _leaf_list(),
            "tuple": _leaf_tuple(),
        }

    def _node_list():
        return [
            _leaf_dict(),
            _leaf_list(),
            _leaf_tuple(),
        ]

    def _node_tuple():
        return (
            _leaf_dict(),
            _leaf_list(),
            _leaf_tuple(),
        )

    root = {
        "top_dict": _node_dict(),
        "top_list": _node_list(),
        "top_tuple": _node_tuple(),
    }

    return root


class IoHdf5Test(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def create_data(self, split=False, base_weather=False, no_meta=False):
        # Create fake observing of a small patch.  Use a multifrequency
        # focalplane so we can test split sessions.

        ppp = 2
        freq_list = [(100 + 10 * x) * u.GHz for x in range(3)]
        data = create_ground_data(
            self.comm,
            freqs=freq_list,
            pixel_per_process=ppp,
            split=split,
        )

        # Add extra metadata attribute
        if not no_meta:
            for ob in data.obs:
                ob.extra = ExtraMeta()
            other = create_other_meta()
            ob.update(other)

        if base_weather:
            # Replace the simulated weather with the base class for testing
            for ob in data.obs:
                old_weather = ob.telescope.site.weather
                new_weather = Weather(
                    time=old_weather.time,
                    ice_water=old_weather.ice_water,
                    liquid_water=old_weather.liquid_water,
                    pwv=old_weather.pwv,
                    humidity=old_weather.humidity,
                    surface_pressure=old_weather.surface_pressure,
                    surface_temperature=old_weather.surface_temperature,
                    air_temperature=old_weather.air_temperature,
                    west_wind=old_weather.west_wind,
                    south_wind=old_weather.south_wind,
                )
                ob.telescope.site.weather = new_weather
                del old_weather

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
            det_data_fields = ["signal", "flags", "alt_signal"]

            # Export the data, and make a copy for later comparison.
            original = list()
            obfiles = list()
            for ob in data.obs:
                original.append(ob.duplicate(times="times"))
                obf = save_hdf5(
                    ob,
                    datadir,
                    detdata=det_data_fields,
                    config=config,
                    force_serial=(droot == "serial"),
                )
                obfiles.append(obf)

            if self.comm is not None:
                self.comm.barrier()

            # Import the data
            check_data = Data(comm=data.comm)

            for hfile in obfiles:
                check_data.obs.append(
                    load_hdf5(hfile, check_data.comm, detdata=det_data_fields)
                )

            # Verify
            for ob, orig in zip(check_data.obs, original):
                if not obs_approx_equal(ob, orig):
                    print(
                        f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}"
                    )
                    self.assertTrue(False)

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
        det_data_fields = ["signal", "flags", "alt_signal"]

        # Make a copy for later comparison.
        original = dict()
        for ob in data.obs:
            original[ob.name] = ob.duplicate(times="times")

        saver = ops.SaveHDF5(
            volume=datadir, detdata=det_data_fields, config=config, verify=True
        )
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        check_data = Data(data.comm)
        loader = ops.LoadHDF5(volume=datadir, detdata=det_data_fields)
        loader.apply(check_data)

        # Verify
        for ob in check_data.obs:
            orig = original[ob.name]
            if not obs_approx_equal(ob, orig):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)
        del check_data

        # Also test loading explicit files
        check_data = Data(data.comm)
        loader.volume = None
        loader.files = glob.glob(f"{datadir}/*.h5")
        loader.apply(check_data)

        for ob in check_data.obs:
            orig = original[ob.name]
            if not obs_approx_equal(ob, orig):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)
        del check_data

        # Also check loading by regex, in this case only one frequency
        check_data = Data(data.comm)
        loader.files = []
        loader.volume = datadir
        loader.pattern = r".*100\.0-GHz.*\.h5"
        loader.apply(check_data)

        for ob in check_data.obs:
            orig = original[ob.name]
            if not obs_approx_equal(ob, orig):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)
        del check_data

        close_data(data)

    def test_save_load_empty_detdata(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_empty_detdata")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        data, config = self.create_data(split=True, base_weather=True)

        # Set detdata to an empty list so that no detector data is written or loaded.
        det_data_fields = []

        # Make a copy for later comparison.
        original = dict()
        for ob in data.obs:
            original[ob.name] = ob.duplicate(times="times")

        saver = ops.SaveHDF5(
            volume=datadir, detdata=det_data_fields, config=config, verify=True
        )
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        check_data = Data(data.comm)
        loader = ops.LoadHDF5(volume=datadir, detdata=det_data_fields)
        loader.apply(check_data)

        # Verify.  Before checking equality, purge detdata from the original.
        for ob in check_data.obs:
            orig = original[ob.name]
            orig.detdata.clear()
            if not obs_approx_equal(ob, orig):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)
        del check_data

        close_data(data)

    def test_save_load_ops_compression(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_ops_compression")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        data, config = self.create_data(split=True)
        det_data_fields = [
            ("signal", {"quanta": 1.0e-7}),
            ("flags", {}),
            ("alt_signal", {"quanta": 1.0e-12}),
        ]
        det_data_names = ["signal", "flags", "alt_signal"]

        # Make a copy for later comparison.
        original = dict()
        for ob in data.obs:
            original[ob.name] = ob.duplicate(times="times")

        saver = ops.SaveHDF5(
            volume=datadir, detdata=det_data_fields, config=config, verify=True
        )
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        check_data = Data(data.comm)
        loader = ops.LoadHDF5(volume=datadir, detdata=det_data_names)
        loader.apply(check_data)

        # Verify
        for ob in check_data.obs:
            orig = original[ob.name]
            if not obs_approx_equal(ob, orig):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)
        del check_data

        close_data(data)

    def test_save_load_version1(self):
        # Here we test loading an old version 1 format file (the original
        # version 1 saving code is kept around for this purpose).
        from ..io.observation_hdf_save_v1 import save_hdf5 as save_v1

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_v1")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        data, config = self.create_data(split=True, no_meta=True)
        det_data_names = ["signal", "flags", "alt_signal"]
        det_data_fields = [
            ("signal", {"type": "flac", "quanta": 1.0e-7}),
            ("flags", {"type": "gzip"}),
            ("alt_signal", {"type": "flac", "quanta": 1.0e-7}),
        ]

        # Export the data, and make a copy for later comparison.
        original = list()
        obfiles = list()
        for ob in data.obs:
            original.append(ob.duplicate(times="times"))
            obf = save_v1(
                ob,
                datadir,
                detdata=det_data_fields,
                config=config,
                force_serial=False,
                detdata_float32=False,
            )
            obfiles.append(obf)

        if self.comm is not None:
            self.comm.barrier()

        # Import the data
        check_data = Data(comm=data.comm)

        for hfile in obfiles:
            check_data.obs.append(
                load_hdf5(hfile, check_data.comm, detdata=det_data_names)
            )

        # Verify
        for ob, orig in zip(check_data.obs, original):
            if not obs_approx_equal(ob, orig):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)

        close_data(data)
