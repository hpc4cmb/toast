# Copyright (c) 2015-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os
import re

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..config import build_config
from ..data import Data
from ..io import load_hdf5, save_hdf5, VolumeIndex
from ..utils import replace_byte_arrays, array_equal
from ..weather import Weather
from .helpers import close_data, create_ground_data, create_outdir, create_comm
from .mpi import MPITestCase


class ExtraMeta(object):
    """Class to test Observation attribute save / load."""

    def __init__(self):
        self._data = np.random.normal(size=100)
        self.meta = {"key": self._data}

    @property
    def data(self):
        return self._data

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
    uarr = np.array(["abc", "defg", "hijklm", "no"], dtype=np.dtype("U6"))
    sarr = np.array([b"abc", b"defg", b"hijklm", b"no"], dtype=np.dtype("|S6"))

    def _leaf_dict():
        return {
            "scalar": scalar,
            "qscalar": qscalar,
            "arr": arr,
            "qarr": qarr,
            "uarr": uarr,
            "sarr": sarr,
        }

    def _leaf_list():
        return [
            scalar,
            qscalar,
            arr,
            qarr,
            uarr,
            sarr,
        ]

    def _leaf_tuple():
        return (
            scalar,
            qscalar,
            arr,
            qarr,
            uarr,
            sarr,
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

    def create_data(self, split=False, base_weather=False, no_meta=False, **kwargs):
        # Create fake observing of a small patch.  Use a multifrequency
        # focalplane so we can test split sessions.

        ppp = 2
        freq_list = [(100 + 10 * x) * u.GHz for x in range(3)]
        data = create_ground_data(
            self.comm,
            freqs=freq_list,
            pixel_per_process=ppp,
            split=split,
            **kwargs,
        )

        # Add extra metadata attribute.  Since we are going to test equality on the
        # observation metadata when saving to HDF5 and loading back in, we convert
        # any bytestrings to unicode first.
        if not no_meta:
            raw_other = create_other_meta()
            other = replace_byte_arrays(raw_other)
            for ob in data.obs:
                ob.extra = ExtraMeta()
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
                if not orig.__eq__(ob, approx=True):
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

        # Disable index for this test, to check that the loader works in this case
        saver = ops.SaveHDF5(
            volume=datadir,
            volume_index=None,
            detdata=det_data_fields,
            config=config,
            verify=True,
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
            if not orig.__eq__(ob, approx=True):
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
            if not orig.__eq__(ob, approx=True):
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
            if not orig.__eq__(ob, approx=True):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)
        del check_data

        close_data(data)

    def test_save_load_session_dirs(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_session")
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
            volume=datadir,
            session_dirs=True,
            unix_time_dirs=True,
            detdata=det_data_fields,
            config=config,
            verify=True,
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
            if not orig.__eq__(ob, approx=True):
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
            if not orig.__eq__(ob, approx=True):
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
            if not orig.__eq__(ob, approx=True):
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

        # Version 1 did not save per-detector flags in the observation to HDF5,
        # so we disable them for this test.
        data, config = self.create_data(split=True, no_meta=True, flagged_pixels=False)
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
            if not orig.__eq__(ob, approx=True):
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
                self.assertTrue(False)

        close_data(data)

    def test_index(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "index")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        # Generate the data and create the index using a single group

        data, config = self.create_data(split=True, single_group=True)
        det_data_fields = [
            ("signal", {"quanta": 1.0e-7}),
            ("flags", {}),
            ("alt_signal", {"quanta": 1.0e-12}),
        ]

        # Write data with NO index
        saver = ops.SaveHDF5(
            volume=datadir,
            volume_index=None,
            session_dirs=True,
            unix_time_dirs=True,
            detdata=det_data_fields,
            config=config,
            verify=True,
        )
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Reference different types of metadata for testing the index
        index_fields = {
            "extra_data": ".extra.data:mean",
            "extra_key": ".extra.meta[key]:median",
            "leaf_scalar": "[top_tuple][0][qarr]:mean",
        }
        check_leaf = data.obs[0]["top_tuple"][0]["qarr"]
        all_obs_names = list()
        all_sessions = set()
        obs_sessions = dict()
        check_times = dict()
        for ob in data.obs:
            all_sessions.add(ob.session.name)
            all_obs_names.append(ob.name)
            if ob.session.name not in obs_sessions:
                obs_sessions[ob.session.name] = set()
            if ob.session.name not in check_times:
                check_times[ob.session.name] = np.mean(ob.shared["times"].data)
            obs_sessions[ob.session.name].add(ob.name)
        all_obs_names = set(all_obs_names)

        # Create an index and close our data.
        iname = os.path.join(datadir, VolumeIndex.default_name)
        vindx = VolumeIndex(iname)
        vindx.reindex(datadir, toastcomm=data.comm, indexfields=index_fields)
        del vindx
        close_data(data)

        # Now test that we can use the index independently and that queries
        # make sense.
        vindx = VolumeIndex(iname)

        # Verify that we can load the index with multiple groups and with
        # the world communicator
        toastcomm = create_comm(self.comm, single_group=False)

        # Select everything.  Should return all obs names.
        for test_comm in [toastcomm.comm_world, toastcomm.comm_group]:
            sel_all = vindx.select("select name from observations", comm=test_comm)
            sel_all = set([x[0] for x in sel_all])
            if sel_all != all_obs_names:
                msg = f"select all returned {sel_all} not {all_obs_names}"
                print(msg, flush=True)
                self.assertTrue(False)
            if toastcomm.comm_world is not None:
                toastcomm.comm_world.barrier()

        # Select by session.  There are 3 observations (at different frequencies)
        # per session.
        for test_comm in [toastcomm.comm_world, toastcomm.comm_group]:
            for ses in sorted(all_sessions):
                sel_str = f"select name from observations where session = '{ses}'"
                sel_session = vindx.select(
                    sel_str,
                    comm=test_comm,
                )
                sel_session = set([x[0] for x in sel_session])
                if sel_session != obs_sessions[ses]:
                    msg = f"select session {ses} returned {sel_session} "
                    msg += f"not {obs_sessions[ses]}"
                    print(msg, flush=True)
                    self.assertTrue(False)
            if toastcomm.comm_world is not None:
                toastcomm.comm_world.barrier()

        # Select by arbitrary metadata (constant in all obs in this case)
        check_val = np.mean(check_leaf.value)
        check_low = 0.9 * check_val
        check_high = 1.1 * check_val
        sel_str = "select name from observations where "
        sel_str += f"leaf_scalar > {check_low} and "
        sel_str += f"leaf_scalar < {check_high}"
        for test_comm in [toastcomm.comm_world, toastcomm.comm_group]:
            sel_leaf = vindx.select(sel_str, comm=test_comm)
            sel_leaf = set([x[0] for x in sel_leaf])
            if sel_all != all_obs_names:
                msg = f"select leaf_scalar returned {sel_leaf} not {all_obs_names}"
                print(msg, flush=True)
                self.assertTrue(False)
            if toastcomm.comm_world is not None:
                toastcomm.comm_world.barrier()

        # Select by a timestamp in the center of each session.
        for test_comm in [toastcomm.comm_world, toastcomm.comm_group]:
            for ses in sorted(check_times.keys()):
                timestamp = check_times[ses]
                sel_str = "select name from observations where "
                sel_str += f"start < {timestamp} and "
                sel_str += f"end > {timestamp}"
                sel_time = vindx.select(sel_str, comm=test_comm)
                sel_time = set([x[0] for x in sel_time])
                if sel_time != obs_sessions[ses]:
                    msg = f"select time {ses} returned {sel_time} "
                    msg += f"not {obs_sessions[ses]}"
                    print(msg, flush=True)
                    self.assertTrue(False)
            if toastcomm.comm_world is not None:
                toastcomm.comm_world.barrier()

    def _allreduce_obs_props(self, dt):
        """Helper function to get observation and session mapping.
        This is needed to find the information across all process groups and
        communicate it to all processes for checks below.
        """
        local_time = {x.name: np.mean(x.shared["times"].data) for x in dt.obs}
        local_props = [(x.name, x.session.name, local_time[x.name]) for x in dt.obs]
        all_props = None
        if dt.comm.group_rank == 0:
            if dt.comm.comm_group_rank is not None:
                proc_props = dt.comm.comm_group_rank.gather(local_props, root=0)
                if dt.comm.comm_group_rank.rank == 0:
                    all_props = list()
                    for pprops in proc_props:
                        all_props.extend(pprops)
                all_props = dt.comm.comm_group_rank.bcast(all_props, root=0)
            else:
                all_props = local_props
        if dt.comm.comm_group is not None:
            all_props = dt.comm.comm_group.bcast(all_props)
        # Build the lookup tables
        a_obs = set([x[0] for x in all_props])
        a_sessions = set([x[1] for x in all_props])
        s_obs = dict()
        s_times = dict()
        for oname, sname, stime in all_props:
            if sname not in s_obs:
                s_obs[sname] = set()
            if sname not in s_times:
                s_times[sname] = stime
            s_obs[sname].add(oname)
        return a_obs, a_sessions, s_obs, s_times

    def test_save_load_index(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "save_load_index")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        # Generate the data

        data, config = self.create_data(split=True)
        det_data_fields = [
            ("signal", {"quanta": 1.0e-7}),
            ("flags", {}),
            ("alt_signal", {"quanta": 1.0e-12}),
        ]

        def _allreduce_obs_props(dt):
            """Helper function to get observation and session mapping.
            This is needed to find the information across all process groups and
            communicate it to all processes for checks below.
            """
            local_time = {x.name: np.mean(x.shared["times"].data) for x in dt.obs}
            local_props = [(x.name, x.session.name, local_time[x.name]) for x in dt.obs]
            all_props = None
            if dt.comm.group_rank == 0:
                if dt.comm.comm_group_rank is not None:
                    proc_props = dt.comm.comm_group_rank.gather(local_props, root=0)
                    if dt.comm.comm_group_rank.rank == 0:
                        all_props = list()
                        for pprops in proc_props:
                            all_props.extend(pprops)
                    all_props = dt.comm.comm_group_rank.bcast(all_props, root=0)
                else:
                    all_props = local_props
            if dt.comm.comm_group is not None:
                all_props = dt.comm.comm_group.bcast(all_props)
            # Build the lookup tables
            a_obs = set([x[0] for x in all_props])
            a_sessions = set([x[1] for x in all_props])
            s_obs = dict()
            s_times = dict()
            for oname, sname, stime in all_props:
                if sname not in s_obs:
                    s_obs[sname] = set()
                if sname not in s_times:
                    s_times[sname] = stime
                s_obs[sname].add(oname)
            return a_obs, a_sessions, s_obs, s_times

        (
            orig_obs,
            orig_sessions,
            orig_ses_obs,
            orig_ses_times,
        ) = self._allreduce_obs_props(data)

        # The fields we will index

        index_fields = {
            "extra_data": ".extra.data:mean",
            "extra_key": ".extra.meta[key]:median",
            "leaf_scalar": "[top_tuple][0][qarr]:mean",
        }
        check_leaf = data.obs[0]["top_tuple"][0]["qarr"]

        # Save data and create the index

        saver = ops.SaveHDF5(
            volume=datadir,
            volume_index_fields=index_fields,
            session_dirs=True,
            unix_time_dirs=True,
            detdata=det_data_fields,
            config=config,
            verify=True,
        )
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Now load the data with various selections and confirm the expected
        # set of observations are loaded.

        # Select everything (using the index).  Should load all obs.

        check_data = Data(data.comm)
        loader = ops.LoadHDF5(
            volume=datadir,
            volume_select=None,
        )
        loader.apply(check_data)

        (loaded_obs, loaded_sessions, loaded_ses_obs, loaded_ses_times) = (
            _allreduce_obs_props(check_data)
        )
        if loaded_obs != orig_obs:
            msg = f"select all returned {loaded_obs} not {orig_obs}"
            print(msg, flush=True)
            self.assertTrue(False)
        del check_data

        # Check that we can load individual sessions

        for ses in sorted(orig_sessions):
            check_data = Data(data.comm)
            loader.volume_select = f"where session = '{ses}'"
            loader.apply(check_data)
            (loaded_obs, loaded_sessions, loaded_ses_obs, loaded_ses_times) = (
                _allreduce_obs_props(check_data)
            )

            if loaded_obs != orig_ses_obs[ses]:
                msg = f"select {ses} returned {loaded_obs} not {orig_ses_obs[ses]}"
                print(msg, flush=True)
                self.assertTrue(False)
            del check_data

        # Check that we can load observations based on some metadata.
        check_val = np.mean(check_leaf.value)
        check_low = 0.9 * check_val
        check_high = 1.1 * check_val
        sel_str = "where "
        sel_str += f"leaf_scalar > {check_low} and "
        sel_str += f"leaf_scalar < {check_high}"

        check_data = Data(data.comm)
        loader.volume_select = sel_str
        loader.apply(check_data)
        (loaded_obs, loaded_sessions, loaded_ses_obs, loaded_ses_times) = (
            _allreduce_obs_props(check_data)
        )

        if loaded_obs != orig_obs:
            msg = f"select on common meta returned {loaded_obs} not {orig_obs}"
            print(msg, flush=True)
            self.assertTrue(False)
        del check_data

        # Check we can load observations based on session times
        for ses in sorted(orig_ses_times.keys()):
            timestamp = orig_ses_times[ses]
            sel_str = "where "
            sel_str += f"start < {timestamp} and "
            sel_str += f"end > {timestamp}"

            check_data = Data(data.comm)
            loader.volume_select = sel_str
            loader.apply(check_data)
            (
                loaded_obs,
                loaded_sessions,
                loaded_ses_obs,
                loaded_ses_times,
            ) = _allreduce_obs_props(check_data)

            if loaded_obs != orig_ses_obs[ses]:
                msg = f"select time {ses} returned {loaded_obs} not {orig_ses_obs[ses]}"
                print(msg, flush=True)
                self.assertTrue(False)
            del check_data

        close_data(data)

    def test_det_select(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        datadir = os.path.join(self.outdir, "det_select")
        if rank == 0:
            os.makedirs(datadir)
        if self.comm is not None:
            self.comm.barrier()

        # Generate the data

        data, config = self.create_data(split=False)
        det_data_fields = [
            ("signal", {"quanta": 1.0e-7}),
            ("flags", {}),
            ("alt_signal", {"quanta": 1.0e-12}),
        ]

        # The fields we will index

        index_fields = {
            "extra_data": ".extra.data:mean",
            "extra_key": ".extra.meta[key]:median",
            "leaf_scalar": "[top_tuple][0][qarr]:mean",
        }

        # Save data and create the index

        saver = ops.SaveHDF5(
            volume=datadir,
            volume_index_fields=index_fields,
            session_dirs=True,
            unix_time_dirs=True,
            detdata=det_data_fields,
            config=config,
            verify=True,
        )
        saver.apply(data)

        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Now load the data with a detector selection.

        check_data = Data(data.comm)
        loader = ops.LoadHDF5(
            volume=datadir,
            volume_select=None,
            det_select={"pixel": r"D.*[02468]"},
        )
        loader.apply(check_data)

        # Verify that loaded observations have the expected set of detectors.
        pat = re.compile(r"D.*[13579]")
        for obs, orig in zip(check_data.obs, data.obs):
            upixels = np.unique(obs.telescope.focalplane.detector_data["pixel"])
            for pix in upixels:
                if pat.match(pix) is not None:
                    msg = f"Loaded incorrect pixel '{pix}' from {obs.name}"
                    print(msg, flush=True)
                    self.assertTrue(False)
            for det in obs.local_detectors:
                self.assertTrue(
                    array_equal(
                        obs.detdata["signal"][det],
                        orig.detdata["signal"][det],
                        f32=True,
                    )
                )

        del check_data
        close_data(data)
