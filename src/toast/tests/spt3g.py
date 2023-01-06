# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..data import Data
from ..mpi import MPI
from ..spt3g import (
    available,
    export_obs,
    export_obs_data,
    export_obs_meta,
    from_g3_array_type,
    from_g3_quats,
    from_g3_scalar_type,
    from_g3_time,
    from_g3_unit,
    import_obs,
    import_obs_data,
    import_obs_meta,
    to_g3_array_type,
    to_g3_map_array_type,
    to_g3_quats,
    to_g3_scalar_type,
    to_g3_time,
    to_g3_unit,
)
from ._helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase

if available:
    from spt3g import core as c3g


class FramePrinter(object):
    def __init__(self, rank):
        self._rank = rank

    def __call__(self, frame):
        if frame is not None and frame.type != c3g.G3FrameType.EndProcessing:
            if frame.type == c3g.G3FrameType.Observation:
                frame_info = str(frame["observation_name"])
                print(
                    f"World rank {self._rank} processing frames for observation {frame_info}"
                )
        return frame


class Spt3gTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.export_shared_names = [
            ("boresight_azel", "boresight_azel", c3g.G3VectorQuat),
            ("boresight_radec", "boresight_radec", c3g.G3VectorQuat),
            ("position", "position", None),
            ("velocity", "velocity", None),
            ("azimuth", "azimuth", None),
            ("elevation", "elevation", None),
            ("hwp_angle", "hwp_angle", None),
            ("flags", "telescope_flags", None),
        ]
        self.export_det_names = [
            ("signal", "signal", None),
            ("flags", "detector_flags", None),
            ("alt_signal", "alt_signal", None),
        ]
        self.export_interval_names = [
            ("scan_leftright", "intervals_scan_leftright"),
            ("turn_leftright", "intervals_turn_leftright"),
            ("scan_rightleft", "intervals_scan_rightleft"),
            ("turn_rightleft", "intervals_turn_rightleft"),
            ("elnod", "intervals_elnod"),
            ("scanning", "intervals_scanning"),
            ("turnaround", "intervals_turnaround"),
            ("sun_up", "intervals_sun_up"),
            ("sun_close", "intervals_sun_close"),
        ]
        self.import_shared_names = [
            ("boresight_azel", "boresight_azel"),
            ("boresight_radec", "boresight_radec"),
            ("position", "position"),
            ("velocity", "velocity"),
            ("azimuth", "azimuth"),
            ("elevation", "elevation"),
            ("hwp_angle", "hwp_angle"),
            ("telescope_flags", "flags"),
        ]
        self.import_det_names = [
            ("signal", "signal"),
            ("detector_flags", "flags"),
            ("alt_signal", "alt_signal"),
        ]
        self.import_interval_names = [
            ("intervals_scan_leftright", "scan_leftright"),
            ("intervals_turn_leftright", "turn_leftright"),
            ("intervals_scan_rightleft", "scan_rightleft"),
            ("intervals_turn_rightleft", "turn_rightleft"),
            ("intervals_elnod", "elnod"),
            ("intervals_scanning", "scanning"),
            ("intervals_turnaround", "turnaround"),
            ("intervals_sun_up", "sun_up"),
            ("intervals_sun_close", "sun_close"),
        ]

    def test_utils(self):
        n_time = 100000
        start = 1628101051.1234567
        fake_times = start + 0.01 * np.arange(n_time, dtype=np.float64)

        gstart = to_g3_time(start)
        check = from_g3_time(gstart)
        self.assertTrue(check == start)

        gtimes = to_g3_time(fake_times)
        check = from_g3_time(gtimes)
        np.testing.assert_allclose(check, fake_times, atol=1.0e-6, rtol=1.0e-15)

        quat = np.array(
            [
                [0.18257419, 0.36514837, 0.54772256, 0.73029674],
                [0.27216553, 0.40824829, 0.54433105, 0.68041382],
            ]
        )
        gquat = to_g3_quats(quat)
        check = from_g3_quats(gquat)
        np.testing.assert_array_equal(check, quat)

    def create_data(self):
        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Delete some problematic intervals that prevent us from getting a
        # round-trip result that matches the original
        for ob in data.obs:
            del ob.intervals["throw_leftright"]
            del ob.intervals["throw_rightleft"]
            del ob.intervals["throw"]

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

        # Make another detdata object with units for testing
        for ob in data.obs:
            ob.detdata.create("alt_signal", dtype=np.float64, units=u.Kelvin)
            ob.detdata["alt_signal"][:] = 2.725 * np.ones(
                (len(ob.local_detectors), ob.n_local_samples)
            )

        # Store the number of frames for each observation and use for a check
        # after exporting.
        ob_n_frames = dict()
        for ob in data.obs:
            n_frames = 0
            first_set = ob.dist.samp_sets[ob.comm.group_rank].offset
            n_set = ob.dist.samp_sets[ob.comm.group_rank].n_elem
            for sset in range(first_set, first_set + n_set):
                for chunk in ob.dist.sample_sets[sset]:
                    n_frames += 1
            if ob.comm_row is not None:
                n_frames = ob.comm_row.allreduce(n_frames, op=MPI.SUM)
            ob_n_frames[ob.name] = n_frames

        return data, ob_n_frames

    def test_import_export(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        data, ob_n_frames = self.create_data()

        # The default export / import classes
        meta_exporter = export_obs_meta(
            noise_models=[
                ("noise_model", "noise_model"),
                ("el_weighted", "el_weighted"),
            ]
        )
        data_exporter = export_obs_data(
            shared_names=self.export_shared_names,
            det_names=self.export_det_names,
            interval_names=self.export_interval_names,
            compress=False,
        )

        exporter = export_obs(
            meta_export=meta_exporter,
            data_export=data_exporter,
        )

        meta_importer = import_obs_meta(
            noise_models=[
                ("noise_model", "noise_model"),
                ("el_weighted", "el_weighted"),
            ]
        )
        data_importer = import_obs_data(
            shared_names=self.import_shared_names,
            det_names=self.import_det_names,
            interval_names=self.import_interval_names,
        )
        importer = import_obs(
            data.comm,
            meta_import=meta_importer,
            data_import=data_importer,
        )

        # Export the data, and make a copy for later comparison.
        original = list()
        g3data = list()
        for ob in data.obs:
            original.append(ob.duplicate(times="times"))
            obframes = exporter(ob)
            # There should be the original number of frame intervals plus
            # one observation frame and one calibration frame
            checktot = len(obframes) - 2
            if ob.comm_row is not None:
                checktot = ob.comm_row.allreduce(checktot, op=MPI.SUM)
            if checktot != ob_n_frames[ob.name]:
                msg = f"proc {data.comm.world_rank} on ob {ob.name} "
                msg += f"has {checktot} scan frames, "
                msg += f"not {ob_n_frames[ob.name]}"
                print(msg, flush=True)
                self.assertTrue(False)
            g3data.append(obframes)

        # Import the data
        check_data = Data(comm=data.comm)

        for obframes in g3data:
            check_data.obs.append(importer(obframes))

        for ob in check_data.obs:
            ob.redistribute(ob.comm.group_size, times="times")

        # Verify
        for ob, orig in zip(check_data.obs, original):
            if ob != orig:
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
            self.assertTrue(ob == orig)
        close_data(data)

    def test_save_load_no_compression(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        data, ob_n_frames = self.create_data()

        # The default export / import classes
        meta_exporter = export_obs_meta(
            noise_models=[
                ("noise_model", "noise_model"),
                ("el_weighted", "el_weighted"),
            ]
        )
        data_exporter = export_obs_data(
            shared_names=self.export_shared_names,
            det_names=self.export_det_names,
            interval_names=self.export_interval_names,
            compress=False,
        )

        exporter = export_obs(
            meta_export=meta_exporter,
            data_export=data_exporter,
            export_rank=0,
        )

        meta_importer = import_obs_meta(
            noise_models=[
                ("noise_model", "noise_model"),
                ("el_weighted", "el_weighted"),
            ]
        )
        data_importer = import_obs_data(
            shared_names=self.import_shared_names,
            det_names=self.import_det_names,
            interval_names=self.import_interval_names,
        )
        importer = import_obs(
            data.comm,
            meta_import=meta_importer,
            data_import=data_importer,
            import_rank=0,
        )

        # Export the data, and make a copy for later comparison.

        save_dir = os.path.join(self.outdir, "test_nocomp")
        dumper = ops.SaveSpt3g(
            directory=save_dir, framefile_mb=0.5, obs_export=exporter
        )
        loader = ops.LoadSpt3g(directory=save_dir, obs_import=importer)

        dumper.apply(data)
        check_data = Data(comm=data.comm)
        loader.apply(check_data)

        # Verify
        for ob, orig in zip(check_data.obs, data.obs):
            if ob != orig:
                print(f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n{ob}")
            self.assertTrue(ob == orig)
        close_data(data)

    def test_run_g3pipe(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        data, ob_n_frames = self.create_data()

        # The default export / import classes
        meta_exporter = export_obs_meta(
            noise_models=[
                ("noise_model", "noise_model"),
                ("el_weighted", "el_weighted"),
            ]
        )
        data_exporter = export_obs_data(
            shared_names=self.export_shared_names,
            det_names=self.export_det_names,
            interval_names=self.export_interval_names,
            compress=False,
        )

        exporter = export_obs(
            meta_export=meta_exporter,
            data_export=data_exporter,
        )

        meta_importer = import_obs_meta(
            noise_models=[
                ("noise_model", "noise_model"),
                ("el_weighted", "el_weighted"),
            ]
        )
        data_importer = import_obs_data(
            shared_names=self.import_shared_names,
            det_names=self.import_det_names,
            interval_names=self.import_interval_names,
        )
        importer = import_obs(
            data.comm,
            meta_import=meta_importer,
            data_import=data_importer,
        )

        # Create a trivial G3 Pipeline for testing

        fprinter = FramePrinter(data.comm.world_rank)

        # Run it
        runner = ops.RunSpt3g(
            obs_export=exporter, obs_import=importer, modules=[(fprinter, None)]
        )

        runner.apply(data)

        close_data(data)
