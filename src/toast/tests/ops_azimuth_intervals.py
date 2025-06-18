# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
from datetime import datetime

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..data import Data
from ..instrument import Focalplane, GroundSite, Telescope
from ..instrument_sim import fake_hexagon_focalplane
from ..mpi import MPI, Comm
from ..observation import Observation
from ..observation import default_values as defaults
from ..ops.sim_ground_utils import scan_between
from ..schedule import GroundSchedule
from ..schedule_sim_ground import run_scheduler
from ..vis import set_matplotlib_backend
from .helpers import (
    close_data,
    create_comm,
    create_ground_telescope,
    create_outdir,
)
from .mpi import MPITestCase


class AzimuthIntervalsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def create_fake_data(self):
        np.random.seed(123456)
        # Just one group with all processes
        toastcomm = create_comm(self.comm, single_group=True)

        rate = 100.0 * u.Hz

        telescope = create_ground_telescope(
            toastcomm.group_size,
            sample_rate=rate,
            pixel_per_process=1,
            fknee=None,
            freqs=None,
            width=5.0 * u.degree,
        )

        data = Data(toastcomm)

        # 8 minutes
        n_samp = int(8 * 60 * rate.to_value(u.Hz))
        n_parked = int(0.1 * n_samp)
        n_scan = int(0.12 * n_samp)

        ob = Observation(toastcomm, telescope, n_samples=n_samp, name="aztest")
        # Create shared objects for timestamps, common flags, boresight, position,
        # and velocity.
        ob.shared.create_column(
            defaults.times,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
        ob.shared.create_column(
            defaults.shared_flags,
            shape=(ob.n_local_samples,),
            dtype=np.uint8,
        )
        ob.shared.create_column(
            defaults.azimuth,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
        ob.shared.create_column(
            defaults.elevation,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )

        # Rank zero of each grid column creates the data
        stamps = None
        azimuth = None
        elevation = None
        flags = None
        scans = None
        if ob.comm_col_rank == 0:
            start_time = 0.0 + float(ob.local_index_offset) / rate.to_value(u.Hz)
            stop_time = start_time + float(ob.n_local_samples - 1) / rate.to_value(u.Hz)
            stamps = np.linspace(
                start_time,
                stop_time,
                num=ob.n_local_samples,
                endpoint=True,
                dtype=np.float64,
            )

            scans = (n_samp - n_parked) // n_scan
            sim_scans = scans + 1

            azimuth = np.zeros(ob.n_local_samples, dtype=np.float64)
            elevation = np.radians(45.0) * np.ones(ob.n_local_samples, dtype=np.float64)

            azimuth[:n_parked] = np.pi / 4

            for iscan in range(sim_scans):
                first_samp = iscan * n_scan + n_parked
                if iscan % 2 == 0:
                    azstart = np.pi / 4
                    azstop = 3 * np.pi / 4
                else:
                    azstart = 3 * np.pi / 4
                    azstop = np.pi / 4
                _, az, el = scan_between(
                    stamps[first_samp],
                    azstart,
                    np.pi / 4,
                    azstop,
                    np.pi / 4,
                    np.radians(1.0),  # rad / s
                    np.radians(0.25),  # rad / s^2
                    np.radians(1.0),  # rad / s
                    np.radians(0.25),  # rad / s^2
                    nstep=n_scan,
                )
                if iscan == scans:
                    azimuth[first_samp:] = az[: n_samp - first_samp]
                    elevation[first_samp:] = el[: n_samp - first_samp]
                else:
                    azimuth[first_samp : first_samp + n_scan] = az
                    elevation[first_samp : first_samp + n_scan] = el

            # Add some noise
            scale = 0.00005
            azimuth[:] += np.random.normal(loc=0, scale=scale, size=ob.n_local_samples)
            elevation[:] += np.random.normal(
                loc=0, scale=scale, size=ob.n_local_samples
            )

            # Periodic flagged samples.  Add garbage spikes there.
            flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
            for fspan in range(5):
                flags[:: 1000 + fspan] = defaults.shared_mask_invalid
            bad_samps = flags != 0
            azimuth[bad_samps] = 10.0
            elevation[bad_samps] = -10.0

        if ob.comm_col is not None:
            scans = ob.comm_col.bcast(scans, root=0)

        ob.shared[defaults.times].set(stamps, offset=(0,), fromrank=0)
        ob.shared[defaults.azimuth].set(azimuth, offset=(0,), fromrank=0)
        ob.shared[defaults.elevation].set(elevation, offset=(0,), fromrank=0)
        ob.shared[defaults.shared_flags].set(flags, offset=(0,), fromrank=0)
        data.obs.append(ob)
        return data, scans

    def test_exec(self):
        data, num_scans = self.create_fake_data()

        azint = ops.AzimuthIntervals(
            debug_root=os.path.join(self.outdir, "az_intervals"),
            window_seconds=5.0,
        )
        azint.apply(data)

        for ob in data.obs:
            n_scans = len(ob.intervals[defaults.scanning_interval])
            if n_scans != num_scans + 1:
                msg = f"Found {n_scans} scanning intervals instead of {num_scans}"
                print(msg, flush=True)
                self.assertTrue(False)

        close_data(data)
