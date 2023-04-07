# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..data import Data
from ..observation import default_values as defaults
from ..pixels_io_healpix import write_healpix_fits
from ..schedule import GroundSchedule
from ..schedule_sim_ground import run_scheduler
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_comm,
    create_ground_data,
    create_ground_telescope,
    create_outdir,
)
from .mpi import MPITestCase


class FlagSSOTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_flag(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # We need a custom observing schedule so we cannot use
        # _helpers.create_ground_data()

        toastcomm = create_comm(self.comm)
        fsample = 1 * u.Hz

        tele = create_ground_telescope(
            toastcomm.group_size,
            sample_rate=fsample,
            pixel_per_process=1,
            fknee=None,
            freqs=None,
        )

        if self.comm is None or self.comm.rank == 0:
            schedule_file_rising = os.path.join(self.outdir, "rising.txt")
            schedule_file_setting = os.path.join(self.outdir, "setting.txt")
            common_opts = [
                "--site-name",
                tele.site.name,
                "--telescope",
                tele.name,
                "--site-lon",
                "{}".format(tele.site.earthloc.lon.to_value(u.degree)),
                "--site-lat",
                "{}".format(tele.site.earthloc.lat.to_value(u.degree)),
                "--site-alt",
                "{}".format(tele.site.earthloc.height.to_value(u.meter)),
                "--start",
                "2030-01-25 00:00:00",
                "--stop",
                "2030-01-26 00:00:00",
                "--ces-max-time",
                "3600",
                "--sun-avoidance-angle-deg",
                "0",
                "--moon-avoidance-angle-deg",
                "0",
            ]

            # Rising schedule
            opts = [
                "--patch",
                "RISING_SCAN_40,HORIZONTAL,1.00,30.00,150.00,40.00,60",
                "--out",
                schedule_file_rising,
            ]
            run_scheduler(opts=common_opts + opts)
            schedule_rising = GroundSchedule()
            schedule_rising.read(schedule_file_rising)

            # Setting schedule
            opts = [
                "--patch",
                "SETTING_SCAN_40,HORIZONTAL,1.00,210.00,330.00,40.00,60",
                "--out",
                schedule_file_setting,
            ]
            run_scheduler(opts=common_opts + opts)
            schedule_setting = GroundSchedule()
            schedule_setting.read(schedule_file_setting)
        else:
            schedule_rising = None
            schedule_setting = None

        if self.comm is not None:
            schedule_rising = self.comm.bcast(schedule_rising, root=0)
            schedule_setting = self.comm.bcast(schedule_setting, root=0)

        data_rising = Data(toastcomm)
        data_setting = Data(toastcomm)
        for data, schedule in [
            (data_rising, schedule_rising),
            (data_setting, schedule_setting),
        ]:
            sim_ground = ops.SimGround(
                name="sim_ground",
                telescope=tele,
                schedule=schedule,
                weather="atacama",
            )
            sim_ground.apply(data)

        for name, data in [("rising", data_rising), ("setting", data_setting)]:
            # Simple detector pointing
            detpointing_azel = ops.PointingDetectorSimple(
                boresight=defaults.boresight_azel, quats="quats_azel"
            )
            detpointing_radec = ops.PointingDetectorSimple(
                boresight=defaults.boresight_radec, quats="quats_radec"
            )
            stokes_weights = ops.StokesWeights(detector_pointing=detpointing_radec)

            # Create a noise model from focalplane detector properties
            default_model = ops.DefaultNoiseModel()
            default_model.apply(data)

            # Flag
            flag_sso = ops.FlagSSO(
                detector_pointing=detpointing_azel,
                det_flags="flags",
                sso_names=["Sun", "Moon"],
                sso_radii=[30 * u.deg, 5 * u.deg],
            )
            flag_sso.apply(data)

            # Make a hit map
            pixels = ops.PixelsHealpix(detector_pointing=detpointing_radec)
            binner = ops.BinMap(pixel_pointing=pixels, stokes_weights=stokes_weights)
            mapmaker = ops.MapMaker(
                name=f"mapmaker_{name}",
                binning=binner,
                output_dir=self.outdir,
                write_hits=True,
                write_rcond=False,
                write_binmap=False,
                write_map=False,
                write_cov=False,
            )
            mapmaker.apply(data)

        if self.comm is None or self.comm.rank == 0:
            # Compare the resulting hit maps
            hits_rising = hp.read_map(
                os.path.join(self.outdir, "mapmaker_rising_hits.fits")
            )
            hits_setting = hp.read_map(
                os.path.join(self.outdir, "mapmaker_setting_hits.fits")
            )
            assert np.abs(np.sum(hits_rising) / np.sum(hits_setting) - 1) < 1e-2
            # Make sure the observed sky matches expectation
            fsky = np.sum(hits_rising != 0) / hits_rising.size
            assert np.abs(fsky - 0.55) < 0.1

        close_data(data_rising)
        close_data(data_setting)
