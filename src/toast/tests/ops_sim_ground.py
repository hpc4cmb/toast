# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
from datetime import datetime

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..data import Data
from ..instrument import Focalplane, GroundSite, Telescope
from ..instrument_sim import fake_hexagon_focalplane
from ..io.observation_hdf_load import load_instrument_file
from ..mpi import MPI, Comm
from ..observation import default_values as defaults
from ..schedule import GroundSchedule
from ..schedule_sim_ground import run_scheduler
from ..vis import plot_projected_quats, set_matplotlib_backend
from ..scripts.toast_fake_telescope import main as fake_tele_main
from .helpers import close_data, create_comm, create_outdir
from .mpi import MPITestCase


class SimGroundTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

        self.toastcomm = create_comm(self.comm)

        npix = 1
        ring = 1
        while npix <= self.toastcomm.group_size:
            npix += 6 * ring
            ring += 1
        self.npix = npix
        self.fp = fake_hexagon_focalplane(n_pix=npix)

    def test_exec(self):
        # Slow sampling
        fp = fake_hexagon_focalplane(
            n_pix=self.npix,
            sample_rate=10.0 * u.Hz,
        )

        site = GroundSite("Atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)

        tele = Telescope("telescope", focalplane=fp, site=site)

        sch_file = os.path.join(self.outdir, "exec_schedule.txt")
        schedule = None

        if self.comm is None or self.comm.rank == 0:
            run_scheduler(
                opts=[
                    "--site-name",
                    site.name,
                    "--telescope",
                    tele.name,
                    "--site-lon",
                    "{}".format(site.earthloc.lon.to_value(u.degree)),
                    "--site-lat",
                    "{}".format(site.earthloc.lat.to_value(u.degree)),
                    "--site-alt",
                    "{}".format(site.earthloc.height.to_value(u.meter)),
                    "--patch",
                    "small_patch,1,40,-40,44,-44",
                    "--start",
                    "2020-01-01 00:00:00",
                    "--stop",
                    "2020-01-01 12:00:00",
                    "--out",
                    sch_file,
                    "--field-separator",
                    "|",
                ]
            )
            schedule = GroundSchedule()
            schedule.read(sch_file)
        if self.comm is not None:
            schedule = self.comm.bcast(schedule, root=0)

        # Sort the schedule to exercise that method
        schedule.sort_by_RA()

        data = Data(self.toastcomm)

        sim_ground = ops.SimGround(
            name="sim_ground",
            telescope=tele,
            schedule=schedule,
            hwp_angle=defaults.hwp_angle,
            hwp_rpm=1.0,
            max_pwv=5 * u.mm,
        )
        sim_ground.apply(data)

        # Plot some pointing
        plotdetpointing = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel,
            quats="pquats",
        )
        plotdetpointing.apply(data)
        if data.comm.world_rank == 0:
            n_debug = 100
            bquat = np.array(data.obs[0].shared[defaults.boresight_azel][10:n_debug, :])
            dquat = data.obs[0].detdata["pquats"][:, 10:n_debug, :]
            invalid = np.array(data.obs[0].shared[defaults.shared_flags][10:n_debug])
            invalid &= defaults.shared_mask_invalid
            valid = np.logical_not(invalid)
            outfile = os.path.join(self.outdir, "pointing.pdf")
            plot_projected_quats(
                outfile, qbore=bquat, qdet=dquat, valid=valid, scale=1.0
            )

        # Pointing
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nest=True,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )

        # Test modifying the noise model

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model="noise_model",
            out_model="el_weighted",
            detector_pointing=detpointing,
            noise_a=0.5,
            noise_c=0.5,
        )
        el_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=el_model.out_model, det_data=defaults.det_data
        )
        sim_noise.apply(data)

        # Expand pointing and make a hit map.

        pixels.nside_submap = 8
        pixels.nside = 512
        pixels.apply(data)

        build_hits = ops.BuildHitMap(pixel_dist="pixel_dist", pixels=pixels.pixels)
        build_hits.apply(data)

        # Plot the hits

        hit_path = os.path.join(self.outdir, "hits.fits")
        data[build_hits.hits].write(hit_path)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            hits = hp.read_map(hit_path, field=None, nest=pixels.nest)
            outfile = os.path.join(self.outdir, "hits.png")
            hp.mollview(hits, xsize=1600, max=50, nest=pixels.nest)
            plt.savefig(outfile)
            plt.close()

        close_data(data)

    def test_qpoint(self):
        try:
            import qpoint
        except Exception:
            print("qpoint not importable, skipping test")
            return

        # Slow sampling
        fp = fake_hexagon_focalplane(
            n_pix=self.npix,
            sample_rate=10.0 * u.Hz,
        )

        site = GroundSite("Atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)

        tele = Telescope("telescope", focalplane=fp, site=site)

        sch_file = os.path.join(self.outdir, "exec_schedule.txt")
        schedule = None

        if self.comm is None or self.comm.rank == 0:
            run_scheduler(
                opts=[
                    "--site-name",
                    site.name,
                    "--telescope",
                    tele.name,
                    "--site-lon",
                    "{}".format(site.earthloc.lon.to_value(u.degree)),
                    "--site-lat",
                    "{}".format(site.earthloc.lat.to_value(u.degree)),
                    "--site-alt",
                    "{}".format(site.earthloc.height.to_value(u.meter)),
                    "--patch",
                    "small_patch,1,40,-40,44,-44",
                    "--start",
                    "2020-01-01 00:00:00",
                    "--stop",
                    "2020-01-01 12:00:00",
                    "--out",
                    sch_file,
                    "--field-separator",
                    "|",
                ]
            )
            schedule = GroundSchedule()
            schedule.read(sch_file)
        if self.comm is not None:
            schedule = self.comm.bcast(schedule, root=0)

        # Sort the schedule to exercise that method
        schedule.sort_by_RA()

        data = Data(self.toastcomm)

        sim_ground = ops.SimGround(
            name="sim_ground",
            telescope=tele,
            schedule=schedule,
            hwp_angle=defaults.hwp_angle,
            hwp_rpm=1.0,
            max_pwv=5 * u.mm,
            weather="atacama",
            median_weather=True,
            use_ephem=False,
            use_qpoint=True,
        )
        sim_ground.apply(data)

    def test_phase(self):
        # Slow sampling
        fp = fake_hexagon_focalplane(
            n_pix=self.npix,
            sample_rate=10.0 * u.Hz,
        )

        site = GroundSite("Atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)

        tele = Telescope("telescope", focalplane=fp, site=site)

        sch_file = os.path.join(self.outdir, "exec_schedule.txt")
        schedule = None

        if self.comm is None or self.comm.rank == 0:
            run_scheduler(
                opts=[
                    "--site-name",
                    site.name,
                    "--telescope",
                    tele.name,
                    "--site-lon",
                    "{}".format(site.earthloc.lon.to_value(u.degree)),
                    "--site-lat",
                    "{}".format(site.earthloc.lat.to_value(u.degree)),
                    "--site-alt",
                    "{}".format(site.earthloc.height.to_value(u.meter)),
                    "--patch",
                    "small_patch,1,40,-40,44,-44",
                    "--start",
                    "2020-01-01 00:00:00",
                    "--stop",
                    "2020-01-01 12:00:00",
                    "--out",
                    sch_file,
                    "--verbose",
                ]
            )
            schedule = GroundSchedule()
            schedule.read(sch_file)
        if self.comm is not None:
            schedule = self.comm.bcast(schedule, root=0)

        data1 = Data(self.toastcomm)
        data2 = Data(self.toastcomm)

        sim_ground = ops.SimGround(
            name="sim_ground",
            telescope=tele,
            schedule=schedule,
            hwp_angle=defaults.hwp_angle,
            hwp_rpm=1.0,
            max_pwv=5 * u.mm,
        )
        sim_ground.apply(data1)
        sim_ground.randomize_phase = True
        sim_ground.apply(data2)

        # Verify the two boresights are in different phase

        az1 = data1.obs[0].shared["azimuth"][:]
        az2 = data2.obs[0].shared["azimuth"][:]

        assert np.std(az1 - az2) > 1e-10

        # Verify that the intervals still identify left-right scans correctly

        good1 = np.zeros(az1.size, dtype=bool)
        good2 = np.zeros(az2.size, dtype=bool)

        for ival in data1.obs[0].intervals[defaults.scan_leftright_interval]:
            good1[ival.first : ival.last] = True
        for ival in data2.obs[0].intervals[defaults.scan_leftright_interval]:
            good2[ival.first : ival.last] = True

        step1 = np.median(np.diff(az1[good1]))
        step2 = np.median(np.diff(az2[good2]))

        assert np.abs((step1 - step2) / step1) < 1e-3

        del data2
        close_data(data1)

    def test_telescope_file(self):
        # Verify that we can run with a telescope file dumped to disk
        tele_file = sch_file = os.path.join(self.outdir, "ground_telescope.h5")
        tele = None
        if self.toastcomm.world_rank == 0:
            fake_tele_main(
                opts=[
                    "--out",
                    tele_file,
                    "--ground_site_loc",
                    "chajnantor",
                    "--telescope_name",
                    "CCAT",
                ]
            )
            tele, _ = load_instrument_file(tele_file)
        if self.toastcomm.comm_world is not None:
            tele = self.toastcomm.comm_world.bcast(tele, root=0)

        sch_file = os.path.join(self.outdir, "fake_telescope_schedule.txt")
        schedule = None

        if self.comm is None or self.comm.rank == 0:
            run_scheduler(
                opts=[
                    "--site-name",
                    "chajnantor",
                    "--telescope",
                    "CCAT",
                    "--site-lon",
                    f"{tele.site.earthloc.lon.to_value(u.degree)}",
                    "--site-lat",
                    f"{tele.site.earthloc.lat.to_value(u.degree)}",
                    "--site-alt",
                    f"{tele.site.earthloc.height.to_value(u.meter)}",
                    "--patch",
                    "small_patch,1,40,-40,44,-44",
                    "--start",
                    "2020-01-01 00:00:00",
                    "--stop",
                    "2020-01-01 12:00:00",
                    "--out",
                    sch_file,
                    "--verbose",
                ]
            )
            schedule = GroundSchedule()
            schedule.read(sch_file)
        if self.comm is not None:
            schedule = self.comm.bcast(schedule, root=0)

        data = Data(self.toastcomm)

        sim_ground = ops.SimGround(
            name="sim_ground",
            telescope_file=tele_file,
            schedule=schedule,
            hwp_angle=defaults.hwp_angle,
            hwp_rpm=1.0,
            max_pwv=5 * u.mm,
        )
        sim_ground.apply(data)

        close_data(data)
        del data
