# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

from datetime import datetime

import numpy as np
import numpy.testing as nt

from astropy import units as u

import healpy as hp

from ..mpi import Comm, MPI

from ..data import Data

from ..instrument import Focalplane, Telescope, GroundSite

from ..instrument_sim import fake_hexagon_focalplane

from ..schedule_sim_ground import run_scheduler

from ..schedule import GroundSchedule

from ..pixels_io import write_healpix_fits

from ..vis import set_matplotlib_backend

from .. import qarray as qa

from .. import ops as ops

from ._helpers import create_outdir, create_comm


class SimGroundTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.toastcomm = create_comm(self.comm)

        npix = 1
        ring = 1
        while 2 * npix < self.toastcomm.group_size:
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
                ]
            )
            schedule = GroundSchedule()
            schedule.read(sch_file)
        if self.comm is not None:
            schedule = self.comm.bcast(schedule, root=0)

        data = Data(self.toastcomm)

        sim_ground = ops.SimGround(
            name="sim_ground",
            telescope=tele,
            schedule=schedule,
            hwp_rpm=1.0,
        )
        sim_ground.apply(data)

        # Expand pointing and make a hit map.

        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nest=True,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.nside_submap = 8
        pointing.nside = 512
        pointing.apply(data)

        build_hits = ops.BuildHitMap(pixel_dist="pixel_dist", pixels=pointing.pixels)
        build_hits.apply(data)

        # Plot the hits

        hit_path = os.path.join(self.outdir, "hits.fits")
        write_healpix_fits(data[build_hits.hits], hit_path, nest=pointing.nest)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            hits = hp.read_map(hit_path, field=None, nest=pointing.nest)
            outfile = os.path.join(self.outdir, "hits.png")
            hp.mollview(hits, xsize=1600, max=50, nest=pointing.nest)
            plt.savefig(outfile)
            plt.close()
