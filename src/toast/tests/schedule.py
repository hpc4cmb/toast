# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..instrument import GroundSite, Telescope
from ..instrument_sim import fake_hexagon_focalplane
from ..schedule import GroundSchedule
from ..schedule_sim_ground import run_scheduler
from .helpers import create_comm, create_outdir
from .mpi import MPITestCase


class SimScheduleTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def compare_schedules(self, output, expected):
        for key in GroundSchedule.META_KEYS:
            out_val = getattr(output, key)
            expt_val = getattr(expected, key)
            if out_val != expt_val:
                print(f"Failed: {out_val} != {expt_val}", flush=True)
                return False
        out_scans = output.scans
        expt_scans = expected.scans
        if len(out_scans) != len(expt_scans):
            print(
                f"Failed: number of scans ({len(out_scans)}) != ({len(expt_scans)})",
                flush=True,
            )
            return False
        for oscn, escn in zip(out_scans, expt_scans):
            if oscn.name != escn.name:
                print(f"Failed: scan name ({oscn.name}) != ({escn.name})", flush=True)
                return False
            if oscn.start != escn.start:
                print(
                    f"Failed: scan start ({oscn.start}) != ({escn.start})", flush=True
                )
                return False
            if oscn.stop != escn.stop:
                print(f"Failed: scan stop ({oscn.stop}) != ({escn.stop})", flush=True)
                return False
            if oscn.boresight_angle != escn.boresight_angle:
                print(
                    f"Failed: boresight_angle ({oscn.boresight_angle}) != ({escn.boresight_angle})",
                    flush=True,
                )
                return False
            if oscn.az_min != escn.az_min:
                print(f"Failed: az min ({oscn.az_min}) != ({escn.az_min})", flush=True)
                return False
            if oscn.az_max != escn.az_max:
                print(f"Failed: az max ({oscn.az_max}) != ({escn.az_max})", flush=True)
                return False
            if oscn.el != escn.el:
                print(f"Failed: el ({oscn.el}) != ({escn.el})", flush=True)
                return False
            if oscn.scan_indx != escn.scan_indx:
                print(
                    f"Failed: scan index ({oscn.scan_indx}) != ({escn.scan_indx})",
                    flush=True,
                )
                return False
            if oscn.subscan_indx != escn.subscan_indx:
                print(
                    f"Failed: subscan index ({oscn.subscan_indx}) != ({escn.subscan_indx})",
                    flush=True,
                )
                return False
        return True

    def test_conversion(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Slow sampling
        fp = fake_hexagon_focalplane(
            n_pix=1,
            sample_rate=10.0 * u.Hz,
        )

        site = GroundSite("Atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)
        tele = Telescope("telescope", focalplane=fp, site=site)

        input_file = os.path.join(self.outdir, "schedule_input.txt")
        input_schedule = None

        if rank == 0:
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
                    input_file,
                    "--field-separator",
                    "|",
                ]
            )
        if self.comm is not None:
            self.comm.barrier()

        input_schedule = GroundSchedule()
        input_schedule.read(input_file, comm=self.comm)

        # Write to astropy ecsv format and compare
        new_file = os.path.join(self.outdir, "schedule_v5.ecsv")
        if rank == 0:
            input_schedule.write(new_file)

        new_schedule = GroundSchedule()
        new_schedule.read(new_file, comm=self.comm)

        if not self.compare_schedules(new_schedule, input_schedule):
            msg = f"ECSV schedule {new_file} does not match {input_file}"
            print(msg, flush=True)
            self.assertTrue(False)

        # Write the data to legacy format and reload to compare
        legacy_file = os.path.join(self.outdir, "schedule_v4.txt")
        if rank == 0:
            input_schedule.write(legacy_file)

        legacy_schedule = GroundSchedule()
        legacy_schedule.read(legacy_file, comm=self.comm)

        if not self.compare_schedules(legacy_schedule, input_schedule):
            msg = f"Legacy schedule {legacy_file} does not match {input_file}"
            print(msg, flush=True)
            self.assertTrue(False)
