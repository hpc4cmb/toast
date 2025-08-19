# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for creating fake ground instruments."""

import os
import re
import shutil
import tempfile

import numpy as np
from astropy import units as u
from astropy.table import vstack as table_vstack

from ... import ops
from ...data import Data
from ...instrument import Focalplane, GroundSite, Telescope
from ...instrument_sim import fake_hexagon_focalplane
from ...observation import default_values as defaults
from ...schedule import GroundSchedule
from ...schedule_sim_ground import run_scheduler
from .utils import create_comm


def create_ground_telescope(
    group_size,
    sample_rate=10.0 * u.Hz,
    pixel_per_process=1,
    fknee=None,
    freqs=None,
    width=5.0 * u.degree,
):
    """Create a fake ground telescope with at least one detector per process."""
    npix = 1
    ring = 1
    while 2 * npix <= group_size * pixel_per_process:
        npix += 6 * ring
        ring += 1
    fwhm = width / (ring + 2)
    if fknee is None:
        fknee = sample_rate / 2000.0
    if freqs is None:
        fp = fake_hexagon_focalplane(
            n_pix=npix,
            sample_rate=sample_rate,
            psd_fmin=1.0e-5 * u.Hz,
            psd_net=0.05 * u.K * np.sqrt(1 * u.second),
            psd_fknee=fknee,
            fwhm=fwhm,
            width=width,
        )
    else:
        fp_detdata = list()
        fov = None
        for freq in freqs:
            fp_freq = fake_hexagon_focalplane(
                n_pix=npix,
                sample_rate=sample_rate,
                psd_fmin=1.0e-5 * u.Hz,
                psd_net=0.05 * u.K * np.sqrt(1 * u.second),
                psd_fknee=fknee,
                fwhm=fwhm,
                bandcenter=freq,
                width=width,
            )
            if fov is None:
                fov = fp_freq.field_of_view
            fp_detdata.append(fp_freq.detector_data)

        fp_detdata = table_vstack(fp_detdata)
        fp = Focalplane(
            detector_data=fp_detdata,
            sample_rate=sample_rate,
            field_of_view=fov,
        )

    site = GroundSite("atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)
    return Telescope("telescope", focalplane=fp, site=site)


def create_ground_data(
    mpicomm,
    sample_rate=60.0 * u.Hz,
    hwp_rpm=59.0,
    fp_width=5.0 * u.degree,
    temp_dir=None,
    el_nod=False,
    el_nods=[-1 * u.degree, 1 * u.degree],
    pixel_per_process=1,
    fknee=None,
    freqs=None,
    split=False,
    turnarounds_invalid=False,
    single_group=False,
    flagged_pixels=True,
    schedule_hours=2,
):
    """Create a data object with a simple ground sim.

    Use the specified MPI communicator to attempt to create 2 process groups.  Create
    a fake telescope and run the ground sim to make some observations for each
    group.  This is useful for testing many operators that need some pre-existing
    observations with boresight pointing.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        sample_rate (Quantity): the sample rate.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = create_comm(mpicomm, single_group=single_group)
    data = Data(toastcomm)

    if flagged_pixels:
        # We are going to flag half the pixels
        pixel_per_process *= 2

    tele = create_ground_telescope(
        toastcomm.group_size,
        sample_rate=sample_rate,
        pixel_per_process=pixel_per_process,
        fknee=fknee,
        freqs=freqs,
        width=fp_width,
    )

    # Create a schedule.

    # FIXME: change this once the ground scheduler supports in-memory creation of the
    # schedule.

    schedule = None

    if mpicomm is None or mpicomm.rank == 0:
        tdir = temp_dir
        if tdir is None:
            tdir = tempfile.mkdtemp()

        sch_hours = f"{int(schedule_hours):02d}"
        sch_file = os.path.join(tdir, "ground_schedule.txt")
        run_scheduler(
            opts=[
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
                "--patch",
                "small_patch,1,40,-40,44,-44",
                "--start",
                "2020-01-01 00:00:00",
                "--stop",
                f"2020-01-01 {sch_hours}:00:00",
                "--out",
                sch_file,
            ]
        )
        schedule = GroundSchedule()
        schedule.read(sch_file)
        if temp_dir is None:
            shutil.rmtree(tdir)
    if mpicomm is not None:
        schedule = mpicomm.bcast(schedule, root=0)

    if freqs is None or not split:
        split_key = None
    else:
        split_key = "bandcenter"

    sim_ground = ops.SimGround(
        name="sim_ground",
        telescope=tele,
        session_split_key=split_key,
        schedule=schedule,
        hwp_angle=defaults.hwp_angle,
        hwp_rpm=hwp_rpm,
        weather="atacama",
        median_weather=True,
        detset_key="pixel",
        elnod_start=el_nod,
        elnods=el_nods,
        scan_accel_az=3 * u.degree / u.second**2,
    )
    if turnarounds_invalid:
        sim_ground.turnaround_mask = 1 + 2
    else:
        sim_ground.turnaround_mask = 2
    sim_ground.apply(data)

    if flagged_pixels:
        det_pat = re.compile(r"D(.*)[AB]-.*")
        for ob in data.obs:
            det_flags = dict()
            for det in ob.local_detectors:
                det_mat = det_pat.match(det)
                idet = int(det_mat.group(1))
                if idet % 2 != 0:
                    det_flags[det] = defaults.det_mask_invalid
            ob.update_local_detector_flags(det_flags)

    return data


def create_overdistributed_data(
    mpicomm,
    sample_rate=60.0 * u.Hz,
    hwp_rpm=59.0,
    fp_width=5.0 * u.degree,
    temp_dir=None,
    el_nod=False,
    el_nods=[-1 * u.degree, 1 * u.degree],
    fknee=None,
    freqs=None,
    turnarounds_invalid=False,
    single_group=False,
    schedule_hours=2,
):
    """Create a data object with more detectors than processes.

    Use the specified MPI communicator to attempt to create 2 process groups.  Create
    a fake telescope and run the ground sim to make some observations for each
    group.  The number of detectors in the telescope is set to half the group size.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        sample_rate (Quantity): the sample rate.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = create_comm(mpicomm, single_group=single_group)
    data = Data(toastcomm)

    # Create telescope with one pixel per process.
    tele = create_ground_telescope(
        toastcomm.group_size,
        sample_rate=sample_rate,
        pixel_per_process=1,
        fknee=fknee,
        freqs=freqs,
        width=fp_width,
    )

    # Modify the focalplane "pixel" key so that each pixel has 4
    # detectors.  When we use the detset key below, there will be
    # half as many detsets as processes.
    fp_table = tele.focalplane.detector_data
    old_pixels = np.array(fp_table["pixel"])
    for idet in range(len(old_pixels)):
        new_pix = old_pixels[idet // 4]
        fp_table["pixel"][idet] = new_pix

    # Create a schedule.

    schedule = None

    if mpicomm is None or mpicomm.rank == 0:
        tdir = temp_dir
        if tdir is None:
            tdir = tempfile.mkdtemp()

        sch_hours = f"{int(schedule_hours):02d}"
        sch_file = os.path.join(tdir, "ground_schedule.txt")
        run_scheduler(
            opts=[
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
                "--patch",
                "small_patch,1,40,-40,44,-44",
                "--start",
                "2020-01-01 00:00:00",
                "--stop",
                f"2020-01-01 {sch_hours}:00:00",
                "--out",
                sch_file,
            ]
        )
        schedule = GroundSchedule()
        schedule.read(sch_file)
        if temp_dir is None:
            shutil.rmtree(tdir)
    if mpicomm is not None:
        schedule = mpicomm.bcast(schedule, root=0)

    sim_ground = ops.SimGround(
        name="sim_ground",
        telescope=tele,
        schedule=schedule,
        hwp_angle=defaults.hwp_angle,
        hwp_rpm=hwp_rpm,
        weather="atacama",
        median_weather=True,
        detset_key="pixel",
        elnod_start=el_nod,
        elnods=el_nods,
        scan_accel_az=3 * u.degree / u.second**2,
    )
    if turnarounds_invalid:
        sim_ground.turnaround_mask = 1 + 2
    else:
        sim_ground.turnaround_mask = 2
    sim_ground.apply(data)

    return data
