# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for creating fake satellite instruments."""

import re
from datetime import datetime

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.table import Column

from ... import ops
from ... import qarray as qa
from ...data import Data
from ...instrument import Focalplane, SpaceSite, Telescope
from ...instrument_sim import fake_boresight_focalplane, fake_hexagon_focalplane
from ...observation import Observation
from ...observation import default_values as defaults
from ...schedule_sim_satellite import create_satellite_schedule
from .utils import create_comm


def create_space_telescope(
    group_size, sample_rate=10.0 * u.Hz, pixel_per_process=1, width=5.0 * u.degree
):
    """Create a fake satellite telescope with at least one pixel per process."""
    npix = 1
    ring = 1
    while 2 * npix <= group_size * pixel_per_process:
        npix += 6 * ring
        ring += 1
    fp = fake_hexagon_focalplane(
        n_pix=npix,
        sample_rate=sample_rate,
        psd_fmin=1.0e-5 * u.Hz,
        psd_net=0.05 * u.K * np.sqrt(1 * u.second),
        psd_fknee=(sample_rate / 2000.0),
        width=width,
    )
    site = SpaceSite("L2")
    return Telescope("test", focalplane=fp, site=site)


def create_boresight_telescope(
    group_size, sample_rate=10.0 * u.Hz, pixel_per_process=1
):
    """Create a fake telescope with one boresight pixel per process."""
    fp = fake_boresight_focalplane(
        n_pix=group_size * pixel_per_process,
        sample_rate=sample_rate,
        fwhm=1.0 * u.degree,
        psd_fmin=1.0e-5 * u.Hz,
        psd_net=0.05 * u.K * np.sqrt(1 * u.second),
        psd_fknee=(sample_rate / 2000.0),
    )
    site = SpaceSite("L2")
    return Telescope("test", focalplane=fp, site=site)


def create_satellite_empty(mpicomm, obs_per_group=1, samples=10):
    """Create a toast communicator and (empty) distributed data object.

    Use the specified MPI communicator to attempt to create 2 process groups,
    each with some empty observations.  Use a space telescope for each observation.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
        samples (int): number of samples per observation.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)
    for obs in range(obs_per_group):
        oname = "test-{}-{}".format(toastcomm.group, obs)
        oid = obs_per_group * toastcomm.group + obs
        tele = create_space_telescope(toastcomm.group_size)
        # FIXME: for full testing we should set detranks as approximately the sqrt
        # of the grid size so that we test the row / col communicators.
        ob = Observation(toastcomm, tele, n_samples=samples, name=oname, uid=oid)
        data.obs.append(ob)
    return data


def create_satellite_data(
    mpicomm,
    obs_per_group=1,
    sample_rate=10.0 * u.Hz,
    obs_time=10.0 * u.minute,
    gap_time=0.0 * u.minute,
    pixel_per_process=1,
    hwp_rpm=9.0,
    width=5.0 * u.degree,
    single_group=False,
    flagged_pixels=True,
):
    """Create a data object with a simple satellite sim.

    Use the specified MPI communicator to attempt to create 2 process groups.  Create
    a fake telescope and run the satellite sim to make some observations for each
    group.  This is useful for testing many operators that need some pre-existing
    observations with boresight pointing.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
        sample_rate (Quantity): the sample rate.
        obs_time (Quantity): the time length of one observation.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = create_comm(mpicomm, single_group=single_group)
    data = Data(toastcomm)

    if flagged_pixels:
        # We are going to flag half the pixels
        pixel_per_process *= 2

    tele = create_space_telescope(
        toastcomm.group_size,
        sample_rate=sample_rate,
        pixel_per_process=pixel_per_process,
        width=width,
    )

    # Create a schedule

    sch = create_satellite_schedule(
        prefix="test_",
        mission_start=datetime(2023, 2, 23),
        observation_time=obs_time,
        gap_time=gap_time,
        num_observations=(toastcomm.ngroups * obs_per_group),
        prec_period=5 * u.minute,
        spin_period=0.5 * u.minute,
    )

    # Scan fast enough to cover some sky in a short amount of time.  Reduce the
    # angles to achieve a more compact hit map.
    if hwp_rpm == 0 or hwp_rpm is None:
        hwp_angle = None
    else:
        hwp_angle = defaults.hwp_angle
    sim_sat = ops.SimSatellite(
        name="sim_sat",
        telescope=tele,
        schedule=sch,
        hwp_angle=hwp_angle,
        hwp_rpm=hwp_rpm,
        spin_angle=3.0 * u.degree,
        prec_angle=7.0 * u.degree,
        detset_key="pixel",
    )
    sim_sat.apply(data)

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


def create_satellite_data_big(
    mpicomm,
    obs_per_group=1,
    sample_rate=10.0 * u.Hz,
    obs_time=10.0 * u.minute,
    pixel_per_process=8,
):
    """Create a data object with a simple satellite sim.

    Use the specified MPI communicator to attempt to create 2 process groups.  Create
    a fake telescope and run the satellite sim to make some observations for each
    group.  This is useful for testing many operators that need some pre-existing
    observations with boresight pointing.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
        samples (int): number of samples per observation.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)

    tele = create_space_telescope(
        group_size=toastcomm.group_size,
        pixel_per_process=pixel_per_process,
        sample_rate=sample_rate,
    )
    det_props = tele.focalplane.detector_data
    fov = tele.focalplane.field_of_view
    sample_rate = tele.focalplane.sample_rate
    # (Add columns to det_props, which is an astropy QTable)
    # divide the detector into two groups
    det_props.add_column(
        Column(
            name="wafer",
            data=[f"W0{detindx % 2}" for detindx, x in enumerate(det_props)],
        )
    )
    new_telescope = Telescope(
        "Big Satellite",
        focalplane=Focalplane(
            detector_data=det_props, field_of_view=fov, sample_rate=sample_rate
        ),
        site=tele.site,
    )
    # Create a schedule

    sch = create_satellite_schedule(
        prefix="test_",
        mission_start=datetime(2023, 2, 23),
        observation_time=obs_time,
        gap_time=0 * u.minute,
        num_observations=(toastcomm.ngroups * obs_per_group),
        prec_period=10 * u.minute,
        spin_period=1 * u.minute,
    )

    # Scan fast enough to cover some sky in a short amount of time.  Reduce the
    # angles to achieve a more compact hit map.
    sim_sat = ops.SimSatellite(
        name="sim_sat",
        telescope=tele,
        schedule=sch,
        hwp_angle=defaults.hwp_angle,
        hwp_rpm=10.0,
        spin_angle=5.0 * u.degree,
        prec_angle=10.0 * u.degree,
        detset_key="pixel",
    )
    sim_sat.apply(data)

    return data


def create_healpix_ring_satellite(
    mpicomm, obs_per_group=1, pix_per_process=4, nside=64
):
    """Create data with boresight samples centered on healpix pixels.

    Use the specified MPI communicator to attempt to create 2 process groups,
    each with some empty observations.  Use a space telescope for each observation.
    Create fake boresight pointing that cycles through every healpix RING ordered
    pixel one time.

    All detectors are placed at the boresight.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
        pix_per_process (int): number of boresight pixels per process.
        nside (int): The NSIDE value to use.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    nsamp = 12 * nside**2
    rate = 10.0

    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)
    for obs in range(obs_per_group):
        oname = "test-{}-{}".format(toastcomm.group, obs)
        oid = obs_per_group * toastcomm.group + obs
        tele = create_boresight_telescope(
            toastcomm.group_size,
            sample_rate=rate * u.Hz,
            pixel_per_process=pix_per_process,
        )

        # FIXME: for full testing we should set detranks as approximately the sqrt
        # of the grid size so that we test the row / col communicators.
        ob = Observation(toastcomm, tele, n_samples=nsamp, name=oname, uid=oid)
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
            defaults.position,
            shape=(ob.n_local_samples, 3),
            dtype=np.float64,
        )
        ob.shared.create_column(
            defaults.velocity,
            shape=(ob.n_local_samples, 3),
            dtype=np.float64,
        )
        ob.shared.create_column(
            defaults.boresight_radec,
            shape=(ob.n_local_samples, 4),
            dtype=np.float64,
        )
        ob.shared.create_column(
            defaults.hwp_angle,
            shape=(ob.n_local_samples,),
            dtype=np.float64,
        )
        ob.detdata.create(defaults.det_data, dtype=np.float64, units=u.K)
        ob.detdata[defaults.det_data][:] = 0
        ob.detdata.create(defaults.det_flags, dtype=np.uint8)
        ob.detdata[defaults.det_flags][:] = 0

        # Rank zero of each grid column creates the data
        stamps = None
        position = None
        velocity = None
        boresight = None
        flags = None
        hwp_angle = None
        if ob.comm_col_rank == 0:
            start_time = 0.0 + float(ob.local_index_offset) / rate
            stop_time = start_time + float(ob.n_local_samples - 1) / rate
            stamps = np.linspace(
                start_time,
                stop_time,
                num=ob.n_local_samples,
                endpoint=True,
                dtype=np.float64,
            )

            # Get the motion of the site for these times.
            position, velocity = tele.site.position_velocity(stamps)

            pix = np.arange(nsamp, dtype=np.int64)

            # Cartesian coordinates of each pixel center on the unit sphere
            x, y, z = hp.pix2vec(nside, pix, nest=False)

            # The focalplane orientation (X-axis) is defined to point to the
            # South.  To get this, we first rotate about the coordinate Z axis,
            # Then rotate about the Y axis.

            # The angle in the X/Y plane to the pixel direction
            phi = np.arctan2(y, x)

            # The angle to rotate about the Y axis from the pole down to the
            # pixel direction
            theta = np.arccos(z)

            # Focalplane orientation is to the south already
            psi = np.zeros_like(theta)

            boresight = qa.from_iso_angles(theta, phi, psi)

            # no flags
            flags = np.zeros(nsamp, dtype=np.uint8)

            # Set HWP angle to all zeros for later modification
            hwp_angle = np.zeros(nsamp, dtype=np.float64)

        ob.shared[defaults.times].set(stamps, offset=(0,), fromrank=0)
        ob.shared[defaults.position].set(position, offset=(0, 0), fromrank=0)
        ob.shared[defaults.velocity].set(velocity, offset=(0, 0), fromrank=0)
        ob.shared[defaults.boresight_radec].set(boresight, offset=(0, 0), fromrank=0)
        ob.shared[defaults.shared_flags].set(flags, offset=(0,), fromrank=0)
        ob.shared[defaults.hwp_angle].set(hwp_angle, offset=(0,), fromrank=0)

        data.obs.append(ob)
    return data
