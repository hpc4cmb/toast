# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from datetime import datetime

import numpy as np

import healpy as hp

from astropy import units as u

from ..mpi import Comm

from ..data import Data

from .. import qarray as qa

from ..instrument import Focalplane, Telescope, GroundSite, SpaceSite

from ..instrument_sim import fake_hexagon_focalplane

from ..schedule_sim_satellite import create_satellite_schedule

from ..observation import DetectorData, Observation

from ..pixels import PixelData

from .. import ops as ops

from astropy.table import QTable, Column

ZAXIS = np.array([0.0, 0.0, 1.0])


# These are helper routines for common operations used in the unit tests.


def create_outdir(mpicomm, subdir=None):
    """Create the top level output directory and per-test subdir.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        subdir (str): the sub directory for this test.

    Returns:
        str: full path to the test subdir if specified, else the top dir.

    """
    pwd = os.path.abspath(".")
    testdir = os.path.join(pwd, "toast_test_output")
    retdir = testdir
    if subdir is not None:
        retdir = os.path.join(testdir, subdir)
    if (mpicomm is None) or (mpicomm.rank == 0):
        if not os.path.isdir(testdir):
            os.mkdir(testdir)
        if not os.path.isdir(retdir):
            os.mkdir(retdir)
    if mpicomm is not None:
        mpicomm.barrier()
    return retdir


def create_comm(mpicomm):
    """Create a toast communicator.

    Use the specified MPI communicator to attempt to create 2 process groups.
    If less than 2 processes are used, create a single process group.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).

    Returns:
        toast.Comm: the 2-level toast communicator.

    """
    toastcomm = None
    if mpicomm is None:
        toastcomm = Comm(world=mpicomm)
    else:
        worldsize = mpicomm.size
        groupsize = 1
        if worldsize >= 2:
            groupsize = worldsize // 2
        toastcomm = Comm(world=mpicomm, groupsize=groupsize)
    return toastcomm


def create_space_telescope(group_size, sample_rate=10.0 * u.Hz, pixel_per_process=1):
    """Create a fake satellite telescope with at least one detector per process."""
    npix = 1
    ring = 1
    while 2 * npix < group_size * pixel_per_process:
        npix += 6 * ring
        ring += 1
    fp = fake_hexagon_focalplane(
        n_pix=npix,
        sample_rate=sample_rate,
        psd_fmin=1.0e-5 * u.Hz,
        psd_net=0.05 * u.K * np.sqrt(1 * u.second),
        psd_fknee=(sample_rate / 2000.0),
    )

    site = SpaceSite("L2")
    return Telescope("test", focalplane=fp, site=site)


# def create_ground_telescope(group_size, sample_rate=10.0 * u.Hz):
#     """Create a fake ground telescope with at least one detector per process."""
#     npix = 1
#     ring = 1
#     while 2 * npix < group_size:
#         npix += 6 * ring
#         ring += 1
#     fp = fake_hexagon_focalplane(
#         n_pix=npix,
#         sample_rate=sample_rate,
#         f_min=1.0e-5 * u.Hz,
#         # net=1.0,
#         net=0.5,
#         f_knee=(sample_rate / 2000.0),
#     )
#     return Telescope("test", focalplane=fp)


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
        ob = Observation(
            tele, n_samples=samples, name=oname, uid=oid, comm=toastcomm.comm_group
        )
        data.obs.append(ob)
    return data


def create_satellite_data(
    mpicomm, obs_per_group=1, sample_rate=10.0 * u.Hz, obs_time=10.0 * u.minute
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

    tele = create_space_telescope(toastcomm.group_size, sample_rate=sample_rate)

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
        hwp_rpm=10.0,
        spin_angle=5.0 * u.degree,
        prec_angle=10.0 * u.degree,
    )
    sim_sat.apply(data)

    return data


def create_satellite_data_big(
    mpicomm, obs_per_group=1, sample_rate=10.0 * u.Hz, obs_time=10.0 * u.minute, pixel_per_process=8
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
        group_size=toastcomm.group_size, pixel_per_process=pixel_per_process, sample_rate=sample_rate
    )
    det_props = tele.focalplane.detector_data
    fov = tele.focalplane.field_of_view
    sample_rate = tele.focalplane.sample_rate
    # (Add columns to det_props, which is an astropy QTable)
    # divide the detector into two groups
    det_props.add_column(
        Column(
            name="wafer", data=[f"W0{detindx%2}" for detindx, x in enumerate(det_props)]
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
        hwp_rpm=10.0,
        spin_angle=5.0 * u.degree,
        prec_angle=10.0 * u.degree,
    )
    sim_sat.apply(data)

    return data


def create_healpix_ring_satellite(mpicomm, obs_per_group=1, nside=64):
    """Create a toast data object with one boresight sample per healpix pixel.

    Use the specified MPI communicator to attempt to create 2 process groups,
    each with some empty observations.  Use a space telescope for each observation.
    Create fake boresight pointing that cycles through every healpix RING ordered
    pixel one time.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
        nside (int): The NSIDE value to use.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    nsamp = 12 * nside ** 2
    rate = 10.0

    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)
    for obs in range(obs_per_group):
        oname = "test-{}-{}".format(toastcomm.group, obs)
        oid = obs_per_group * toastcomm.group + obs
        tele = create_space_telescope(toastcomm.group_size)
        # FIXME: for full testing we should set detranks as approximately the sqrt
        # of the grid size so that we test the row / col communicators.
        ob = Observation(
            tele, n_samples=nsamp, name=oname, uid=oid, comm=toastcomm.comm_group
        )
        # Create shared objects for timestamps, common flags, boresight, position,
        # and velocity.
        ob.shared.create(
            "times",
            shape=(ob.n_local_samples,),
            dtype=np.float64,
            comm=ob.comm_col,
        )
        ob.shared.create(
            "flags",
            shape=(ob.n_local_samples,),
            dtype=np.uint8,
            comm=ob.comm_col,
        )
        ob.shared.create(
            "position",
            shape=(ob.n_local_samples, 3),
            dtype=np.float64,
            comm=ob.comm_col,
        )
        ob.shared.create(
            "velocity",
            shape=(ob.n_local_samples, 3),
            dtype=np.float64,
            comm=ob.comm_col,
        )
        ob.shared.create(
            "boresight_radec",
            shape=(ob.n_local_samples, 4),
            dtype=np.float64,
            comm=ob.comm_col,
        )
        # Rank zero of each grid column creates the data
        stamps = None
        position = None
        velocity = None
        boresight = None
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

            x, y, z = hp.pix2vec(nside, pix, nest=False)

            # z axis is obviously normalized
            zaxis = np.array([0, 0, 1], dtype=np.float64)
            ztiled = np.tile(zaxis, x.shape[0]).reshape((-1, 3))

            # ... so dir is already normalized
            dir = np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

            # get the rotation axis
            v = np.cross(ztiled, dir)
            v = v / np.sqrt(np.sum(v * v, axis=1)).reshape((-1, 1))

            # this is the vector-wise dot product
            zdot = np.sum(ztiled * dir, axis=1).reshape((-1, 1))
            ang = 0.5 * np.arccos(zdot)

            # angle element
            s = np.cos(ang)

            # axis
            v *= np.sin(ang)

            # build the normalized quaternion
            boresight = qa.norm(np.concatenate((v, s), axis=1))

        ob.shared["times"].set(stamps, offset=(0,), fromrank=0)
        ob.shared["position"].set(position, offset=(0, 0), fromrank=0)
        ob.shared["velocity"].set(velocity, offset=(0, 0), fromrank=0)
        ob.shared["boresight_radec"].set(boresight, offset=(0, 0), fromrank=0)

        data.obs.append(ob)
    return data


def create_fake_sky(data, dist_key, map_key):
    np.random.seed(987654321)
    dist = data[dist_key]
    pix_data = PixelData(dist, np.float64, n_value=3)
    # Just replicate the fake data across all local submaps
    off = 0
    for submap in range(dist.n_submap):
        I_data = 0.1 * np.random.normal(size=dist.n_pix_submap)
        Q_data = 0.01 * np.random.normal(size=dist.n_pix_submap)
        U_data = 0.01 * np.random.normal(size=dist.n_pix_submap)
        if submap in dist.local_submaps:
            pix_data.data[off, :, 0] = I_data
            pix_data.data[off, :, 1] = Q_data
            pix_data.data[off, :, 2] = U_data
            off += 1
    data[map_key] = pix_data


def uniform_chunks(samples, nchunk=100):
    """Divide some number of samples into chunks.

    This is often needed when constructing a TOD class, and usually we want
    the number of chunks to be larger than any number of processes we might
    be using for the unit tests.

    Args:
        samples (int): The number of samples.
        nchunk (int): The number of chunks to create.

    Returns:
        array: This list of chunk sizes.

    """
    chunksize = samples // nchunk
    chunks = np.ones(nchunk, dtype=np.int64)
    chunks *= chunksize
    remain = samples - (nchunk * chunksize)
    for r in range(remain):
        chunks[r] += 1
    return chunks


def create_fake_sky_alm(lmax=128, fwhm=10 * u.degree, pol=True, pointsources=False):
    if pointsources:
        nside = 512
        while nside < lmax:
            nside *= 2
        npix = 12 * nside ** 2
        m = np.zeros(npix)
        for lon in np.linspace(-180, 180, 6):
            for lat in np.linspace(-80, 80, 6):
                m[hp.ang2pix(nside, lon, lat, lonlat=True)] = 1
        if pol:
            m = np.vstack([m, m, m])
        m = hp.smoothing(m, fwhm=fwhm.to_value(u.radian))
        a_lm = hp.map2alm(m, lmax=lmax)
    else:
        # Power spectrum
        if pol:
            cl = np.ones(4 * (lmax + 1)).reshape([4, -1])
        else:
            cl = np.ones(lmax + 1)
        # Draw a_lm
        nside = 2
        while 4 * nside < lmax:
            nside *= 2
        _, a_lm = hp.synfast(
            cl,
            nside,
            alm=True,
            lmax=lmax,
            fwhm=fwhm.to_value(u.radian),
            verbose=False,
        )

    return a_lm


def create_fake_beam_alm(
    lmax=128,
    mmax=10,
    fwhm_x=10 * u.degree,
    fwhm_y=10 * u.degree,
    pol=True,
    separate_IQU=False,
):

    # pick an nside >= lmax to be sure that the a_lm will be fairly accurate
    nside = 2
    while nside < lmax:

        nside *= 2
    npix = 12 * nside ** 2
    pix = np.arange(npix)
    vec = hp.pix2vec(nside, pix, nest=False)
    theta, phi = hp.vec2dir(vec)
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    sigma_x = fwhm_x.to_value(u.radian) / np.sqrt(8 * np.log(2))
    sigma_y = fwhm_y.to_value(u.radian) / np.sqrt(8 * np.log(2))
    beam_map = np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2))
    empty = np.zeros_like(beam_map)
    if pol and separate_IQU:
        beam_map_I = np.vstack([beam_map, empty, empty])
        beam_map_Q = np.vstack([empty, beam_map, empty])
        beam_map_U = np.vstack([empty, empty, beam_map])

        try:
            a_lm = [
                hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax, verbose=False),
                hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax, verbose=False),
                hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax, verbose=False),
            ]
        except TypeError:
            # older healpy which does not have verbose keyword
            a_lm = [
                hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax),
                hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax),
                hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax),
            ]
    else:
        if pol:
            beam_map = np.vstack([beam_map, beam_map, empty])
        try:
            a_lm = hp.map2alm(beam_map, lmax=lmax, mmax=mmax, verbose=False)
        except TypeError:
            a_lm = hp.map2alm(beam_map, lmax=lmax, mmax=mmax)

    return a_lm
