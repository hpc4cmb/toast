# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from datetime import datetime

import numpy as np

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


def create_space_telescope(group_size, sample_rate=10.0 * u.Hz):
    """Create a fake satellite telescope with at least one detector per process."""
    npix = 1
    ring = 1
    while 2 * npix < group_size:
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
