# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import shutil
import tempfile
from datetime import datetime

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.table import Column, QTable

from .. import ops as ops
from .. import qarray as qa
from ..data import Data
from ..instrument import Focalplane, GroundSite, SpaceSite, Telescope
from ..instrument_sim import fake_hexagon_focalplane
from ..mpi import Comm
from ..observation import DetectorData, Observation
from ..observation import default_values as defaults
from ..pixels import PixelData
from ..schedule import GroundSchedule
from ..schedule_sim_ground import run_scheduler
from ..schedule_sim_satellite import create_satellite_schedule
from ..vis import set_matplotlib_backend

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
    )

    site = SpaceSite("L2")
    return Telescope("test", focalplane=fp, site=site)


def create_boresight_telescope(group_size, sample_rate=10.0 * u.Hz):
    """Create a fake telescope with one boresight detector per process."""
    nullquat = np.array([0, 0, 0, 1], dtype=np.float64)
    n_det = group_size
    det_names = [f"d{x:03d}" for x in range(n_det)]
    pol_ang = np.array([(2 * np.pi * x / n_det) for x in range(n_det)])

    det_table = QTable(
        [
            Column(name="name", data=det_names),
            Column(name="quat", data=[nullquat for x in range(n_det)]),
            Column(name="pol_leakage", length=n_det, unit=None),
            Column(name="psi_pol", data=pol_ang, unit=u.rad),
            Column(name="fwhm", length=n_det, unit=u.arcmin),
            Column(name="psd_fmin", length=n_det, unit=u.Hz),
            Column(name="psd_fknee", length=n_det, unit=u.Hz),
            Column(name="psd_alpha", length=n_det, unit=None),
            Column(name="psd_net", length=n_det, unit=(u.K * np.sqrt(1.0 * u.second))),
            Column(name="bandcenter", length=n_det, unit=u.GHz),
            Column(name="bandwidth", length=n_det, unit=u.GHz),
            Column(name="pixel", data=[0 for x in range(n_det)]),
        ]
    )

    fwhm = 5.0 * u.arcmin

    for idet in range(len(det_table)):
        det_table[idet]["pol_leakage"] = 0.0
        det_table[idet]["fwhm"] = fwhm
        det_table[idet]["bandcenter"] = 150.0 * u.GHz
        det_table[idet]["bandwidth"] = 20.0 * u.GHz
        det_table[idet]["psd_fmin"] = 1.0e-5 * u.Hz
        det_table[idet]["psd_fknee"] = 0.05 * u.Hz
        det_table[idet]["psd_alpha"] = 1.0
        det_table[idet]["psd_net"] = 100 * (u.K * np.sqrt(1.0 * u.second))

    fp = Focalplane(
        detector_data=det_table,
        sample_rate=sample_rate,
        field_of_view=1.1 * (2 * fwhm),
    )

    site = SpaceSite("L2")
    return Telescope("test", focalplane=fp, site=site)


def create_ground_telescope(
    group_size, sample_rate=10.0 * u.Hz, pixel_per_process=1, fknee=None
):
    """Create a fake ground telescope with at least one detector per process."""
    npix = 1
    ring = 1
    while 2 * npix <= group_size * pixel_per_process:
        npix += 6 * ring
        ring += 1
    if fknee is None:
        fknee = sample_rate / 2000.0
    fp = fake_hexagon_focalplane(
        n_pix=npix,
        sample_rate=sample_rate,
        psd_fmin=1.0e-5 * u.Hz,
        psd_net=0.05 * u.K * np.sqrt(1 * u.second),
        psd_fknee=fknee,
    )

    site = GroundSite("Atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)
    return Telescope("telescope", focalplane=fp, site=site)


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
    pixel_per_process=1,
    hwp_rpm=10.0,
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
    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)

    tele = create_space_telescope(
        toastcomm.group_size,
        sample_rate=sample_rate,
        pixel_per_process=pixel_per_process,
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
        spin_angle=5.0 * u.degree,
        prec_angle=10.0 * u.degree,
        detset_key="pixel",
    )
    sim_sat.apply(data)

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
        hwp_angle=defaults.hwp_angle,
        hwp_rpm=10.0,
        spin_angle=5.0 * u.degree,
        prec_angle=10.0 * u.degree,
        detset_key="pixel",
    )
    sim_sat.apply(data)

    return data


def create_healpix_ring_satellite(mpicomm, obs_per_group=1, nside=64):
    """Create data with boresight samples centered on healpix pixels.

    Use the specified MPI communicator to attempt to create 2 process groups,
    each with some empty observations.  Use a space telescope for each observation.
    Create fake boresight pointing that cycles through every healpix RING ordered
    pixel one time.

    All detectors are placed at the boresight.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
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
        tele = create_space_telescope(toastcomm.group_size)

        # Move all detectors to the boresight
        for row in tele.focalplane.detector_data:
            row["quat"] = np.array([0, 0, 0, 1], dtype=np.float64)

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
        ob.detdata.create(defaults.det_data, dtype=np.float64)
        ob.detdata.create(defaults.det_flags, dtype=np.uint8)
        # Rank zero of each grid column creates the data
        stamps = None
        position = None
        velocity = None
        boresight = None
        flags = None
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

            # no flags
            flags = np.zeros(nsamp, dtype=np.uint8)

        ob.shared[defaults.times].set(stamps, offset=(0,), fromrank=0)
        ob.shared[defaults.position].set(position, offset=(0, 0), fromrank=0)
        ob.shared[defaults.velocity].set(velocity, offset=(0, 0), fromrank=0)
        ob.shared[defaults.boresight_radec].set(boresight, offset=(0, 0), fromrank=0)
        ob.shared[defaults.shared_flags].set(flags, offset=(0,), fromrank=0)

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


def create_fake_mask(data, dist_key, mask_key):
    np.random.seed(987654321)
    dist = data[dist_key]
    pix_data = PixelData(dist, np.uint8, n_value=1)
    # Just replicate the fake data across all local submaps
    off = 0
    for submap in range(dist.n_submap):
        mask_data = np.random.normal(size=dist.n_pix_submap) > 0.5
        if submap in dist.local_submaps:
            pix_data.data[off, :, 0] = mask_data
            off += 1
    data[mask_key] = pix_data


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
        npix = 12 * nside**2
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
    separate_TP=False,
    detB_beam=False,
    normalize_beam=False,
):

    # pick an nside >= lmax to be sure that the a_lm will be fairly accurate
    nside = 2
    while nside < lmax:
        nside *= 2
    npix = 12 * nside**2
    pix = np.arange(npix)
    x, y, z = hp.pix2vec(nside, pix, nest=False)
    sigma_z = fwhm_x.to_value(u.radian) / np.sqrt(8 * np.log(2))
    sigma_y = fwhm_y.to_value(u.radian) / np.sqrt(8 * np.log(2))
    beam = np.exp(-((z**2 / 2 / sigma_z**2 + y**2 / 2 / sigma_y**2)))
    beam[x < 0] = 0
    beam_map = np.zeros([3, npix])
    beam_map[0] = beam
    if detB_beam:
        # we make sure that the two detectors within the same pair encode
        # two beams with the  flipped sign in Q   U beams
        beam_map[1] = -beam
    else:
        beam_map[1] = beam
    blm = hp.map2alm(beam_map, lmax=lmax, mmax=mmax)
    hp.rotate_alm(blm, psi=0, theta=-np.pi / 2, phi=0, lmax=lmax, mmax=mmax)

    if normalize_beam:
        # We make sure that the simulated beams are normalized in the test
        # for the normalization we follow the convention adopted in Conviqt,
        # i.e. the monopole term in the map is left unchanged
        idx = hp.Alm.getidx(lmax=lmax, l=0, m=0)
        norm = 2 * np.pi * blm[0, idx].real

    else:
        norm = 1.0

    blm /= norm
    if separate_IQU:
        empty = np.zeros_like(beam_map[0])
        beam_map_I = np.vstack([beam_map[0], empty, empty])
        beam_map_Q = np.vstack([empty, beam_map[1], empty])
        beam_map_U = np.vstack([empty, empty, beam_map[1]])
        try:
            blmi00 = (
                hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax, verbose=False, pol=True)
                / norm
            )
            blm0i0 = (
                hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax, verbose=False, pol=True)
                / norm
            )
            blm00i = (
                hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax, verbose=False, pol=True)
                / norm
            )
        except TypeError:
            # older healpy which does not have verbose keyword
            blmi00 = hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax, pol=True) / norm
            blm0i0 = hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax, pol=True) / norm
            blm00i = hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax, pol=True) / norm
        for b_lm in blmi00, blm0i0, blm00i:
            hp.rotate_alm(b_lm, psi=0, theta=-np.pi / 2, phi=0, lmax=lmax, mmax=mmax)
        return [blmi00, blm0i0, blm00i]

    elif separate_TP:
        blmT = blm[0].copy()
        blmP = blm.copy()
        blmP[0] = 0

        return [blmT, blmP]
    else:
        return blm


def fake_flags(
    data,
    shared_name=defaults.shared_flags,
    shared_val=defaults.shared_mask_invalid,
    det_name=defaults.det_flags,
    det_val=defaults.det_mask_invalid,
):
    """Create fake flags.

    This will flag the first half of each detector's data for all observations.
    """

    for ob in data.obs:
        ob.detdata.ensure(det_name, sample_shape=(), dtype=np.uint8)
        if shared_name not in ob.shared:
            ob.shared.create_column(
                shared_name,
                shape=(ob.n_local_samples,),
                dtype=np.uint8,
            )
        half = ob.n_local_samples // 2
        fshared = None
        if ob.comm_col_rank == 0:
            fshared = np.zeros(ob.n_local_samples, dtype=np.uint8)
            fshared[::100] = shared_val
            fshared |= ob.shared[shared_name].data
        ob.shared[shared_name].set(fshared, offset=(0,), fromrank=0)
        for det in ob.local_detectors:
            ob.detdata[det_name][det, :half] |= det_val


def create_ground_data(
    mpicomm,
    sample_rate=10.0 * u.Hz,
    temp_dir=None,
    el_nod=False,
    el_nods=[-1 * u.degree, 1 * u.degree],
    pixel_per_process=1,
    fknee=None,
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
    toastcomm = create_comm(mpicomm)
    data = Data(toastcomm)

    tele = create_ground_telescope(
        toastcomm.group_size,
        sample_rate=sample_rate,
        pixel_per_process=pixel_per_process,
        fknee=fknee,
    )

    # Create a schedule.

    # FIXME: change this once the ground scheduler supports in-memory creation of the
    # schedule.

    schedule = None

    if mpicomm is None or mpicomm.rank == 0:
        tdir = temp_dir
        if tdir is None:
            tdir = tempfile.mkdtemp()

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
                "2020-01-01 06:00:00",
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
        hwp_rpm=120.0,
        weather="atacama",
        detset_key="pixel",
        elnod_start=el_nod,
        elnods=el_nods,
        scan_accel_az=3 * u.degree / u.second**2,
    )
    sim_ground.apply(data)

    return data


def plot_projected_quats(outfile, qbore=None, qdet=None, valid=slice(None), scale=1.0):
    """Plot a list of quaternion arrays in longitude / latitude."""

    set_matplotlib_backend()
    import matplotlib.pyplot as plt

    # Convert boresight and detector quaternions to angles

    qbang = None
    if qbore is not None:
        qbang = np.zeros((3, qbore.shape[0]), dtype=np.float64)
        qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(qbore)
        qbang[0] *= 180.0 / np.pi
        qbang[1] *= 180.0 / np.pi
        lon_min = np.amin(qbang[0])
        lon_max = np.amax(qbang[0])
        lat_min = np.amin(qbang[1])
        lat_max = np.amax(qbang[1])

    qdang = None
    if qdet is not None:
        qdang = np.zeros((qdet.shape[0], 3, qdet.shape[1]), dtype=np.float64)
        for det in range(qdet.shape[0]):
            qdang[det, 0], qdang[det, 1], qdang[det, 2] = qa.to_lonlat_angles(qdet[det])
            qdang[det, 0] *= 180.0 / np.pi
            qdang[det, 1] *= 180.0 / np.pi
        lon_min = np.amin(qdang[:, 0])
        lon_max = np.amax(qdang[:, 0])
        lat_min = np.amin(qdang[:, 1])
        lat_max = np.amax(qdang[:, 1])

    # Set the sizes of shapes based on the plot range

    span_lon = lon_max - lon_min
    span_lat = lat_max - lat_min
    span = min(span_lon, span_lat)
    bmag = 0.05 * span * scale
    dmag = 0.02 * span * scale

    if span_lat > span_lon:
        fig_y = 10
        fig_x = fig_y * (span_lon / span_lat)
        if fig_x < 4:
            fig_x = 4
    else:
        fig_x = 10
        fig_y = fig_x * (span_lat / span_lon)
        if fig_y < 4:
            fig_y = 4

    figdpi = 100

    fig = plt.figure(figsize=(fig_x, fig_y), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    # Compute the font size to use for detector labels
    fontpix = 0.1 * figdpi
    fontpt = int(0.75 * fontpix)

    # Plot boresight if we have it

    if qbang is not None:
        ax.scatter(qbang[0][valid], qbang[1][valid], color="black", marker="x")
        for ln, lt, ps in np.transpose(qbang)[valid]:
            wd = 0.05 * bmag
            dx = bmag * np.sin(ps)
            dy = -bmag * np.cos(ps)
            ax.arrow(
                ln,
                lt,
                dx,
                dy,
                width=wd,
                head_width=4.0 * wd,
                head_length=0.2 * bmag,
                length_includes_head=True,
                ec="red",
                fc="red",
            )

    # Plot detectors if we have them

    if qdang is not None:
        for idet, dang in enumerate(qdang):
            ax.scatter(dang[0][valid], dang[1][valid], color="blue", marker=".")
            for ln, lt, ps in np.transpose(dang)[valid]:
                wd = 0.05 * dmag
                dx = dmag * np.sin(ps)
                dy = -dmag * np.cos(ps)
                ax.arrow(
                    ln,
                    lt,
                    dx,
                    dy,
                    width=wd,
                    head_width=4.0 * wd,
                    head_length=0.2 * dmag,
                    length_includes_head=True,
                    ec="blue",
                    fc="blue",
                )
            ax.text(
                dang[0][valid][0] + (idet % 2) * 1.5 * dmag,
                dang[1][valid][0] + 1.0 * dmag,
                f"{idet:02d}",
                color="k",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
            )

    # Invert x axis so that longitude reflects what we would see from
    # inside the celestial sphere
    plt.gca().invert_xaxis()

    ax.set_xlabel("Longitude Degrees", fontsize="medium")
    ax.set_ylabel("Latitude Degrees", fontsize="medium")

    fig.suptitle("Projected Pointing and Polarization on Sky")

    plt.savefig(outfile)
    plt.close()
