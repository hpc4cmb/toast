# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# Pointing utility functions used by templates and operators


import numpy as np
from astropy import units as u

from . import qarray as qa
from .mpi import MPI
from .timing import GlobalTimers, Timer, function_timer


@function_timer
def scan_range_lonlat(
    obs,
    boresight,
    flags=None,
    flag_mask=0,
    field_of_view=None,
    is_azimuth=False,
    center_offset=None,
    samples=None,
):
    """Compute the range of detector pointing in longitude / latitude.

    This uses the focalplane field of view and the boresight pointing to compute
    the extent of the pointing for one observation.

    Args:
        obs (Observation):  The observation to process.
        boresight (str):  The boresight Az/El pointing name.
        flags (str):  The flag name to use for excluding pointing samples
        flag_mask (int):  The flag mask for excluding samples.
        field_of_view (Quantity):  Override the focalplane FOV.
        is_azimuth (bool):  If True, we are working in Az/El coordinates where
            the azimuth angle is the negative of the ISO phi angle.
        center_offset (str):  The optional source
        samples (slice):  The sample slice to use for the calculation.

    Returns:
        (tuple):  The (Lon min, Lon max, Lat min, Lat max) as quantities.

    """
    if field_of_view is not None:
        fov = field_of_view
    else:
        fov = obs.telescope.focalplane.field_of_view
    fp_radius = 0.5 * fov.to_value(u.radian)

    if samples is None:
        slc = slice(0, obs.n_local_samples, 1)
    else:
        slc = samples

    # Get the flags if needed.
    fdata = None
    if flags is not None:
        fdata = np.array(obs.shared[flags][slc])
        fdata &= flag_mask

    # work in parallel
    rank = obs.comm.group_rank
    ntask = obs.comm.group_size

    # Create a fake focalplane of detectors in a circle around the boresight
    xaxis, yaxis, zaxis = np.eye(3)
    ndet = 64
    phidet = np.linspace(0, 2 * np.pi, ndet, endpoint=False)
    detquats = []
    thetarot = qa.rotation(yaxis, fp_radius)
    for phi in phidet:
        phirot = qa.rotation(zaxis, phi)
        detquat = qa.mult(phirot, thetarot)
        detquats.append(detquat)

    # Get fake detector pointing

    center_lonlat = None
    if center_offset is not None:
        center_lonlat = np.array(obs.shared[center_offset][slc, :])
        center_lonlat[:, :] *= np.pi / 180.0

    lon = []
    lat = []
    quats = obs.shared[boresight][slc, :][rank::ntask].copy()
    rank_good = slice(None)
    if fdata is not None:
        rank_good = fdata[rank::ntask] == 0

    for idet, detquat in enumerate(detquats):
        theta, phi, _ = qa.to_iso_angles(qa.mult(quats, detquat))
        if center_lonlat is None:
            if is_azimuth:
                lon.append(2 * np.pi - phi[rank_good])
            else:
                lon.append(phi[rank_good])
            lat.append(np.pi / 2 - theta[rank_good])
        else:
            if is_azimuth:
                lon.append(
                    2 * np.pi
                    - phi[rank_good]
                    - center_lonlat[rank::ntask, 0][rank_good]
                )
            else:
                lon.append(phi[rank_good] - center_lonlat[rank::ntask, 0][rank_good])
            lat.append(
                (np.pi / 2 - theta[rank_good])
                - center_lonlat[rank::ntask, 1][rank_good]
            )
    lon = np.unwrap(np.hstack(lon))
    lat = np.hstack(lat)

    # find the extremes
    lonmin = np.amin(lon)
    lonmax = np.amax(lon)
    latmin = np.amin(lat)
    latmax = np.amax(lat)

    if lonmin < -2 * np.pi:
        lonmin += 2 * np.pi
        lonmax += 2 * np.pi
    elif lonmax > 2 * np.pi:
        lonmin -= 2 * np.pi
        lonmax -= 2 * np.pi

    # Combine results
    if obs.comm.comm_group is not None:
        lonlatmin = np.zeros(2, dtype=np.float64)
        lonlatmax = np.zeros(2, dtype=np.float64)
        lonlatmin[0] = lonmin
        lonlatmin[1] = latmin
        lonlatmax[0] = lonmax
        lonlatmax[1] = latmax
        all_lonlatmin = np.zeros(2, dtype=np.float64)
        all_lonlatmax = np.zeros(2, dtype=np.float64)
        obs.comm.comm_group.Allreduce(lonlatmin, all_lonlatmin, op=MPI.MIN)
        obs.comm.comm_group.Allreduce(lonlatmax, all_lonlatmax, op=MPI.MAX)
        lonmin = all_lonlatmin[0]
        latmin = all_lonlatmin[1]
        lonmax = all_lonlatmax[0]
        latmax = all_lonlatmax[1]

    return (
        lonmin * u.radian,
        lonmax * u.radian,
        latmin * u.radian,
        latmax * u.radian,
    )
