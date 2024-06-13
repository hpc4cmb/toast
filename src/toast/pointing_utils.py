# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# Pointing utility functions used by templates and operators

import numpy as np
from astropy import units as u

from . import qarray as qa
from .instrument_coords import quat_to_xieta
from .mpi import MPI
from .timing import GlobalTimers, Timer, function_timer


def center_offset_lonlat(
    quats,
    center_offset=None,
    degrees=False,
    is_azimuth=False,
):
    """Compute relative longitude / latitude from a dynamic center position.

    Args:
        quats (array):  Input pointing quaternions
        center_offset (array):  Center longitude, latitude in radians for each sample
        degrees (bool):  If True, return longitude / latitude values in degrees
        is_azimuth (bool):  If True, we are using azimuth and the sign of the
            longitude values should be negated

    Returns:
        (tuple):  The (longitude, latitude) arrays

    """
    if center_offset is None:
        lon_rad, lat_rad, _ = qa.to_lonlat_angles(quats)
    else:
        if len(quats.shape) == 2:
            n_samp = quats.shape[0]
        else:
            n_samp = 1
        if center_offset.shape[0] != n_samp:
            msg = f"center_offset dimensions {center_offset.shape}"
            msg += f" not compatible with {n_samp} quaternion values"
            raise ValueError(msg)
        q_center = qa.from_lonlat_angles(
            center_offset[:, 0],
            center_offset[:, 1],
            np.zeros_like(center_offset[:, 0]),
        )
        q_final = qa.mult(qa.inv(q_center), quats)
        lon_rad, lat_rad, _ = quat_to_xieta(q_final)
    if is_azimuth:
        lon_rad = 2 * np.pi - lon_rad
    # Normalize range
    shift = lon_rad >= 2 * np.pi
    lon_rad[shift] -= 2 * np.pi
    shift = lon_rad < 0
    lon_rad[shift] += 2 * np.pi
    # Convert units
    if degrees:
        lon = np.degrees(lon_rad)
        lat = np.degrees(lat_rad)
    else:
        lon = lon_rad
        lat = lat_rad
    return (lon, lat)


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

    # The observation samples we are considering
    if samples is None:
        slc = slice(0, obs.n_local_samples, 1)
    else:
        slc = samples

    # Apply the flags to boresight pointing if needed.
    bore_quats = np.array(obs.shared[boresight].data[slc, :])
    if flags is not None:
        fdata = np.array(obs.shared[flags].data[slc])
        fdata &= flag_mask
        bore_quats = bore_quats[fdata == 0, :]

    # The remaining good samples we have left
    n_good = bore_quats.shape[0]

    # Check that the top of the focalplane is below the zenith
    _, el_bore, _ = qa.to_lonlat_angles(bore_quats)
    elmax_bore = np.amax(el_bore)
    if elmax_bore + fp_radius > np.pi / 2:
        msg = f"The scan range includes the zenith."
        msg += f" Max boresight elevation is {np.degrees(elmax_bore)} deg"
        msg += f" and focalplane radius is {np.degrees(fp_radius)} deg."
        msg += " Scan range facility cannot handle this case."
        raise RuntimeError(msg)

    # Work in parallel
    rank = obs.comm.group_rank
    ntask = obs.comm.group_size
    rank_slice = slice(rank, n_good, ntask)

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

    # Get source center positions if needed
    center_lonlat = None
    if center_offset is not None:
        center_lonlat = np.array(
            (obs.shared[center_offset].data[slc, :])[rank_slice, :]
        )
        # center_offset is in degrees
        center_lonlat[:, :] *= np.pi / 180.0

    # Compute pointing of fake detectors
    lon = list()
    lat = list()
    for idet, detquat in enumerate(detquats):
        dquats = qa.mult(bore_quats[rank_slice, :], detquat)
        det_lon, det_lat = center_offset_lonlat(
            dquats,
            center_offset=center_lonlat,
            degrees=False,
            is_azimuth=is_azimuth,
        )
        lon.append(det_lon)
        lat.append(det_lat)

    lon = np.unwrap(np.hstack(lon))
    lat = np.hstack(lat)

    # find the extremes
    lonmin = np.amin(lon)
    lonmax = np.amax(lon)
    latmin = np.amin(lat)
    latmax = np.amax(lat)

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
