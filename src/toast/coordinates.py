# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
from collections import OrderedDict
from datetime import datetime, timezone

import ephem
import numpy as np
from astropy import coordinates as acoord
from astropy import time as atime
from astropy import units as u

from . import qarray as qa
from .healpix import ang2vec
from .timing import function_timer


def to_UTC(t):
    # Convert UNIX time stamp to a date string
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def to_JD(t):
    # Unix time stamp to Julian date
    # (days since -4712-01-01 12:00:00 UTC)
    return t / 86400.0 + 2440587.5


def to_MJD(t):
    # Convert Unix time stamp to modified Julian date
    # (days since 1858-11-17 00:00:00 UTC)
    return to_JD(t) - 2400000.5


def to_DJD(t):
    # Convert Unix time stamp to Dublin Julian date
    # (days since 1899-12-31 12:00:00)
    # This is the time format used by PyEphem
    return to_JD(t) - 2415020


def DJDtoUNIX(djd):
    # Convert Dublin Julian date to a UNIX time stamp
    return ((djd + 2415020) - 2440587.5) * 86400.0


def _ephem_transform(site, t):
    """Get the Az/El -> Ra/Dec conversion quaternion for boresight."""
    observer = ephem.Observer()
    observer.lon = site.earthloc.lon.to_value(u.radian)
    observer.lat = site.earthloc.lat.to_value(u.radian)
    observer.elevation = site.earthloc.height.to_value(u.meter)
    observer.epoch = ephem.J2000
    observer.compute_pressure()

    observer.date = to_DJD(t)
    observer.pressure = 0

    # Rotate the X, Y and Z axes from horizontal to equatorial frame.
    # Strictly speaking, two coordinate axes would suffice but the
    # math is cleaner with three axes.
    #
    # PyEphem measures the azimuth East (clockwise) from North.
    # The direction is standard but opposite to ISO spherical coordinates.
    try:
        xra, xdec = observer.radec_of(0, 0, fixed=False)
        yra, ydec = observer.radec_of(-np.pi / 2, 0, fixed=False)
        zra, zdec = observer.radec_of(0, np.pi / 2, fixed=False)
    except Exception as e:
        # Modified pyephem not available.
        # Translated pointing will include stellar aberration.
        xra, xdec = observer.radec_of(0, 0)
        yra, ydec = observer.radec_of(-np.pi / 2, 0)
        zra, zdec = observer.radec_of(0, np.pi / 2)
    xvec, yvec, zvec = ang2vec(
        np.pi / 2 - np.array([xdec, ydec, zdec]), np.array([xra, yra, zra])
    )
    # Orthonormalize for numerical stability
    xvec /= np.sqrt(np.dot(xvec, xvec))
    yvec -= np.dot(xvec, yvec) * xvec
    yvec /= np.sqrt(np.dot(yvec, yvec))
    zvec -= np.dot(xvec, zvec) * xvec + np.dot(yvec, zvec) * yvec
    zvec /= np.sqrt(np.dot(zvec, zvec))
    # Solve for the quaternions from the transformed axes.
    X = (xvec[1] + yvec[0]) / 4
    Y = (xvec[2] + zvec[0]) / 4
    Z = (yvec[2] + zvec[1]) / 4
    d = np.sqrt(np.abs(Y * Z / X))  # Choose positive root
    c = d * X / Y
    b = X / c
    a = (xvec[1] / 2 - b * c) / d
    # qarray has the scalar part as the last index
    quat = qa.norm(np.array([b, c, d, a]))
    return quat


def _transform(site, obstime):
    """Helper function to get the coordinate transform quaternions."""
    # azel_frame = acoord.AltAz(
    #     location=site.earthloc,
    #     obstime=obstime,
    # )
    radec_frame = acoord.ICRS()

    # X axis
    azel_frame = acoord.AltAz(
        location=site.earthloc,
        obstime=obstime,
        az=np.zeros_like(obstime) * u.radian,
        alt=np.zeros_like(obstime) * u.radian,
    )
    # azel_frame.az = np.zeros_like(obstime) * u.radian
    # azel_frame.alt = np.zeros_like(obstime) * u.radian
    radec = azel_frame.transform_to(acoord.ICRS())
    xvec = ang2vec(
        np.pi / 2 - radec.dec.to_value(u.radian), radec.ra.to_value(u.radian)
    ).reshape((-1, 3))

    # Y axis
    azel_frame = acoord.AltAz(
        location=site.earthloc,
        obstime=obstime,
        az=(-np.pi / 2) * np.ones_like(obstime) * u.radian,
        alt=np.zeros_like(obstime) * u.radian,
    )
    # azel_frame.az = (-np.pi / 2) * np.ones_like(obstime) * u.radian
    # azel_frame.alt = np.zeros_like(obstime) * u.radian
    radec = azel_frame.transform_to(acoord.ICRS())
    yvec = ang2vec(
        np.pi / 2 - radec.dec.to_value(u.radian), radec.ra.to_value(u.radian)
    ).reshape((-1, 3))

    # Z axis
    azel_frame = acoord.AltAz(
        location=site.earthloc,
        obstime=obstime,
        az=np.zeros_like(obstime) * u.radian,
        alt=(np.pi / 2) * np.ones_like(obstime) * u.radian,
    )
    # azel_frame.az = np.zeros_like(obstime) * u.radian
    # azel_frame.alt = (np.pi / 2) * np.ones_like(obstime) * u.radian
    radec = azel_frame.transform_to(acoord.ICRS())
    zvec = ang2vec(
        np.pi / 2 - radec.dec.to_value(u.radian), radec.ra.to_value(u.radian)
    ).reshape((-1, 3))

    # Orthonormalize for numerical stability
    # FIXME:  use tensordot.
    # xvec /= np.sqrt(np.dot(xvec[:], xvec[:]))
    # yvec -= np.dot(xvec[:], yvec[:]) * xvec
    # yvec /= np.sqrt(np.dot(yvec[:], yvec[:]))
    # zvec -= np.dot(xvec[:], zvec[:]) * xvec + np.dot(yvec[:], zvec[:]) * yvec
    # zvec /= np.sqrt(np.dot(zvec[:], zvec[:]))

    # Solve for the quaternions from the transformed axes.
    X = (xvec[:, 1] + yvec[:, 0]) / 4
    Y = (xvec[:, 2] + zvec[:, 0]) / 4
    Z = (yvec[:, 2] + zvec[:, 1]) / 4
    d = np.sqrt(np.abs(Y * Z / X))  # Choose positive root
    c = d * X / Y
    b = X / c
    a = (xvec[:, 1] / 2 - b * c) / d
    # qarray has the scalar part as the last index
    return qa.norm(np.vstack([b, c, d, a]).T.reshape((-1, 4)))


@function_timer
def azel_to_radec(site, times, azel, use_ephem=False):
    """Transform Az / El quaternions to RA / DEC.

    This uses the Earth location of the ground site to convert timestamped Az / El
    quaternions into RA / DEC.  This uses astropy coordinate transforms which in turn
    use IERS data (either internal or downloaded).  Before using this function, the
    calling code should use `toast.utils.astropy_control()` to set the desired
    behavior.  Note that even if using high-resolution, downloaded IERS, the predictions
    to times beyond one year in the future will include inaccuracies at a fraction of
    an arcsecond.

    Args:
        site (GroundSite):  The ground site location of the telescope.
        times (array):  The timestamps in UTC seconds.
        azel (array):  The Az / El boresight quaternions.
        use_ephem (bool):  Use pyephem instead of astropy

    Returns:
        (array):  The RA / DEC boresight quaternions.

    """
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    # The conversion between coordinate systems is slowly varying.  We compute the
    # transform quaternion at sparse time stamps and then use SLERP to get the full
    # times.
    sparse_step = 120.0  # 2 minutes
    sparse_nstep = int((times[-1] - times[0]) / sparse_step) + 1
    sparse_times = np.linspace(times[0], times[-1], num=sparse_nstep, endpoint=True)
    n_sparse = len(sparse_times)

    if use_ephem:
        sparse_quat = np.array([_ephem_transform(site, t) for t in sparse_times])
        # print("ephem: ", sparse_quat)
    else:
        # Astropy format times
        ast_times = [atime.Time(x, format="unix") for x in sparse_times]
        sparse_quat = _transform(site, ast_times)
        # print("astropy: ", sparse_quat)

    # Make sure we have a consistent branch in the quaternions.
    # Otherwise we'll get interpolation issues.

    for i in range(n_sparse):
        if i > 0 and (
            np.sum(np.abs(sparse_quat[i - 1] + sparse_quat[i]))
            < np.sum(np.abs(sparse_quat[i - 1] - sparse_quat[i]))
        ):
            sparse_quat[i] *= -1

    sparse_quat = qa.norm(sparse_quat)

    # Construct dense transform
    transform = qa.slerp(times, sparse_times, sparse_quat)

    # return qa.mult(azel, transform)
    # return qa.mult(azel, qa.inv(transform))
    return qa.mult(transform, azel)
    # return qa.mult(qa.inv(transform), azel)
