# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np
import scipy.constants as constants
from astropy import units as u

from . import qarray as qa
from .timing import function_timer
from .utils import array_dot


# A&A 643, A42 (2020) Table 11.
# Amplitude = 3366.6 uK is converted to speed according to
# r = amplitude / T_CMB
# v = (r**2 + 2 * r) / (r**2 + 2 * r + 2) * c
t_cmb = 2.72548 * u.Kelvin
solar_speed = 370.08 * u.kilometer / u.second
solar_gal_lat = 48.25 * u.degree
solar_gal_lon = 263.99 * u.degree


@function_timer
def dipole(det_pointing, vel=None, solar=None, cmb=t_cmb, freq=0 * u.Hz):
    """Compute a dipole timestream.

    This uses detector pointing, telescope velocity and the solar system
    motion to compute the observed dipole.  It is assumed that the detector
    pointing, the telescope velocity vectors, and the solar system velocity
    vector are all in the same coordinate system.

    Args:
        pntg (array): the 2D array of quaternions of detector pointing.
        vel (array): 2D array of velocity vectors relative to the solar
            system barycenter.  if None, return only the solar system dipole.
            Units are km/s
        solar (array): a 3 element vector containing the solar system velocity
            vector relative to the CMB rest frame.  Units are km/s.
        cmb (Quantity): CMB monopole in Kelvin.  Default value from Fixsen
            2009 (see arXiv:0911.1955)
        freq (Quantity): optional observing frequency.

    Returns:
        (array):  detector dipole timestream.

    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    nsamp = det_pointing.shape[0]

    inv_light = 1.0e3 / constants.speed_of_light

    if (vel is not None) and (solar is not None):
        # relativistic addition of velocities

        solar_speed = np.sqrt(np.sum(solar * solar, axis=0))

        vpar = (array_dot(vel, solar) / solar_speed**2) * solar
        vperp = vel - vpar

        vdot = 1.0 / (1.0 + array_dot(solar, vel) * inv_light**2)
        invgamma = np.sqrt(1.0 - (solar_speed * inv_light) ** 2)

        vpar += solar
        vperp *= invgamma

        v = vdot * (vpar + vperp)
    elif solar is not None:
        v = np.tile(solar, nsamp).reshape((-1, 3))
    elif vel is not None:
        v = vel.copy()

    speed = np.sqrt(array_dot(v, v))
    v /= speed

    beta = inv_light * speed.flatten()

    direct = qa.rotate(det_pointing, zaxis)

    dipoletod = None

    cmb_kelvin = cmb.to_value(u.Kelvin)
    freq_hz = freq.to_value(u.Hz)

    if freq_hz == 0:
        inv_gamma = np.sqrt(1.0 - beta**2)
        num = 1.0 - beta * np.sum(v * direct, axis=1)
        dipoletod = cmb_kelvin * (inv_gamma / num - 1.0)
    else:
        # Use frequency for quadrupole correction
        fx = constants.h * freq_hz / (constants.k * cmb_kelvin)
        fcor = (fx / 2) * (np.exp(fx) + 1) / (np.exp(fx) - 1)
        bt = beta * np.sum(v * direct, axis=1)
        dipoletod = cmb_kelvin * (bt + fcor * bt**2)

    return dipoletod


def dipole_map(nside, freq=0 * u.Hz, coord="G"):
    # Compute the solar system velocity in galactic coordinates

    solar_gal_theta = np.deg2rad(90.0 - solar_gal_lat.to_value(u.degree))
    solar_gal_phi = np.deg2rad(solar_gal_lon.to_value(u.degree))

    solar_speed_kms = solar_speed.to_value(u.kilometer / u.second)
    solar_projected = solar_speed_kms * np.sin(solar_gal_theta)

    sol_z = solar_speed_kms * np.cos(solar_gal_theta)
    sol_x = solar_projected * np.cos(solar_gal_phi)
    sol_y = solar_projected * np.sin(solar_gal_phi)
    solar_gal_vel = np.array([sol_x, sol_y, sol_z])

    # Rotate solar system velocity to desired coordinate frame

    solar_vel = None
    if coord == "G":
        solar_vel = solar_gal_vel
    else:
        rotmat = hp.rotator.Rotator(coord=["G", self.coord]).mat
        solar_vel = np.ravel(np.dot(rotmat, solar_gal_vel))

    # Evaluate in each pixel

    pix = np.arange(12 * nside**2)
    theta, phi = hp.pix2ang(nside, pix)
    quat = qa.from_iso_angles(theta, phi, np.zeros_like(theta))
    dipo = dipole(quat, solar=solar_vel, freq=freq)

    return dipo
