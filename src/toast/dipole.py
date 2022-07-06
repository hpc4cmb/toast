# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import scipy.constants as constants
from astropy import units as u

from . import qarray as qa
from .timing import function_timer
from .utils import array_dot


@function_timer
def dipole(det_pointing, vel=None, solar=None, cmb=2.72548 * u.Kelvin, freq=0 * u.Hz):
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
