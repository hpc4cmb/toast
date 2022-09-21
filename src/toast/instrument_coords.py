# Copyright (c) 2015-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from . import qarray as qa


def xieta_to_iso(xi, eta, gamma):
    """Convert xi, eta, gamma coordinates to ISO theta, phi, psi.

    The xi, eta, gamma coordinate system is a 2D projection useful for focalplane
    visualization and comparison.

    Args:
        xi (array):  Array or scalar xi values in radians
        eta (array):  Array or scalar eta values in radians
        gamma (array):  Array or scalar gamma values in radians

    Returns:
        (tuple):  Tuple of theta, phi, psi values

    """
    eps = 1.0e-12
    theta = np.arcsin(np.sqrt(xi**2 + eta**2))
    try:
        lt = len(theta)
        # Handle special values
        theta_zero = theta < eps
        theta_pi = np.pi - theta < eps
        theta_extreme = np.logical_or(theta_zero, theta_pi)
        theta_normal = np.logical_not(theta_extreme)

        theta[theta_zero] = 0.0
        theta[theta_pi] = np.pi

        phi = np.zeros_like(theta)
        phi[theta_normal] = np.arctan2(-xi[theta_normal], -eta[theta_normal])
        psi = gamma - phi
    except TypeError:
        # Scalar values
        if theta < eps:
            theta = 0.0
            phi = 0.0
        elif np.pi - theta < eps:
            theta = np.pi
            phi = 0.0
        else:
            phi = np.arctan2(-xi, -eta)
        psi = gamma - phi
    return (theta, phi, psi)


def iso_to_xieta(theta, phi, psi):
    """Convert ISO theta, phi, psi coordinates to xi, eta, gamma coordinates.

    The xi, eta, gamma coordinate system is a 2D projection useful for focalplane
    visualization and comparison.

    Args:
        theta (array):  Array or scalar theta values in radians
        phi (array):  Array or scalar phi values in radians
        psi (array):  Array or scalar psi values in radians

    Returns:
        (tuple):  Tuple of xi, eta, gamma values

    """
    eps = 1.0e-12
    stheta = np.sin(theta)
    try:
        lt = len(theta)
        # Handle special values
        theta_normal = np.logical_and(theta > eps, np.pi - theta > eps)

        xi = np.zeros_like(theta)
        xi[theta_normal] = -stheta[theta_normal] * np.sin(phi[theta_normal])

        eta = np.zeros_like(theta)
        eta[theta_normal] = -stheta[theta_normal] * np.cos(phi[theta_normal])

        gamma = psi + phi
    except TypeError:
        # Scalar values
        if (theta > eps) and (np.pi - theta > eps):
            # Not extremes
            xi = -stheta * np.sin(phi)
            eta = -stheta * np.cos(phi)
        else:
            # special values
            xi = 0.0
            eta = 0.0
        gamma = psi + phi
    return (xi, eta, gamma)


def xieta_to_quat(xi, eta, gamma):
    """Convert xi, eta, gamma coordinates to quaternions.

    The xi, eta, gamma coordinate system is a 2D projection useful for focalplane
    visualization and comparison.

    Args:
        xi (array):  Array or scalar xi values in radians
        eta (array):  Array or scalar eta values in radians
        gamma (array):  Array or scalar gamma values in radians

    Returns:
        (array):  Array of one or more quaternions

    """
    theta, phi, psi = xieta_to_iso(xi, eta, gamma)
    return qa.from_iso_angles(theta, phi, psi)


def quat_to_xieta(quats):
    """Convert quaternions to xi, eta, gamma coordinates.

    The xi, eta, gamma coordinate system is a 2D projection useful for focalplane
    visualization and comparison.

    Args:
        quats (array):  Array of one or more quaternions

    Returns:
        (tuple):  Tuple of xi, eta, gamma values in radians

    """
    theta, phi, psi = qa.to_iso_angles(quats)
    return iso_to_xieta(theta, phi, psi)
