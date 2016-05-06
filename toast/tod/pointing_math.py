# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np

import healpy as hp

import quaternionarray as qa

from scipy.constants import c

cinv = 1e3 / c # Inverse light speed in km / s ( the assumed unit for velocity )

xaxis, yaxis, zaxis = np.eye( 3, dtype=np.float64 )


def quat2angle( quat, no_pa=False ):
    """
    Convert orientation quaternions into pointing angles.

    Args:
        quat (float):  Normalized quaternions.
        no_pa (float):  Only return the pointing angles without position angle.

    Returns:
        2- or 3-tuple of pointing angles
    """

    vec_dir = qa.rotate( quat, zaxis )

    theta, phi = hp.vec2ang( vec_dir )

    if no_pa: return theta, phi

    vec_dir = vec_dir.T
    vec_orient = qa.rotate( quat, xaxis ).T
        
    ypa = vec_orient[0]*vec_dir[1] - vec_orient[1]*vec_dir[0]
    xpa = -vec_orient[0]*vec_dir[2]*vec_dir[0] - vec_orient[1]*vec_dir[2]*vec_dir[1] \
          + vec_orient[2]*(vec_dir[0]**2 + vec_dir[1]**2)

    psi = np.arctan2( ypa, xpa )

    return theta, phi, psi


def aberrate( quat, vel, inplace=True ):
    """
    Apply velocity aberration to the orientation quaternions.

    Args:
        quat (float):  Normalized quaternions.
        vel (float):  Telescope velocity with respect to the signal rest frame.

    Returns:
        Corrected quaternions either in the orinal container or a 2D ndarray.
    """
    
    vec = qa.rotate( quat, zaxis )
    abvec = np.cross( vec, vel[ind] )
    lens = np.linalg.norm( abvec, axis=1 )
    ang = lens * cinv
    abvec /= np.tile( lens, (3,1) ).T # Normalize for direction
    abquat = qa.rotation( abvec, -ang )

    if inplace:
        quat[:] = qa.mult( abquat, quat )
        return
    
    return qa.mult( abquat, quat )

