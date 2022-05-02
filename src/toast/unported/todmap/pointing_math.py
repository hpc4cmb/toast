# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.constants import c

from .. import qarray as qa
from ..timing import function_timer

cinv = 1e3 / c  # Inverse light speed in km / s ( the assumed unit for velocity )

xaxis, yaxis, zaxis = np.eye(3, dtype=np.float64)

# J2000 coordinate transforms

# RA, DEC to galactic coordinates

coordmat_J2000radec2gal = np.array(
    [
        -0.054875539726,
        -0.873437108010,
        -0.483834985808,
        0.494109453312,
        -0.444829589425,
        0.746982251810,
        -0.867666135858,
        -0.198076386122,
        0.455983795705,
    ]
).reshape([3, 3])
quat_equ2gal = qa.from_rotmat(coordmat_J2000radec2gal)

# RA, DEC to (geocentric) ecliptic coordinates

coordmat_J2000radec2ecl = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.917482062069182,
        0.397777155931914,
        0.0,
        -0.397777155931914,
        0.917482062069182,
    ]
).reshape([3, 3])
quat_equ2ecl = qa.from_rotmat(coordmat_J2000radec2ecl)

# Ecliptic coordinates (geocentric) to galactic
# (use the same rotation as HEALPix, to avoid confusion)
coordmat_J2000ecl2gal = np.array(
    [
        -0.054882486,
        -0.993821033,
        -0.096476249,
        0.494116468,
        -0.110993846,
        0.862281440,
        -0.867661702,
        -0.000346354,
        0.497154957,
    ]
).reshape([3, 3])
quat_ecl2gal = qa.from_rotmat(coordmat_J2000ecl2gal)


@function_timer
def aberrate(quat, vel, inplace=True):
    """Apply velocity aberration to the orientation quaternions.

    Args:
        quat (float):  Normalized quaternions.
        vel (float):  Telescope velocity with respect to the signal rest frame.

    Returns:
        Corrected quaternions either in the orinal container or a 2D ndarray.

    """
    vec = qa.rotate(quat, zaxis)
    abvec = np.cross(vec, vel)
    lens = np.linalg.norm(abvec, axis=1)
    ang = lens * cinv
    abvec /= np.tile(lens, (3, 1)).T  # Normalize for direction
    abquat = qa.rotation(abvec, -ang)

    if inplace:
        quat[:] = qa.mult(abquat, quat)
        return

    return qa.mult(abquat, quat)
