# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np


def calibrate( toitimes, toi, gaintimes, gains, order=0, inplace=False ):
    """
    Interpolate the gains to TOI samples and apply them.
    Args:
        toitimes (float):  Increasing TOI sample times in same units as gaintimes
        toi (float):  TOI samples to calibrate
        gaintimes (float):  Increasing timestamps of the gain values in same units
                            as toitimes
        gains (float):  Multiplicative gains
        order (int):  Gain interpolation order. 0 means steps at the gain times,
                      all other are polynomial interpolations.
        inplace (bool):  Overwrite input TOI.
    """

    if len(gaintimes) == 1:
        g = gains
    else:    
        if order == 0:
            ind = np.searchsorted( gaintimes, toitimes )
            ind[ind > 0] -= 1
            g = gains[ind]
        else:
            if len(gaintimes) <= order:
                order = len(gaintimes) - 1
            p = np.polyfit( gaintimes, gains, order )
            g = np.polyval( p, toitimes )

    if inplace:
        toi_out = toi
    else:
        toi_out = np.zeros_like(toi)

    toi_out[:] = toi * g

    return toi_out

