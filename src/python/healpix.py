# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from . import ctoast as ctoast


def ang2vec(theta, phi):
    n = len(theta)
    if len(phi) != n:
        raise RuntimeError("theta / phi vectors must have the same length")
    vec = np.zeros(3*n, dtype=np.float64)
    ctoast.healpix_ang2vec(n, theta.flatten().astype(np.float64, copy=False), phi.flatten().astype(np.float64, copy=False), vec)
    if n == 1:
        return vec
    else:
        return vec.reshape((-1,3))


def vec2ang(vec):
    n = None
    if vec.ndim == 1:
        n = 1
    else:
        n = vec.shape[0]
    theta = np.zeros(n, dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    ctoast.healpix_vec2ang(n, vec.flatten().astype(np.float64, copy=False), theta, phi)
    if n == 1:
        return (theta[0], phi[0])
    else:
        return (theta, phi)


def vecs2angpa(vec):
    n = None
    if vec.ndim == 1:
        n = 1
    else:
        n = vec.shape[0]
    theta = np.zeros(n, dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    pa = np.zeros(n, dtype=np.float64)
    ctoast.healpix_vecs2angpa(n, vec.flatten().astype(np.float64, copy=False), theta, phi, pa)
    if n == 1:
        return (theta[0], phi[0], pa[0])
    else:
        return (theta, phi, pa)


class Pixels(object):
    """
    Class for HEALPix pixel operations at a fixed NSIDE.

    This class wraps the underlying toast::healpix::pixels class.

    Args:
        nside (int): the map NSIDE.
    """
    def __init__(self, nside=0):
        self.nside = nside
        self.hpix = None
        if self.nside > 0:
            self.hpix = ctoast.healpix_pixels_alloc(self.nside)

    def __del__(self):
        if self.hpix is not None:
            ctoast.healpix_pixels_free(self.hpix)

    def reset(self, nside):
        """
        Reset the class instance to use a new NSIDE value.

        Args:
            nside (int): the map NSIDE.
        """
        self.nside = nside
        
        if self.nside == 0:
            # free the hpix structure
            if self.hpix is not None:
                ctoast.healpix_pixels_free(self.hpix)
            self.hpix = None
        else:
            if self.hpix is None:
                self.hpix = ctoast.healpix_pixels_alloc(self.nside)
            else:
                ctoast.healpix_pixels_reset(self.hpix, nside)
        return

    def vec2zphi(self, vec):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = None
        if vec.ndim == 1:
            n = 1
        else:
            n = vec.shape[0]
        phi = np.zeros(n, dtype=np.float64)
        region = np.zeros(n, dtype=np.int32)
        z = np.zeros(n, dtype=np.float64)
        rtz = np.zeros(n, dtype=np.float64)
        ctoast.healpix_pixels_vec2zphi(self.hpix, n, 
            vec.flatten().astype(np.float64, copy=False), 
            phi, region, z, rtz)
        return (phi, region, z, rtz)

    def theta2z(self, theta):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(theta)
        region = np.zeros(n, dtype=np.int32)
        z = np.zeros(n, dtype=np.float64)
        rtz = np.zeros(n, dtype=np.float64)
        ctoast.healpix_pixels_theta2z(self.hpix, n, 
            theta.flatten().astype(np.float64, copy=False), 
            region, z, rtz)
        return (region, z, rtz)

    def zphi2nest(self, phi, region, z, rtz):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(phi)
        if len(region) != n:
            raise RuntimeError("All inputs must be the same length")
        if len(z) != n:
            raise RuntimeError("All inputs must be the same length")
        if len(rtz) != n:
            raise RuntimeError("All inputs must be the same length")
        pix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_zphi2nest(self.hpix, n, 
            phi.flatten().astype(np.float64, copy=False), 
            region.flatten().astype(np.int32, copy=False), 
            z.flatten().astype(np.float64, copy=False), 
            rtz.flatten().astype(np.float64, copy=False), pix)
        return pix

    def zphi2ring(self, phi, region, z, rtz):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(phi)
        if len(region) != n:
            raise RuntimeError("All inputs must be the same length")
        if len(z) != n:
            raise RuntimeError("All inputs must be the same length")
        if len(rtz) != n:
            raise RuntimeError("All inputs must be the same length")
        pix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_zphi2nest(self.hpix, n, 
            phi.flatten().astype(np.float64, copy=False), 
            region.flatten().astype(np.int32, copy=False), 
            z.flatten().astype(np.float64, copy=False), 
            rtz.flatten().astype(np.float64, copy=False), pix)
        return pix

    def ang2nest(self, theta, phi):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(theta)
        if len(phi) != n:
            raise RuntimeError("All inputs must be the same length")
        pix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_ang2nest(self.hpix, n, 
            theta.flatten().astype(np.float64, copy=False), 
            phi.flatten().astype(np.float64, copy=False), pix)
        return pix

    def ang2ring(self, theta, phi):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(theta)
        if len(phi) != n:
            raise RuntimeError("All inputs must be the same length")
        pix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_ang2ring(self.hpix, n, 
            theta.flatten().astype(np.float64, copy=False), 
            phi.flatten().astype(np.float64, copy=False), pix)
        return pix

    def vec2nest(self, vec):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = None
        if vec.ndim == 1:
            n = 1
        else:
            n = vec.shape[0]
        pix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_vec2nest(self.hpix, n, 
            vec.flatten().astype(np.float64, copy=False), pix)
        return pix

    def vec2ring(self, vec):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = None
        if vec.ndim == 1:
            n = 1
        else:
            n = vec.shape[0]
        pix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_vec2ring(self.hpix, n, 
            vec.flatten().astype(np.float64, copy=False), pix)
        return pix

    def ring2nest(self, ringpix):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(ringpix)
        nestpix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_ring2nest(self.hpix, n, 
            ringpix.flatten().astype(np.int64, copy=False), nestpix)
        return nestpix

    def nest2ring(self, nestpix):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(nestpix)
        ringpix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_nest2ring(self.hpix, n, 
            nestpix.flatten().astype(np.int64, copy=False), ringpix)
        return ringpix

    def degrade_ring(self, factor, inpix):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(inpix)
        outpix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_degrade_ring(self.hpix, factor, n, 
            inpix.flatten().astype(np.int64, copy=False), outpix)
        return outpix

    def degrade_nest(self, factor, inpix):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(inpix)
        outpix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_degrade_nest(self.hpix, factor, n, 
            inpix.flatten().astype(np.int64, copy=False), outpix)
        return outpix

    def upgrade_ring(self, factor, inpix):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(inpix)
        outpix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_upgrade_ring(self.hpix, factor, n, 
            inpix.flatten().astype(np.int64, copy=False), outpix)
        return outpix

    def upgrade_nest(self, factor, inpix):
        if self.hpix is None:
            raise RuntimeError("healpix Pixels class must be initialized with an NSIDE value")
        n = len(inpix)
        outpix = np.zeros(n, dtype=np.int64)
        ctoast.healpix_pixels_upgrade_nest(self.hpix, factor, n, 
            inpix.flatten().astype(np.int64, copy=False), outpix)
        return outpix

