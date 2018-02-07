# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

import healpy as hp

from .. import qarray as qa

from .tod import TOD

from ..op import Operator

from .tod_math import dipole

from .. import timing


class OpSimDipole(Operator):
    """
    Operator which generates dipole signal for detectors.

    This uses the detector pointing, the telescope velocity vectors, and
    the solar system motion with respect to the CMB rest frame to compute
    the observed CMB dipole signal.  The dipole timestream is either added
    (default) or subtracted from a cache object.

    The telescope velocity and detector quaternions are assumed to be in
    the same coordinate system.

    Args:
        mode (str): this determines what components of the telescope motion
            are included in the observed dipole.  Valid options are 'solar'
            for just the solar system motion, 'orbital' for just the motion
            of the telescope with respect to the solarsystem barycenter, and
            'total' which is the sum of both (and the default).
        coord (str): coordinate system of detector pointing.  Valid values
            are 'C' for equatorial, 'E' for ecliptic and 'G' for galactic.
        subtract (bool): if True, subtract timestream from cache object,
            otherwise add it (default).
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        cmb (float): CMB monopole in Kelvin.  Default value from Fixsen
            2009 (see arXiv:0911.1955)
        solar_speed (float): the amplitude of the solarsystem barycenter
            velocity with respect to the CMB in Km/s.  The default value is
            based on http://arxiv.org/abs/0803.0732.
        solar_gal_lat (float): the latitude in degrees in galactic
            coordinates for the direction of motion of the solarsystem with
            respect to the CMB rest frame.
        solar_gal_lon (float): the longitude in degrees in galactic
            coordinates for the direction of motion of the solarsystem with
            respect to the CMB rest frame.
        freq (float): optional observing frequency in Hz (not GHz).
    """
    def __init__(self, mode='total', coord='C', subtract=False, out='dipole',
                 cmb=2.72548, solar_speed=369.0, solar_gal_lat=48.26,
                 solar_gal_lon=263.99, freq=0, keep_quats=False, keep_vel=False,
                 flag_mask=255, common_flag_mask=255):
        self._mode = mode
        self._coord = coord
        self._subtract = subtract
        self._out = out
        self._cmb = cmb
        self._freq = freq
        self._solar_speed = solar_speed
        self._solar_gal_theta = np.deg2rad(90.0 - solar_gal_lat)
        self._solar_gal_phi = np.deg2rad(solar_gal_lon)
        self._keep_quats = keep_quats
        self._keep_vel = keep_vel
        self._flag_mask = flag_mask
        self._common_flag_mask = common_flag_mask

        projected = self._solar_speed * np.sin(self._solar_gal_theta)
        z = self._solar_speed * np.cos(self._solar_gal_theta)
        x = projected * np.cos(self._solar_gal_phi)
        y = projected * np.sin(self._solar_gal_phi)
        self._solar_gal_vel = np.array([x, y, z])

        # rotate solar system velocity to desired coordinate frame

        if self._coord == 'G':
            self._solar_vel = self._solar_gal_vel
        else:
            rotmat = hp.rotator.Rotator(coord=['G', self._coord]).mat
            self._solar_vel = np.ravel(np.dot(rotmat, self._solar_gal_vel))

        super().__init__()


    @timing.auto_timer
    def exec(self, data):
        """
        Create the timestreams.

        This loops over all observations and detectors and uses the pointing,
        the telescope motion, and the solar system motion to compute the
        observed dipole.

        Args:
            data (toast.Data): The distributed data.
        """
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

        for obs in data.obs:
            tod = obs['tod']

            offset, nsamp = tod.local_samples

            vel = None
            sol = None

            if (self._mode == 'solar') or (self._mode == 'total'):
                sol = self._solar_vel
            if (self._mode == 'orbital') or (self._mode == 'total'):
                vel = tod.local_velocity()

            common = tod.local_common_flags() & self._common_flag_mask

            for det in tod.local_dets:

                flags = tod.local_flags(det) & self._flag_mask
                totflags = (flags | common)
                del flags

                pdata = tod.local_pointing(det).copy()
                pdata[(totflags != 0), :] = nullquat

                dipoletod = dipole(pdata, vel=vel, solar=sol, cmb=self._cmb,
                                   freq=self._freq)
                del pdata

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (nsamp, ))

                ref = tod.cache.reference(cachename)
                if self._subtract:
                    ref[:] -= dipoletod
                else:
                    ref[:] += dipoletod
                del ref

                if not self._keep_quats:
                    cachename = 'quat_{}'.format(det)
                    tod.cache.destroy(cachename)

            del common

            if vel is not None and not self._keep_vel:
                del vel
                tod.cache.destroy('velocity')

        return
