# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

import healpy as hp

from .. import qarray as qa

from .tod import TOD

from .noise import Noise

from ..op import Operator

from .tod_math import sim_noise_timestream, dipole


class OpSimNoise(Operator):
    """
    Operator which generates noise timestreams.

    This passes through each observation and every process generates data
    for its assigned samples.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        realization (int): if simulating multiple realizations, the realization
            index.
        component (int): the component index to use for this noise simulation.
        noise (str): PSD key in the observation dictionary.
    """

    def __init__(self, out='noise', realization=0, component=0, noise='noise',
                 rate=None, altFFT=False):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._out = out
        self._oversample = 2
        self._realization = realization
        self._component = component
        self._noisekey = noise
        self._rate = rate
        self._altfft = altFFT

    @property
    def timedist(self):
        return self._timedist

    def exec(self, data):
        """
        Generate noise timestreams.

        This iterates over all observations and detectors and generates
        the noise timestreams based on the noise object for the current
        observation.

        Args:
            data (toast.Data): The distributed data.
        """
        comm = data.comm

        for obs in data.obs:
            obsindx = 0
            if 'id' in obs:
                obsindx = obs['id']
            else:
                print("Warning: observation ID is not set, using zero!")

            telescope = 0
            if 'telescope' in obs:
                telescope = obs['telescope']

            tod = obs['tod']
            if self._noisekey in obs:
                nse = obs[self._noisekey]
            else:
                raise RuntimeError('Observation does not contain noise under '
                                   '"{}"'.format(self._noisekey))
            if tod.local_chunks is None:
                raise RuntimeError('noise simulation for uniform distributed '
                                   'samples not implemented')

            # eventually we'll redistribute, to allow long correlations...

            if self._rate is None:
                times = tod.read_times(local_start=0, n=tod.local_samples[1])

            # Iterate over each chunk.

            tod_first = tod.local_samples[0]
            chunk_first = tod_first

            for curchunk in range(tod.local_chunks[1]):
                abschunk = tod.local_chunks[0] + curchunk
                chunk_samp = tod.total_chunks[abschunk]
                local_offset = chunk_first - tod_first

                if self._rate is None:
                    # compute effective sample rate
                    dt = np.median(np.diff(
                            times[local_offset:local_offset+chunk_samp]))
                    rate = 1.0 / dt
                else:
                    rate = self._rate

                idet = 0
                for det in tod.local_dets:

                    detindx = tod.detindx[det]

                    (nsedata, freq, psd) = sim_noise_timestream(
                        self._realization, telescope, self._component, obsindx,
                        detindx, rate, chunk_first, chunk_samp,
                        self._oversample, nse.freq(det), nse.psd(det), 
                        self._altfft)

                    # write to cache

                    cachename = "{}_{}".format(self._out, det)

                    ref = None
                    if tod.cache.exists(cachename):
                        ref = tod.cache.reference(cachename)
                    else:
                        ref = tod.cache.create(cachename, np.float64,
                                    (tod.local_samples[1],))

                    ref[local_offset:local_offset+chunk_samp] += nsedata
                    del ref

                    idet += 1

                chunk_first += chunk_samp

        return


class OpSimGradient(Operator):
    """
    Generate a fake sky signal as a gradient between the poles.

    This passes through each observation and creates a fake signal timestream
    based on the cartesian Z coordinate of the HEALPix pixel containing the
    detector pointing.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        nside (int): the HEALPix NSIDE value to use.
        min (float): the minimum value to use at the South Pole.
        max (float): the maximum value to use at the North Pole.
        nest (bool): whether to use NESTED ordering.
    """

    def __init__(self, out='grad', nside=512, min=-100.0, max=100.0, nest=False, flag_mask=255, common_flag_mask=255):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._out = out
        self._min = min
        self._max = max
        self._nest = nest
        self._flag_mask = flag_mask
        self._common_flag_mask = common_flag_mask

    def exec(self, data):
        """
        Create the gradient timestreams.

        This pixelizes each detector's pointing and then assigns a 
        timestream value based on the cartesian Z coordinate of the pixel
        center.

        Args:
            data (toast.Data): The distributed data.
        """
        comm = data.comm

        zaxis = np.array([0,0,1], dtype=np.float64)
        nullquat = np.array([0,0,0,1], dtype=np.float64)

        range = self._max - self._min

        for obs in data.obs:
            tod = obs['tod']
            base = obs['baselines']
            nse = obs['noise']
            intrvl = obs['intervals']

            for det in tod.local_dets:
                pdata = np.copy(tod.read_pntg(detector=det, local_start=0,
                                              n=tod.local_samples[1]))
                flags, common = tod.read_flags(detector=det, local_start=0,
                                               n=tod.local_samples[1])
                totflags = flags & self._flag_mask
                totflags |= (common & self._common_flag_mask)

                del flags
                del common

                pdata[totflags != 0,:] = nullquat

                dir = qa.rotate(pdata, zaxis)
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2],
                                    nest=self._nest)
                x, y, z = hp.pix2vec(self._nside, pixels, nest=self._nest)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                z[totflags != 0] = 0.0

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64,
                                     (tod.local_samples[1],))
                ref = tod.cache.reference(cachename)
                ref[:] += z
                del ref
                #print('Grad timestream:', ref, np.sum(ref!=0),' non-zeros', flush=True) # DEBUG

        return

    def sigmap(self):
        """
        (array): Return the underlying signal map (full map on all processes).
        """
        range = self._max - self._min
        pix = np.arange(0, 12*self._nside*self._nside, dtype=np.int64)
        x, y, z = hp.pix2vec(self._nside, pix, nest=self._nest)
        z += 1.0
        z *= 0.5
        z *= range
        z += self._min
        return z


class OpSimScan(Operator):
    """
    Operator which generates sky signal by scanning from a map.

    The signal to use should already be in a distributed pixel structure,
    and local pointing should already exist.

    Args:
        distmap (DistPixels): the distributed map domain data.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
    """
    def __init__(self, distmap=None, pixels='pixels', weights='weights',
                 out='scan'):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._map = distmap
        self._pixels = pixels
        self._weights = weights
        self._out = out

    def exec(self, data):
        """
        Create the timestreams by scanning from the map.

        This loops over all observations and detectors and uses the pointing
        matrix to project the distributed map into a timestream.

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

        for obs in data.obs:
            tod = obs['tod']

            for det in tod.local_dets:

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                nnz = weights.shape[1]

                sm, lpix = self._map.global_to_local(pixels)

                f = (np.dot(weights[x], self._map.data[sm[x], lpix[x]])
                     if (lpix[x] >= 0) else 0
                     for x in range(tod.local_samples[1]))
                maptod = np.fromiter(f, np.float64, count=tod.local_samples[1])

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64,
                                     (tod.local_samples[1],))
                ref = tod.cache.reference(cachename)
                ref[:] += maptod
                
                del ref
                del pixels
                del weights

        return


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
        cmb=2.72548, solar_speed=369.0, solar_gal_lat=48.26, solar_gal_lon=263.99,
        freq=0):

        self._mode = mode
        self._coord = coord
        self._subtract = subtract
        self._out = out
        self._cmb = cmb
        self._freq = freq
        self._solar_speed = solar_speed
        self._solar_gal_theta = np.deg2rad(90.0 - solar_gal_lat)
        self._solar_gal_phi = np.deg2rad(solar_gal_lon)

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

        nullquat = np.array([0,0,0,1], dtype=np.float64)

        for obs in data.obs:
            tod = obs['tod']

            nsamp = tod.local_samples[1]

            vel = None
            sol = None

            if (self._mode == 'solar') or (self._mode == 'total'):
                sol = self._solar_vel
            if (self._mode == 'orbital') or (self._mode == 'total'):
                vel = tod.read_velocity()

            for det in tod.local_dets:

                pdata = np.copy(tod.read_pntg(detector=det, local_start=0, n=nsamp))
                flags, common = tod.read_flags(detector=det, local_start=0, n=nsamp)
                totflags = np.copy(flags)
                totflags |= common

                del flags
                del common

                pdata[(totflags != 0),:] = nullquat

                dipoletod = dipole(pdata, vel=vel, solar=sol, cmb=self._cmb, freq=self._freq)

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (tod.local_samples[1],))
                
                ref = tod.cache.reference(cachename)
                if self._subtract:
                    ref[:] -= dipoletod
                else:
                    ref[:] += dipoletod
                del ref

        return

