# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from ..utils import Logger

from ..timing import function_timer, Timer
from ..rng import random

from ..op import Operator

from toast.mpi import MPI

import toast.qarray as qa

import healpy as hp


class OpSimScanSynchronousSignal(Operator):
    """Operator which generates scan-synchronous signal timestreams.
    
    Args:
        out (str): accumulate data to the cache with name
            <out>_<detector>.  If the named cache objects do not exist,
            then they are created.
        realization (int): if simulating multiple realizations, the
            realization index.
        component (int): the component index to use for this noise
            simulation.
        nside (int): ground map healpix resolution
        fwhm (int): ground map smoothing scale
        lmax (int): ground map expansion order
        scale (float): RMS of the ground signal fluctuations at el=45deg
        power (float): exponential for suppressing ground pickup at
             higher observing elevation
        path (string): path to a horizontal Healpix map to
            sample for the SSS *instead* of synthesizing Gaussian maps
        report_timing (bool):  Print out time taken to initialize,
             simulate and observe
    """

    def __init__(
        self,
        out="sss",
        realization=0,
        component=663056,
        nside=128,
        fwhm=10,
        lmax=256,
        scale=1e-3,
        power=-1,
        path=None,
        report_timing=False,
    ):
        # Call the parent class constructor
        super().__init__()

        self._out = out
        self._realization = realization
        self._component = component
        self._nside = nside
        self._lmax = lmax
        self._fwhm = fwhm
        self._scale = scale
        self._power = power
        self._path = path
        self._report_timing = report_timing
        return

    @function_timer
    def exec(self, data):
        """Generate timestreams.

        This iterates over all observations and detectors and generates
        the scan-synchronous signal timestreams.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            None

        """

        log = Logger.get()
        group = data.comm.group
        for obs in data.obs:
            try:
                obsname = obs["name"]
            except Exception:
                obsname = "observation"
            prefix = "{} : {} : ".format(group, obsname)
            tod = self._get_from_obs("tod", obs)
            comm = tod.mpicomm
            rank = 0
            if comm is not None:
                rank = comm.rank
            site = self._get_from_obs("site_id", obs)
            weather = self._get_from_obs("weather", obs)

            # Get the observation time span and initialize the weather
            # object if one is provided.
            times = tod.local_times()
            tmin = times[0]
            tmax = times[-1]
            tmin_tot = tmin
            tmax_tot = tmax
            if comm is not None:
                tmin_tot = comm.allreduce(tmin, op=MPI.MIN)
                tmax_tot = comm.allreduce(tmax, op=MPI.MAX)
            weather.set(site, self._realization, tmin_tot)

            key1, key2, counter1, counter2 = self._get_rng_keys(obs)

            if comm is not None:
                comm.Barrier()
            if rank == 0:
                log.info("{}Setting up SSS simulation".format(prefix))

            tmr = Timer()
            if self._report_timing:
                if comm is not None:
                    comm.Barrier()
                tmr.start()

            sssmap = self._simulate_sss(key1, key2, counter1, counter2, weather)

            self._observe_sss(sssmap, tod, comm, prefix)

            del sssmap

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                tmr.report(
                    "{}Simulated and observed scan-synchronous signal" "".format(prefix)
                )
        return

    def _get_from_obs(self, name, obs):
        """ Extract value for name from observation.

        If name is not defined in observation, raise an exception.

        """
        if name not in obs:
            raise RuntimeError(
                "Error simulating SSS: observation " 'does not define "{}"'.format(name)
            )
        return obs[name]

    def _get_rng_keys(self, obs):
        """
        The random number generator accepts a key and a counter,
        each made of two 64bit integers.
        Following tod_math.py we set
        key1 = realization * 2^32 + telescope * 2^16 + component
        key2 = obsindx * 2^32
        counter1 = hierarchical cone counter
        counter2 = sample in stream (incremented internally in the atm code)
        """
        telescope = self._get_from_obs("telescope_id", obs)
        site = self._get_from_obs("site_id", obs)
        obsindx = self._get_from_obs("id", obs)
        key1 = self._realization * 2 ** 32 + telescope * 2 ** 16 + self._component
        key2 = site * 2 ** 16 + obsindx
        counter1 = 0
        counter2 = 0
        return key1, key2, counter1, counter2

    def _simulate_sss(self, key1, key2, counter1, counter2, weather):
        """
        Create a map of the ground signal to observe with all detectors
        """
        # FIXME: we could store the map in node-shared memory
        #
        # Surface temperature is made available but not used yet
        # to scale the SSS
        temperature = weather.surface_temperature
        if self._path:
            sssmap = hp.read_map(self._path)
        else:
            npix = 12 * self._nside ** 2
            sssmap = random(
                npix, key=(key1, key2), counter=(counter1, counter2), sampler="gaussian"
            )
            sssmap = np.array(sssmap, dtype=np.float)
            sssmap = hp.smoothing(sssmap, fwhm=np.radians(self._fwhm), lmax=self._lmax)
            sssmap /= np.std(sssmap)
            lon, lat = hp.pix2ang(self._nside, np.arange(npix, dtype=np.int), lonlat=True)
            scale = self._scale * (np.abs(lat) / 90 + 0.5) ** self._power
            sssmap *= scale
        return sssmap

    @function_timer
    def _observe_sss(self, sssmap, tod, comm, prefix):
        """
        Use healpy bilinear interpolation to observe the ground signal map
        """
        log = Logger.get()
        rank = 0
        if comm is not None:
            rank = comm.rank
        tmr = Timer()
        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            tmr.start()

        nsamp = tod.local_samples[1]

        if rank == 0:
            log.info("{}Observing the scan-synchronous signal".format(prefix))

        for det in tod.local_dets:
            # Cache the output signal
            cachename = "{}_{}".format(self._out, det)
            if tod.cache.exists(cachename):
                ref = tod.cache.reference(cachename)
            else:
                ref = tod.cache.create(cachename, np.float64, (nsamp,))

            try:
                # Some TOD classes provide a shortcut to Az/El
                az, el = tod.read_azel(detector=det)
                phi = 2 * np.pi - az
                theta = np.pi / 2 - el
            except Exception as e:
                azelquat = tod.read_pntg(detector=det, azel=True)
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, phi = qa.to_position(azelquat)
                # Azimuth is measured in the opposite direction
                # than longitude
                # az = 2 * np.pi - phi
                # el = np.pi / 2 - theta

            ref[:] += hp.get_interp_val(sssmap, theta, phi)

            del ref

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                tmr.report("{}OpSimSSS: Observe signal".format(prefix))
        return
