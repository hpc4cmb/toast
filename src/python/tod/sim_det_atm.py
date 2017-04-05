# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import numpy as np

import healpy as hp

from .. import qarray as qa

from .tod import TOD

from ..op import Operator

from ..ctoast import (MPI_Comm, atm_sim_alloc, atm_sim_free,
    atm_sim_simulate, atm_sim_observe)


class OpSimAtmosphere(Operator):
    """
    Operator which generates atmosphere timestreams.

    All processes collectively generate the atmospheric realization.  Then 
    each process passes through its local data and observes the atmosphere.
    This operator is only compatible with TOD objects that can return AZ/EL 
    pointing.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        lmin_center (float): Kolmogorov turbulence dissipation scale center.
        lmin_sigma (float): Kolmogorov turbulence dissipation scale sigma.
        lmax_center (float): Kolmogorov turbulence injection scale center.
        lmax_sigma (float): Kolmogorov turbulence injection scale sigma.
        zatm (float): atmosphere extent for temperature profile.
        zmax (float): atmosphere extent for water vapor integration.
        xstep (float): size of volume elements in X direction.
        ystep (float): size of volume elements in Y direction.
        zstep (float): size of volume elements in Z direction.
        nelem_sim_max (int): controls the size of the simulation slices.
        verbosity (int): more information is printed for values > 0.
        gangsize (int): size of the gangs that create slices.
        fnear (float): multiplier for the near field simulation.
        fixed_r (float): positive number for start of integration.

    """
    def __init__(self, out='atm', lmin_center=0.01, lmin_sigma=0.001, 
        lmax_center=10, lmax_sigma=10, zatm=40000.0, zmax=2000.0, xstep=100.0, 
        ystep=100.0, zstep=100.0, nelem_sim_max=1000, verbosity=0, gangsize=-1, 
        fnear=0.1):

        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._out = out
        self._lmin_center = lmin_center
        self._lmin_sigma = lmin_sigma
        self._lmax_center = lmax_center
        self._lmax_sigma = lmax_sigma
        self._zatm = zatm
        self._zmax = zmax
        self._xstep = xstep
        self._ystep = ystep
        self._zstep = zstep
        self._nelem_sim_max = nelem_sim_max
        self._verbosity = verbosity
        self._gangsize = gangsize
        self._fnear = fnear
        self._fixed_r = fixed_r


    def exec(self, data):
        """
        Generate atmosphere timestreams.

        This iterates over all observations and detectors and generates
        the atmosphere timestreams.

        Args:
            data (toast.Data): The distributed data.
        """

        for obs in data.obs:
            tod = obs['tod']
            comm = tod.mpicomm

            # FIXME: these functions don't exist.  It also probably makes more
            # sense to read a vector of wind properties and then find the
            # averages here, rather than have the TOD class provide a method
            # which returns the averages...

            # Call functions from TOD object to get the wind speed and
            # direction for this observation.

            w_center, w_sigma = tod.average_wind_speed()
            wdir_center, wdir_sigma = tod.average_water_vapor()

            # Call functions from TOD object to get the water vapor
            # distribution for this observation.

            z0_center, z0_sigma = tod.average_water_vapor()

            # Call functions from TOD object to get the ground temperature
            # for this observation.

            T0_center, T0_sigma = tod.average_ground_temperature()

            # Read the extent of the AZ/EL boresight pointing, and use that 
            # to compute the range of angles needed for simulating the slab.

            az_bore, el_bore = tod.read_boresight()
            azmin = np.min(az_bore) # plus some margin...
            azmax = np.max(az_bore) # plus some margin...
            elmin = np.min(el_bore) # plus some margin...
            elmax = np.max(el_bore) # plus some margin...

            # Get the timestamps

            times = tod.read_times()

            sim = atm_sim_alloc(azmin, azmax, elmin, elmax, times[0], times[-1],
                self._lmin_center, self._lmin_sigma, self._lmax_center, 
                self._lmax_sigma, w_center, w_sigma, wdir_center, wdir_sigma, 
                z0_center, z0_sigma, T0_center, T0_sigma, self._zatm, 
                self._zmax, self._xstep, self._ystep, self._zstep, 
                self._nelem_sim_max, self._verbosity, comm, self._gangsize, 
                self._fnear)

            atm_sim_simulate(sim, 0)

            nsamp = tod.local_samples[1]

            for det in tod.local_dets:

                # FIXME: We need to standardize the interface to get AZ/EL
                # detector pointing...
                az, el = tod.read_pntg_azel(detector=det)

                atmdata = np.zeros(nsamp, dtype=np.float64)

                atm_sim_observe(sim, times, az, el, atmdata, nsamp, self._fixed_r)

                # write to cache

                cachename = "{}_{}".format(self._out, det)

                ref = None
                if tod.cache.exists(cachename):
                    ref = tod.cache.reference(cachename)
                else:
                    ref = tod.cache.create(cachename, np.float64,
                                (tod.local_samples[1],))

                ref[local_offset:local_offset+chunk_samp] += atmdata
                del ref


        return

