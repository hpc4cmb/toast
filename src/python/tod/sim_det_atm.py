# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, MPI_Comm

import numpy as np

from scipy.constants import degree, arcmin

import healpy as hp

from .. import qarray as qa

from .tod import TOD

from ..op import Operator

from ..ctoast import (atm_sim_alloc, atm_sim_free,
    atm_sim_simulate, atm_sim_observe)


# FIXME:  For now, we use a fixed distribution of the "weather" (wind speed,
# temperature, etc) for all CESs.  Eventually we plan to have 2 TOD base 
# classes (TODSatellite and TODGround).  The TODGround class and descendants
# will have methods to return the site conditions for a given CES.  Once that
# is in place, then we should have this operator call those methods for each
# observation to get the actual weather conditions.  Simulating a realization
# of the weather could then move to a class that derives from TODGround.


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
        w_center (float):  central value of the wind speed distribution.
        w_sigma (float):  sigma of the wind speed distribution.
        wdir_center (float):  central value of the wind direction distribution.
        wdir_sigma (float):  sigma of the wind direction distribution.
        z0_center (float):  central value of the water vapor distribution.
        z0_sigma (float):  sigma of the water vapor distribution.
        T0_center (float):  central value of the temperature distribution.
        T0_sigma (float):  sigma of the temperature distribution.

    """
    def __init__(self, out='atm', lmin_center=0.01, lmin_sigma=0.001,
        lmax_center=10, lmax_sigma=10, zatm=40000.0, zmax=2000.0, xstep=100.0,
        ystep=100.0, zstep=100.0, nelem_sim_max=10000, verbosity=0, gangsize=-1,
        fnear=0.1, fixed_r=-1, w_center=10, w_sigma=1, wdir_center=0, wdir_sigma=100,
        z0_center=2000, z0_sigma=0, T0_center=280, T0_sigma=10):

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

        # FIXME: eventually these will come from the TOD object
        # for each obs...
        self._w_center = w_center
        self._w_sigma = w_sigma
        self._wdir_center = wdir_center
        self._wdir_sigma = wdir_sigma
        self._z0_center = z0_center
        self._z0_sigma = z0_sigma
        self._T0_center = T0_center
        self._T0_sigma = T0_sigma


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

            # FIXME: This is where (eventually) we should get the wind speed,
            # wind direction, temperature, and water vapor from the tod
            # object, which will be derived from the new TODGround base class.

            # Read the extent of the AZ/EL boresight pointing, and use that 
            # to compute the range of angles needed for simulating the slab.

            (min_az_bore, max_az_bore, min_el_bore, max_el_bore) = tod.scan_range
            #print("boresight scan range = {}, {}, {}, {}".format(
            #min_az_bore, max_az_bore, min_el_bore, max_el_bore))

            # Go through all detectors and compute the maximum angular extent
            # of the focalplane from the boresight.  Add a tiny margin, since
            # the atmosphere simulation already adds some margin.

            # FIXME: the TOD class should really provide a method to return
            # the detector quaternions relative to the boresight.  For now, we
            # jump through hoops by getting one sample of the pointing.

            zaxis = np.array([0.0, 0.0, 1.0])
            detdir = []
            detmeanx = 0.0
            detmeany = 0.0
            for det in tod.local_dets:
                dquat = tod.read_pntg(
                    detector=det, local_start=0, n=1, azel=True)
                dvec = qa.rotate(dquat, zaxis).flatten()
                detmeanx += dvec[0]
                detmeany += dvec[1]
                detdir.append(dvec)
            detmeanx /= len(detdir)
            detmeany /= len(detdir)

            #print("detmeanx = {}, detmeany = {}".format(detmeanx, detmeany))

            detrad = [np.sqrt((detdir[t][0] - detmeanx)**2
                              + (detdir[t][1] - detmeany)**2)
                      for t in range(len(detdir))]

            fp_radius = np.arcsin(np.max(detrad)) + 1 * arcmin

            #print("fp_radius = {}", fp_radius)

            azmin = min_az_bore - fp_radius
            azmax = max_az_bore + fp_radius
            elmin = min_el_bore - fp_radius
            elmax = max_el_bore + fp_radius

            #print("patch = {}, {}, {}, {}".format(azmin, azmax, elmin, elmax))

            # Get the timestamps

            times = tod.read_times()

            sim = atm_sim_alloc(azmin, azmax, elmin, elmax, times[0],
                times[-1], self._lmin_center, self._lmin_sigma,
                self._lmax_center, self._lmax_sigma, self._w_center,
                self._w_sigma, self._wdir_center, self._wdir_sigma,
                self._z0_center, self._z0_sigma, self._T0_center,
                self._T0_sigma, self._zatm, self._zmax, self._xstep,
                self._ystep, self._zstep, self._nelem_sim_max,
                self._verbosity, comm, self._gangsize, self._fnear)

            atm_sim_simulate(sim, 0)

            nsamp = tod.local_samples[1]

            for det in tod.local_dets:

                azelquat = tod.read_pntg(detector=det, azel=True)

                atmdata = np.zeros(nsamp, dtype=np.float64)

                # Convert Az/El quaternion of the detector back into angles for
                # the simulation.

                theta, phi, pa = qa.to_angles(azelquat)
                az = phi
                el = np.pi/2 - theta

                azmin_det = np.amin(az)
                azmax_det = np.amax(az)
                elmin_det = np.amin(el)
                elmax_det = np.amax(el)
                if (azmin_det < azmin or azmax < azmax_det or
                    elmin_det < elmin or elmax < elmax_det):
                    raise RuntimeError(
                        'Detector Az/El: [{:.5f}, {:.5f}], [{:.5f}, {:.5f}] '
                        'is not contained in [{:.5f}, {:.5f}], [{:.5f} {:.5f}]'
                        ''.format(
                            azmin_det, azmax_det, elmin_det, elmax_det,
                            azmin, azmax, elmin, elmax))

                # Integrate detector signal

                atm_sim_observe(
                    sim, times, az, el, atmdata, nsamp, self._fixed_r)

                # write to cache

                cachename = "{}_{}".format(self._out, det)

                ref = None
                if tod.cache.exists(cachename):
                    ref = tod.cache.reference(cachename)
                else:
                    ref = tod.cache.create(cachename, np.float64, (nsamp,))

                ref += atmdata
                del ref

        return
