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
        realization (int): if simulating multiple realizations, the realization
            index.
        component (int): the component index to use for this noise simulation.
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
        w_center (float):  central value of the wind speed distribution.
        w_sigma (float):  sigma of the wind speed distribution.
        wdir_center (float):  central value of the wind direction distribution.
        wdir_sigma (float):  sigma of the wind direction distribution.
        z0_center (float):  central value of the water vapor distribution.
        z0_sigma (float):  sigma of the water vapor distribution.
        T0_center (float):  central value of the temperature distribution.
        T0_sigma (float):  sigma of the temperature distribution.
        fp_radius (float):  focal plane radius in degrees.

    """
    def __init__(
            self, out='atm', realization=0, component=123456,
            lmin_center=0.01, lmin_sigma=0.001,
            lmax_center=10, lmax_sigma=10, zatm=40000.0, zmax=2000.0,
            xstep=100.0, ystep=100.0, zstep=100.0, nelem_sim_max=10000,
            verbosity=0, gangsize=-1,
            fnear=0.1, w_center=10, w_sigma=1, wdir_center=0, wdir_sigma=100,
            z0_center=2000, z0_sigma=0, T0_center=280, T0_sigma=10,
            fp_radius=1):

        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._out = out
        self._realization = realization
        self._component = component
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
        self._fp_radius = fp_radius

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
            try:
                obsname = obs['name']
            except:
                obsname = 'observation'

            obsindx = 0
            if 'id' in obs:
                obsindx = obs['id']
            else:
                print("Warning: observation ID is not set, using zero!")

            telescope = 0
            if 'telescope' in obs:
                telescope = obs['telescope']

            """
            The random number generator accepts a key and a counter,
            each made of two 64bit integers.
            Following tod_math.py we set
            key1 = realization * 2^32 + telescope * 2^16 + component
            key2 = obsindx * 2^32
            counter1 = currently unused (0)
            counter2 = sample in stream (incremented internally in the atm code)
            """
            key1 = self._realization * 2**32 + telescope * 2**16 \
                   + self._component
            key2 = obsindx
            counter1 = 0
            counter2 = 0

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

            if self._verbosity:

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

                fp_radius = np.arcsin(np.max(detrad)) / degree
                if fp_radius > self._fp_radius:
                    raise RuntimeError(
                        'Detectors in the TOD span {:.2} degrees, but the '
                        'atmosphere simulation only covers {:.2f} degrees'
                        ''.format(fp_radius, self._fp_radius))

            # Use the fixed focal plane radius so that changing the actual
            # set of detectors will not affect the simulated atmosphere.

            fp_radius = self._fp_radius * degree

            azmin = min_az_bore - fp_radius
            azmax = max_az_bore + fp_radius
            elmin = min_el_bore - fp_radius
            elmax = max_el_bore + fp_radius

            azmin = comm.allreduce(azmin, op=MPI.MIN)
            azmax = comm.allreduce(azmax, op=MPI.MAX)
            elmin = comm.allreduce(elmin, op=MPI.MIN)
            elmax = comm.allreduce(elmax, op=MPI.MAX)

            if elmin < 0 or elmax > np.pi/2:
                raise RuntimeError(
                    'Error in CES elevation: elmin = {:.2f}, elmax = {:.2f}'
                    ''.format(elmin, elmax))

            #print("patch = {}, {}, {}, {}".format(azmin, azmax, elmin, elmax))

            # Get the timestamps

            times = tod.read_times()

            tmin = times[0]
            tmax = times[-1]
            tmin = comm.allreduce(tmin, op=MPI.MIN)
            tmax = comm.allreduce(tmax, op=MPI.MAX)

            sim = atm_sim_alloc(
                azmin, azmax, elmin, elmax, tmin, tmax,
                self._lmin_center, self._lmin_sigma,
                self._lmax_center, self._lmax_sigma, self._w_center,
                self._w_sigma, self._wdir_center, self._wdir_sigma,
                self._z0_center, self._z0_sigma, self._T0_center,
                self._T0_sigma, self._zatm, self._zmax, self._xstep,
                self._ystep, self._zstep, self._nelem_sim_max,
                self._verbosity, comm, self._gangsize, self._fnear,
                key1, key2, counter1, counter2)

            atm_sim_simulate(sim, 0)

            if self._verbosity > 0:
                # Create snapshots of the atmosphere
                import matplotlib.pyplot as plt
                import sys
                azelstep = .01 * degree
                azgrid = np.linspace( azmin, azmax, (azmax-azmin)//azelstep+1 )
                elgrid = np.linspace( elmin, elmax, (elmax-elmin)//azelstep+1 )
                AZ, EL = np.meshgrid( azgrid, elgrid )
                nn = AZ.size
                az = AZ.ravel()
                el = EL.ravel()
                atmdata = np.zeros(nn, dtype=np.float64)
                atmtimes = np.zeros(nn, dtype=np.float64)

                rank = comm.rank
                ntask = comm.size
                r = 0
                t = 0
                #for i, r in enumerate(np.linspace(1, 5000, 100)):
                for i, t in enumerate(np.linspace(tmin, tmax, 100)):
                    if i % ntask != rank:
                        continue
                    atm_sim_observe(sim, atmtimes+t, az, el, atmdata, nn, r)
                    atmdata2d = atmdata.reshape(AZ.shape)
                    plt.figure()
                    plt.imshow(atmdata2d, interpolation='nearest',
                               origin='lower', extent=np.array(
                                   [azmin, azmax, elmin, elmax])/degree,
                               cmap=plt.get_cmap('Blues'))
                    plt.colorbar()
                    ax = plt.gca()
                    ax.set_title('t = {:15.1f} s, r = {:15.1f} m'.format(t, r))
                    ax.set_xlabel('az [deg]')
                    ax.set_ylabel('el [deg]')
                    ax.set_yticks([elmin/degree, elmax/degree])
                    plt.savefig('atm_{}_t_{:04}_r_{:04}.png'.format(
                        obsname, int(t), int(r)))
                    plt.close()

            nsamp = tod.local_samples[1]

            for det in tod.local_dets:

                azelquat = tod.read_pntg(detector=det, azel=True)

                atmdata = np.zeros(nsamp, dtype=np.float64)

                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.

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
                    sim, times, az, el, atmdata, nsamp, 0)

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
