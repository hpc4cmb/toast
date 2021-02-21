# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np

import h5py

from ..utils import Environment, Logger

from ..timing import Timer, function_timer, GlobalTimers

from ..rng import random as toast_rng

from ..mpi import MPI, MPIShared

from .._libtoast import (
    atm_sim_compute_slice,
    atm_sim_observe,
    atm_sim_compress_flag_hits_rank,
    atm_sim_compress_flag_extend_rank,
    atm_sim_kolmogorov_init_rank,
)

available_utils = None
if available_utils is None:
    available_utils = True
    try:
        from .._libtoast import (
            atm_absorption_coefficient,
            atm_absorption_coefficient_vec,
            atm_atmospheric_loading,
            atm_atmospheric_loading_vec,
        )
    except ImportError:
        available_utils = False


class AtmSim(object):
    """Class representing a single atmosphere simulation.

    This simulation consists of a particular realization of a "slab" of the
    atmosphere that moves with a constant wind speed and can be observed by
    individual detectors.

    Args:
        azmin (float):  The minimum of the azimuth range.
        azmax (float):  The maximum of the azimuth range.
        elmin (float):  The minimum of the elevation range.
        elmax (float):  The maximum of the elevation range.
        tmin (float):  The minimum of the time range.
        tmax (float):  The maximum of the time range.
        lmin_center (float):  Center point of the distribution of the dissipation
            scale of the Kolmogorov turbulence.
        lmin_sigma (float):  Width of the distribution of the dissipation
            scale of the Kolmogorov turbulence.
        lmax_center (float):  Center point of the distribution of the injection
            scale of the Kolmogorov turbulence.
        lmax_sigma (float):  Width of the distribution of the injection
            scale of the Kolmogorov turbulence.
        w_center (float):  Center point of the distribution of wind speed (m/s).
        w_sigma (float):  Width of the distribution of wind speed.
        wdir_center (float):  Center point of the distribution of wind direction
            (radians).
        wdir_sigma (float):  Width of the distribution of wind direction.
        z0_center (float):  Center point of the distribution of the water vapor (m).
        z0_sigma (float):  Width of the distribution of the water vapor.
        T0_center (float):  Center point of the distribution of ground temperature
            (Kelvin).
        T0_sigma (float):  Width of the distribution of ground temperature.
        zatm (float):  Atmosphere extent for temperature profile.
        zmax (float):  Water vaport extent for integration.
        xstep (float):  Size of volume element in the X direction.
        ystep (float):  Size of volume element in the Y direction.
        zstep (float):  Size of volume element in the Z direction.
        nelem_sim_max (int):  Size of the simulation slices.
        comm (mpi4py.MPI.Comm):  The MPI communicator or None.
        key1 (uint64):  Streamed RNG key 1.
        key2 (uint64):  Streamed RNG key 2.
        counterval1 (uint64):  Streamed RNG counter 1.
        counterval2 (uint64):  Streamed RNG counter 2.
        cachedir (str):  The location of the cached simulation.
        rmin (float):  Minimum line of sight observing distance.
        rmax (float):  Maximum line of sight observing distance.
        write_debug (bool): If true, write out intermediate text files for debugging.

    """

    def __init__(
        self,
        azmin,
        azmax,
        elmin,
        elmax,
        tmin,
        tmax,
        lmin_center,
        lmin_sigma,
        lmax_center,
        lmax_sigma,
        w_center,
        w_sigma,
        wdir_center,
        wdir_sigma,
        z0_center,
        z0_sigma,
        T0_center,
        T0_sigma,
        zatm,
        zmax,
        xstep,
        ystep,
        zstep,
        nelem_sim_max,
        comm,
        key1,
        key2,
        counterval1,
        counterval2,
        cachedir,
        rmin,
        rmax,
        write_debug=False,
    ):
        self._azmin = azmin
        self._azmax = azmax
        self._elmin = elmin
        self._elmax = elmax
        self._tmin = tmin
        self._tmax = tmax
        self._lmin_center = lmin_center
        self._lmin_sigma = lmin_sigma
        self._lmax_center = lmax_center
        self._lmax_sigma = lmax_sigma
        self._w_center = w_center
        self._w_sigma = w_sigma
        self._wdir_center = wdir_center
        self._wdir_sigma = wdir_sigma
        self._z0_center = z0_center
        self._z0_sigma = z0_sigma
        self._T0_center = T0_center
        self._T0_sigma = T0_sigma
        self._zatm = zatm
        self._zmax = zmax
        self._xstep = xstep
        self._ystep = ystep
        self._zstep = zstep
        self._nelem_sim_max = nelem_sim_max
        self._comm = comm
        self._key1 = key1
        self._key2 = key2
        self._counter1start = counterval1
        self._counter2start = counterval2
        self._cachedir = cachedir
        self._rmin = rmin
        self._rmax = rmax
        self._write_debug = write_debug

        self._counter1 = self._counter1start
        self._counter2 = self._counter2start

        self._corrlim = 1e-3

        self._ntask = 1
        self._rank = 0
        if self._comm is not None:
            self._ntask = self._comm.size
            self._rank = self._comm.rank

        env = Environment.get()
        self._nthread = env.max_threads()

        log = Logger.get()

        if self._rank == 0:
            log.debug(
                "AtmSim constructed with {} processes, {} threads per process".format(
                    self._ntask, self._nthread
                )
            )

        if azmin >= azmax:
            raise RuntimeError("AtmSim: azmin >= azmax")

        if elmin < 0:
            raise RuntimeError("AtmSim: elmin < 0")

        if elmax > 0.5 * np.pi:
            raise RuntimeError("AtmSim: elmax > pi/2")

        if elmin > elmax:
            raise RuntimeError("AtmSim: elmin >= elmax")

        if tmin >= tmax:
            raise RuntimeError("AtmSim: tmin >= tmax")

        if lmin_center > lmax_center:
            raise RuntimeError("AtmSim: lmin_center >= lmax_center")

        self._delta_az = self._azmax - self._azmin
        self._delta_el = self._elmax - self._elmin
        self._delta_t = self._tmax - self._tmin

        self._az0 = self._azmin + self._delta_az / 2
        self._el0 = self._elmin + self._delta_el / 2
        self._sinel0 = np.sin(self._el0)
        self._cosel0 = np.cos(self._el0)

        self._xxstep = self._xstep * self._cosel0 - self._zstep * self._sinel0
        self._yystep = self._ystep
        self._zzstep = self._xstep * self._sinel0 + self._zstep * self._cosel0

        if self._rank == 0:
            msg = "\nInput parameters:\n"
            msg += "             az = [{} - {}] ({} degrees)\n".format(
                np.degrees(self._azmin),
                np.degrees(self._azmax),
                np.degrees(self._delta_az),
            )
            msg += "             el = [{} - {}] ({} degrees)\n".format(
                np.degrees(self._elmin),
                np.degrees(self._elmax),
                np.degrees(self._delta_el),
            )
            msg += "              t = [{} - {}] ({} s)\n".format(
                self._tmin, self._tmax, self._delta_t
            )
            msg += "           lmin = {} +- {} m\n".format(
                self._lmin_center, self._lmin_sigma
            )
            msg += "           lmax = {} +- {} m\n".format(
                self._lmax_center, self._lmax_sigma
            )
            msg += "              w = {} +- {} m/s\n".format(
                self._w_center, self._w_sigma
            )
            msg += "           wdir = {} +- {} degrees\n".format(
                np.degrees(self._wdir_center), np.degrees(self._wdir_sigma)
            )
            msg += "             z0 = {} +- {} m\n".format(
                self._z0_center, self._z0_sigma
            )
            msg += "             T0 = {} +- {} K\n".format(
                self._T0_center, self._T0_sigma
            )
            msg += "           zatm = {} m\n".format(self._zatm)
            msg += "           zmax = {} m\n".format(self._zmax)
            msg += "       scan frame:\n"
            msg += "          xstep = {} m\n".format(self._xstep)
            msg += "          ystep = {} m\n".format(self._ystep)
            msg += "          zstep = {} m\n".format(self._zstep)
            msg += "  nelem_sim_max = {}\n".format(self._nelem_sim_max)
            msg += "        corrlim = {}\n".format(self._corrlim)
            msg += "           rmin = {} m\n".format(self._rmin)
            msg += "           rmax = {} m\n".format(self._rmax)
            log.debug(msg)

        self._compressed_index = None
        self._full_index = None
        self._realization = None
        self._cached = False
        self._nelem = None

    @function_timer
    def simulate(self, use_cache=False, smooth=False):
        """Perform the simulation.

        Args:
            use_cache (bool):  If True, load / save from / to cachedir.
            smooth (bool):  If True, apply smoothing.

        Returns:
            (int):  A status value (zero == good).

        """
        if use_cache:
            if self._cachedir is None:
                raise RuntimeError("Cannot use the cache if cachedir is not set")
            self.load_realization()

        if self._cached:
            return 0

        log = Logger.get()

        self.draw()

        self.get_volume()

        self.compress_volume()

        if self._rank == 0:
            log.debug("Resizing realization to {}".format(self._nelem))

        timer = Timer()
        timer.start()

        # This memory is already zeroed on construction.
        self._realization = MPIShared((self._nelem,), np.float64, self._comm)

        timer.stop()
        if self._rank == 0:
            log.debug(
                "Allocated shared realization buffer in {} s".format(timer.seconds())
            )
        timer.clear()
        timer.start()

        ind_start = 0
        ind_stop = 0
        slice = 0

        # Simulate the atmosphere in independent slices, each slice
        # assigned to one process.

        slice_starts = list()
        slice_stops = list()

        # Set a per-process global timer to check for load imbalance
        gt = GlobalTimers.get()
        gt.start("AtmSim compute slices")

        while True:
            ind_start, ind_stop = self._get_slice(ind_start, ind_stop)
            slice_starts.append(ind_start)
            slice_stops.append(ind_stop)

            if slice % self._ntask == self._rank:
                atm_sim_compute_slice(
                    ind_start,
                    ind_stop,
                    self._rmin_kolmo,
                    self._rmax_kolmo,
                    self._kolmo_x,
                    self._kolmo_y,
                    self._rcorr,
                    self._xstart,
                    self._ystart,
                    self._zstart,
                    self._xstep,
                    self._ystep,
                    self._zstep,
                    self._xstride,
                    self._ystride,
                    self._zstride,
                    self._z0,
                    self._cosel0,
                    self._sinel0,
                    self._full_index.data,
                    smooth,
                    self._xxstep,
                    self._zzstep,
                    self._rank,
                    self._key1,
                    self._key2,
                    self._counter1,
                    self._counter2,
                    self._realization.data,
                )

            # Advance the RNG counter on all processes
            self._counter2 += ind_stop - ind_start

            if ind_stop == self._nelem:
                break

            slice += 1

        slice_starts = np.array(slice_starts, dtype=np.int32)
        slice_stops = np.array(slice_stops, dtype=np.int32)

        gt.stop("AtmSim compute slices")
        if self._comm is not None:
            self._comm.barrier()

        timer.stop()
        if self._rank == 0:
            log.debug("Simulated all slices in {} s".format(timer.seconds()))
        timer.clear()
        timer.start()

        # Gather the slices.  Every process has written their assigned slices to their
        # node-local shared memory buffer.  However, these slices do not exist on the
        # other nodes.  For each slice, the assigned process copies the slice to a
        # buffer and then uses the MPIShared.set() method to replicate to shared
        # memory on all nodes.

        for slice in range(len(slice_starts)):
            ind_start = slice_starts[slice]
            ind_stop = slice_stops[slice]
            nind = ind_stop - ind_start
            root = slice % self._ntask

            tempvec = None
            if self._rank == root:
                # This process has the slice, copy to a buffer
                tempvec = np.array(self._realization.data[ind_start:ind_stop])

            # Set the slice across all nodes
            self._realization.set(tempvec, (ind_start,), fromrank=root)

        timer.stop()
        if self._rank == 0:
            log.debug("Distributed slices to all nodes in {} s".format(timer.seconds()))
        timer.clear()
        timer.start()

        # self.smooth()

        self._cached = True
        if use_cache:
            self.save_realization()

        timer.stop()
        if use_cache and self._rank == 0:
            log.debug("Saved realization in {} s".format(timer.seconds()))

        return 0

    @function_timer
    def observe(self, times, az, el, tod, fixed_r=-1):
        """Observe the atmosphere with a detector.

        The timestamps and Azimuth / Elevation pointing are provided.  The TOD
        buffer is filled with the integrated atmospheric signal.

        For each sample, integrate along the line of sight by summing the
        atmosphere values. See Church (1995) Section 2.2, first equation.
        We omit the optical depth factor which is close to unity.

        Args:
            times (array_like):  Detector timestamps.
            az (array like):  Azimuth values.
            el (array_like):  Elevation values.
            tod (array_like):  The output buffer to fill.
            fixed_r (float):  If greater than zero, use this single radial value.

        Returns:
            (int):  A status value (zero == good).

        """
        if not self._cached:
            raise RuntimeError("There is no cached observation to observe")

        log = Logger.get()
        timer = Timer()
        timer.start()

        nsamp = len(times)

        status = 0

        status = atm_sim_observe(
            times,
            az,
            el,
            tod,
            self._T0,
            self._azmin,
            self._azmax,
            self._elmin,
            self._elmax,
            self._tmin,
            self._tmax,
            self._rmin,
            self._rmax,
            fixed_r,
            self._zatm,
            self._zmax,
            self._wx,
            self._wy,
            self._wz,
            self._xstep,
            self._ystep,
            self._zstep,
            self._xstart,
            self._delta_x,
            self._ystart,
            self._delta_y,
            self._zstart,
            self._delta_z,
            self._maxdist,
            self._nx,
            self._ny,
            self._nz,
            self._xstride,
            self._ystride,
            self._zstride,
            self._compressed_index.data,
            self._full_index.data,
            self._realization.data,
        )

        timer.stop()

        if self._rank == 0:
            if fixed_r > 0:
                log.debug(
                    "Observed {} samples at r = {} in {} s".format(
                        nsamp, fixed_r, timer.seconds()
                    )
                )
            else:
                log.debug("Observed {} samples in {} s".format(nsamp, timer.seconds()))

        if status != 0:
            log.error("Observing {} samples failed with error {}".format(nsamp, status))
        return status

    def _get_slice(self, ind_start, ind_stop):
        """Identify a manageable slice of compressed indices to simulate next."""
        log = Logger.get()

        # Move element counter to the end of the most recent simulated slice
        ind_start = ind_stop

        # TK: The original code multiplied a double and a long and implicitly cast
        # back to a long.  implementing that here, but should verify that it is
        # correct.
        xstrideinv = 1.0 / self._xstride
        ix_start = int(self._full_index[ind_start] * xstrideinv)
        ix1 = ix_start
        ix2 = 0

        while True:
            # Advance element counter by one layer of elements
            ix2 = ix1
            while ix1 == ix2:
                ind_stop += 1
                if ind_stop == self._nelem:
                    break
                ix2 = int(self._full_index[ind_stop] * xstrideinv)
            # Check if there are no more elements
            if ind_stop == self._nelem:
                break

            # Check if we have enough to meet the minimum number of elements
            if ind_stop - ind_start >= self._nelem_sim_max:
                break

            # Check if we have enough layers
            # nlayer_sim_max = 10
            # if ix2 - ix_start >= nlayer_sim_max:
            #     break
            ix1 = ix2

        if self._rank == 0:
            log.debug(
                "X-slice: {} -- {} ({} {} m layers) m out of {} m indices {} -- {} ({}) out of {}".format(
                    ix_start * self._xstep,
                    ix2 * self._xstep,
                    ix2 - ix_start,
                    self._xstep,
                    self._nx * self._xstep,
                    ind_start,
                    ind_stop,
                    ind_stop - ind_start,
                    self._nelem,
                )
            )
        return (ind_start, ind_stop)

    def draw(self):
        # Draw gaussian variates to use in drawing the simulation
        # parameters.
        log = Logger.get()

        nrand = 10000

        randn = np.array(
            toast_rng(
                nrand,
                key=(self._key1, self._key2),
                counter=(self._counter1, self._counter2),
                sampler="gaussian",
                threads=False,
            ),
            dtype=np.float64,
        )

        self._counter2 += nrand

        irand = 0

        # The random numbers above are identical on all processes, so every process
        # just computes the same parameters locally.

        self._lmin = 0
        self._lmax = 0
        self._w = -1
        self._wdir = 0
        self._z0 = 0
        self._T0 = 0

        while self._lmin >= self._lmax:
            self._lmin = 0
            self._lmax = 0
            while (self._lmin <= 0) and (irand < nrand - 1):
                self._lmin = self._lmin_center + randn[irand] * self._lmin_sigma
                irand += 1
            while (self._lmax <= 0) and (irand < nrand - 1):
                self._lmax = self._lmax_center + randn[irand] * self._lmax_sigma
                irand += 1

        while (self._w < 0) and (irand < nrand - 1):
            self._w = self._w_center + randn[irand] * self._w_sigma
            irand += 1

        self._wdir = np.fmod(self._wdir_center + randn[irand] * self._wdir_sigma, np.pi)
        irand += 1

        while (self._z0 <= 0) and (irand < nrand):
            self._z0 = self._z0_center + randn[irand] * self._z0_sigma
            irand += 1
        while (self._T0 <= 0) and (irand < nrand):
            self._T0 = self._T0_center + randn[irand] * self._T0_sigma
            irand += 1

        if irand == nrand:
            raise RuntimeError(
                "Failed to draw parameters so satisfy boundary conditions"
            )

        # Wind is parallel to surface. Rotate to a frame where the scan
        # is across the X-axis.

        # self._w *= 0.5  # DEBUG
        # self._wdir += np.pi  # DEBUG

        eastward_wind = self._w * np.cos(self._wdir)
        northward_wind = self._w * np.sin(self._wdir)

        angle = self._az0 - 0.5 * np.pi
        wx_h = eastward_wind * np.cos(angle) - northward_wind * np.sin(angle)
        self._wy = eastward_wind * np.sin(angle) + northward_wind * np.cos(angle)

        self._wx = wx_h * self._cosel0
        self._wz = -wx_h * self._sinel0

        # Inverse the wind direction so we can apply it to the
        # telescope position

        self._wx *= -1
        self._wy *= -1
        self._wz *= -1

        if self._rank == 0:
            msg = "\nAtmospheric realization parameters:\n"
            msg += "           lmin = {} m\n".format(self._lmin)
            msg += "           lmax = {} m\n".format(self._lmax)
            msg += "              w = {} m/s\n".format(self._w)
            msg += "  eastward wind = {} m/s\n".format(eastward_wind)
            msg += " northward wind = {} m/s\n".format(northward_wind)
            msg += "            az0 = {} degrees\n".format(np.degrees(self._az0))
            msg += "            el0 = {} degrees\n".format(np.degrees(self._el0))
            msg += "             wx = {} m/s\n".format(self._wx)
            msg += "             wy = {} m/s\n".format(self._wy)
            msg += "             wz = {} m/s\n".format(self._wz)
            msg += "           wdir = {} degrees\n".format(np.degrees(self._wdir))
            msg += "             z0 = {} m\n".format(self._z0)
            msg += "             T0 = {} K\n".format(self._T0)
            log.debug(msg)
        return

    def get_volume(self):
        """Compute the volume."""
        log = Logger.get()

        # Trim zmax if rmax sets a more stringent limit
        zmax_test = self._rmax * np.sin(self._elmax)
        if self._zmax > zmax_test:
            self._zmax = zmax_test

        # Horizontal volume

        delta_z_h = self._zmax

        # Maximum distance observed through the simulated volume

        self._maxdist = delta_z_h / self._sinel0

        # Volume length
        delta_x_h = self._maxdist * np.cos(self._elmin)

        # double x, y, z, xx, zz, r, rproj, z_min, z_max;

        r = self._maxdist

        z = r * np.sin(self._elmin)
        rproj = r * np.cos(self._elmin)
        x = rproj * np.cos(0)
        z_min = -x * self._sinel0 + z * self._cosel0

        z = r * np.sin(self._elmax)
        rproj = r * np.cos(self._elmax)
        x = rproj * np.cos(self._delta_az / 2)
        z_max = -x * self._sinel0 + z * self._cosel0

        # Cone width
        rproj = r * np.cos(self._elmin)
        delta_y_cone = 0
        if self._delta_az > np.pi:
            delta_y_cone = 2 * rproj
        else:
            delta_y_cone = 2 * rproj * np.cos(0.5 * (np.pi - self._delta_az))

        # Cone height
        delta_z_cone = z_max - z_min

        # Rotate to observation plane
        self._delta_x = self._maxdist
        self._delta_z = delta_z_cone

        # Wind effects

        wdx = np.absolute(self._wx) * self._delta_t
        wdy = np.absolute(self._wy) * self._delta_t
        wdz = np.absolute(self._wz) * self._delta_t
        self._delta_x += wdx
        self._delta_y = delta_y_cone + wdy
        self._delta_z += wdz

        # Margin for interpolation

        self._delta_x += self._xstep
        self._delta_y += 2 * self._ystep
        self._delta_z += 2 * self._zstep

        # Translate the volume to allow for wind.  Telescope sits
        # at (0, 0, 0) at t=0

        if self._wx < 0:
            self._xstart = -wdx
        else:
            self._xstart = 0

        if self._wy < 0:
            self._ystart = -0.5 * delta_y_cone - wdy - self._ystep
        else:
            self._ystart = -0.5 * delta_y_cone - self._ystep

        if self._wz < 0:
            self._zstart = z_min - wdz - self._zstep
        else:
            self._zstart = z_min - self._zstep

        # Grid points

        # TK: the original code had an implicity conversion from double to long,
        # which is replicated here.  Is this the desired behavior?
        self._nx = int(self._delta_x / self._xstep) + 1
        self._ny = int(self._delta_y / self._ystep) + 1
        self._nz = int(self._delta_z / self._zstep) + 1
        self._nn = self._nx * self._ny * self._nz

        # 1D storage of the 3D volume elements

        self._zstride = 1
        self._ystride = self._zstride * self._nz
        self._xstride = self._ystride * self._ny

        if self._rank == 0:
            msg = "Simulation volume:\n"
            msg += "       delta_x = {} m\n".format(self._delta_x)
            msg += "       delta_y = {} m\n".format(self._delta_y)
            msg += "       delta_z = {} m\n".format(self._delta_z)
            msg += "Observation cone along the X-axis:\n"
            msg += "  delta_y_cone = {} m\n".format(delta_y_cone)
            msg += "  delta_z_cone = {} m\n".format(delta_z_cone)
            msg += "        xstart = {} m\n".format(self._xstart)
            msg += "        ystart = {} m\n".format(self._ystart)
            msg += "        zstart = {} m\n".format(self._zstart)
            msg += "       maxdist = {} m\n".format(self._maxdist)
            msg += "            nx = {}\n".format(self._nx)
            msg += "            ny = {}\n".format(self._ny)
            msg += "            nz = {}\n".format(self._nz)
            msg += "            nn = {}\n".format(self._nn)
            log.debug(msg)

        self._initialize_kolmogorov()
        return

    @function_timer
    def compress_volume(self):
        """Establish a mapping between full and observed volume indices."""
        log = Logger.get()

        if self._rank == 0:
            log.debug("Compressing volume, N = {}".format(self._nn))

        timer = Timer()
        timer.start()

        self._compressed_index = MPIShared((self._nn,), np.int64, self._comm)
        self._full_index = MPIShared((self._nn,), np.int64, self._comm)

        timer.stop()
        if self._rank == 0:
            log.debug(
                "Allocated shared full and compressed indices in {} s".format(
                    timer.seconds()
                )
            )
        timer.clear()
        timer.start()

        hit = np.zeros(self._nn, dtype=np.uint8)

        # Start by flagging all elements that are hit

        atm_sim_compress_flag_hits_rank(
            hit,
            self._ntask,
            self._rank,
            self._nx,
            self._ny,
            self._nz,
            self._xstart,
            self._ystart,
            self._zstart,
            self._delta_t,
            self._delta_az,
            self._elmin,
            self._elmax,
            self._wx,
            self._wy,
            self._wz,
            self._xstep,
            self._ystep,
            self._zstep,
            self._xstride,
            self._ystride,
            self._zstride,
            self._maxdist,
            self._cosel0,
            self._sinel0,
        )

        timer.stop()
        if self._rank == 0:
            log.debug("Flagged hits in {} s".format(timer.seconds()))
        timer.clear()
        timer.start()

        if self._comm is not None:
            self._comm.Allreduce(MPI.IN_PLACE, [hit, MPI.UNSIGNED_CHAR], MPI.LOR)

        timer.stop()
        if self._rank == 0:
            log.debug("Reduced hits in {} s".format(timer.seconds()))
        timer.clear()
        timer.start()

        # For extra margin, flag all the neighbors of the hit elements

        hit2 = np.array(hit)

        atm_sim_compress_flag_extend_rank(
            hit,
            hit2,
            self._ntask,
            self._rank,
            self._nx,
            self._ny,
            self._nz,
            self._xstride,
            self._ystride,
            self._zstride,
        )

        timer.stop()
        if self._rank == 0:
            log.debug("Extended flags in {} s".format(timer.seconds()))
        timer.clear()
        timer.start()

        del hit2

        if self._comm is not None:
            self._comm.Allreduce(MPI.IN_PLACE, [hit, MPI.UNSIGNED_CHAR], MPI.LOR)

        timer.stop()
        if self._rank == 0:
            log.debug("Reduced extended hits in {} s".format(timer.seconds()))
        timer.clear()
        timer.start()

        # Create the mappings between the compressed and full indices.  The hit
        # vector is duplicated across all processes and we are modifying node shared
        # memory.  Just one process per node manipulates the copy on each node.

        if (self._full_index._nodecomm is None) or (
            self._full_index._nodecomm.rank == 0
        ):
            # we are the root process on this node.
            i = 0
            for ifull in range(self._nn):
                if hit[ifull]:
                    self._full_index.data[i] = ifull
                    self._compressed_index.data[ifull] = i
                    i += 1
            self._nelem = i
        if self._full_index._nodecomm is not None:
            self._nelem = self._full_index._nodecomm.bcast(self._nelem, root=0)

        timer.stop()
        if self._rank == 0:
            log.debug("Created compression table in {} s".format(timer.seconds()))
        timer.clear()
        timer.start()

        del hit

        # Shrink the full index to what is needed.  Only one process per node copies
        # existing data.

        new_full = MPIShared((self._nelem,), np.int64, self._comm)
        if (self._full_index._nodecomm is None) or (
            self._full_index._nodecomm.rank == 0
        ):
            new_full.data[:] = self._full_index.data[: self._nelem]

        if self._comm is not None:
            self._comm.barrier()

        self._full_index.close()
        del self._full_index

        self._full_index = new_full

        if self._rank == 0:
            msg = "\nVolume compressed:\n"
            msg += "  {} / {} ({} %) elements needed for the simulation\n".format(
                self._nelem, self._nn, (self._nelem * 100 / self._nn)
            )
            msg += "  nx = {} ny = {} nz = {}\n".format(self._nx, self._ny, self._nz)
            msg += "  wx = {} wy = {} wz = {}\n".format(self._wx, self._wy, self._wz)
            log.debug(msg)
        if self._nelem == 0:
            raise RuntimeError("No elements in the observation cone.")
        return

    def smooth(self):
        pass

    def _meta_keys(self):
        """Helper function to return the list of member variables to read / write."""
        return list(
            [
                ("nn", int),
                ("nelem", int),
                ("nx", int),
                ("ny", int),
                ("nz", int),
                ("xstride", int),
                ("ystride", int),
                ("zstride", int),
                ("delta_x", float),
                ("delta_y", float),
                ("delta_z", float),
                ("xstart", float),
                ("ystart", float),
                ("zstart", float),
                ("maxdist", float),
                ("wx", float),
                ("wy", float),
                ("wz", float),
                ("lmin", float),
                ("lmax", float),
                ("w", float),
                ("wdir", float),
                ("z0", float),
                ("T0", float),
            ]
        )

    @function_timer
    def save_realization(self):
        if (
            (self._realization is None)
            or (self._full_index is None)
            or (self._compressed_index is None)
        ):
            raise RuntimeError("Cannot save realization: not loaded or generated")

        if self._rank == 0:

            log = Logger.get()

            rname = "{}_{}_{}_{}".format(
                self._key1, self._key2, self._counter1start, self._counter2start
            )
            outfile = os.path.join(self._cachedir, "{}.h5".format(rname))
            tmpfile = "{}.tmp".format(outfile)

            hf = h5py.File(tmpfile, "w")

            # Store metadata as attributes of the root group
            meta = hf.attrs
            for k, tp in self._meta_keys():
                meta.create(k, getattr(self, "_{}".format(k)))

            log.debug("Saved metadata for {}".format(rname))

            realiz = hf.create_dataset("realization", data=self._realization.data)
            log.debug("Saved realization for {}".format(rname))

            full = hf.create_dataset("full_index", data=self._full_index.data)
            log.debug("Saved full index for {}".format(rname))

            comp = hf.create_dataset(
                "compressed_index", data=self._compressed_index.data
            )
            log.debug("Saved compressed index for {}".format(rname))
            hf.flush()
            hf.close()

            # Move file into place
            os.rename(tmpfile, outfile)

        if self._comm is not None:
            self._comm.barrier()
        return

    @function_timer
    def load_realization(self):
        rname = None
        cachefile = None
        found = False
        if self._rank == 0:
            rname = "{}_{}_{}_{}".format(
                self._key1, self._key2, self._counter1start, self._counter2start
            )
            cachefile = os.path.join(self._cachedir, "{}.h5".format(rname))
            if os.path.isfile(cachefile):
                found = True
        if self._comm is not None:
            found = self._comm.bcast(found, root=0)

        if not found:
            return

        if self._realization is not None:
            del self._realization
        if self._full_index is not None:
            del self._full_index
        if self._compressed_index is not None:
            del self._compressed_index

        mdata = dict()
        hf = None
        if self._rank == 0:
            log = Logger.get()

            try:
                hf = h5py.File(cachefile, "r")
                # Read metadata
                meta = hf.attrs
                for k, tp in self._meta_keys():
                    # Copy the metadata value into a dictionary, casting to correct type
                    mdata[k] = tp(meta[k])
                log.debug("Loaded metadata for {}".format(rname))
            except Exception as e:
                print(
                    f"ERROR: failed to load cached atmosphere realization"
                    f" from {cachefile}. Will simulate again.",
                    flush=True,
                )
                mdata = None

        # Broadcast the metadata
        if self._comm is not None:
            mdata = self._comm.bcast(mdata, root=0)

        if mdata is None:
            return

        # Copy the metadata into class instance member variables
        for k, tp in self._meta_keys():
            setattr(self, "_{}".format(k), mdata[k])

        # Allocate the shared memory buffers
        self._realization = MPIShared((self._nelem,), np.float64, self._comm)
        self._compressed_index = MPIShared((self._nn,), np.int64, self._comm)
        self._full_index = MPIShared((self._nelem,), np.int64, self._comm)

        # Read and set shared memory
        buffer = None
        if self._rank == 0:
            buffer = np.zeros(self._nelem, dtype=np.float64)
            hf["realization"].read_direct(buffer)
            log.debug("Loaded realization for {}".format(rname))
        self._realization.set(buffer, (0,), fromrank=0)

        if self._rank == 0:
            buffer = np.zeros(self._nelem, dtype=np.int64)
            hf["full_index"].read_direct(buffer)
            log.debug("Loaded full index for {}".format(rname))
        self._full_index.set(buffer, (0,), fromrank=0)

        if self._rank == 0:
            buffer = np.zeros(self._nn, dtype=np.int64)
            hf["compressed_index"].read_direct(buffer)
            log.debug("Loaded compressed index for {}".format(rname))
        self._compressed_index.set(buffer, (0,), fromrank=0)

        if self._comm is not None:
            self._comm.barrier()

        self._cached = True
        return

    @function_timer
    def _initialize_kolmogorov(self):
        log = Logger.get()
        timer = Timer()
        timer.start()

        # Size of interpolation grid
        self._nr = 1000

        self._rmin_kolmo = 0.0
        diag = np.sqrt(self._delta_x ** 2 + self._delta_y ** 2)
        self._rmax_kolmo = np.sqrt(diag ** 2 + self._delta_z ** 2) * 1.01

        self._rstep = (self._rmax_kolmo - self._rmin_kolmo) / (self._nr - 1)

        self._kolmo_x, self._kolmo_y = atm_sim_kolmogorov_init_rank(
            self._nr,
            self._rmin_kolmo,
            self._rmax_kolmo,
            self._rstep,
            self._lmin,
            self._lmax,
            self._ntask,
            self._rank,
        )

        if self._comm is not None:
            self._comm.Allreduce(MPI.IN_PLACE, [self._kolmo_y, MPI.DOUBLE], MPI.SUM)

        # Normalize

        norm = 1.0 / self._kolmo_y[0]
        self._kolmo_y *= norm

        if (self._rank == 0) and self._write_debug:
            with open("kolmogorov.txt", "w") as f:
                for ir in range(self._nr):
                    f.write(
                        "{:.15e} {:.15e}\n".format(self._kolmo_x[ir], self._kolmo_y[ir])
                    )

        # Measure the correlation length
        icorr = self._nr - 1
        while np.absolute(self._kolmo_y[icorr]) < self._corrlim:
            icorr -= 1
        self._rcorr = self._kolmo_x[icorr]

        timer.stop()
        if self._rank == 0:
            log.debug("Initialized Kolmogorov in {} s".format(timer.seconds()))
        return

    def __repr__(self):
        ret = "<AtmSim rank = {}".format(self._rank)
        if self._rank == 0:
            ret += ":\n"
            ret += "  {} processes\n".format(self._ntask)
            ret += "  {} threads per process\n".format(self._nthread)
            ret += "  cachedir = {}\n".format(self._cachedir)
            ret += "  key1 = {}\n".format(self._key1)
            ret += "  key2 = {}\n".format(self._key2)
            ret += "  counter1 = {}\n".format(self._counter1)
            ret += "  counter2 = {}\n".format(self._counter2)
            ret += "  counter1start = {}\n".format(self._counter1start)
            ret += "  counter2start = {}\n".format(self._counter2start)
            ret += "  azmin = {}\n".format(self._azmin)
            ret += "  azmax = {}\n".format(self._azmax)
            ret += "  elmin = {}\n".format(self._elmin)
            ret += "  elmax = {}\n".format(self._elmax)
            ret += "  tmin = {}\n".format(self._tmin)
            ret += "  tmax = {}\n".format(self._tmax)
            ret += "  lmin_center = {}\n".format(self._lmin_center)
            ret += "  lmax_center = {}\n".format(self._lmax_center)
            ret += "  w_center = {}\n".format(self._w_center)
            ret += "  w_sigma = {}\n".format(self._w_sigma)
            ret += "  wdir_center = {}\n".format(self._wdir_center)
            ret += "  wdir_sigma = {}\n".format(self._wdir_sigma)
            ret += "  z0_center = {}\n".format(self._z0_center)
            ret += "  z0_sigma = {}\n".format(self._z0_sigma)
            ret += "  T0_center = {}\n".format(self._T0_center)
            ret += "  T0_sigma = {}\n".format(self._T0_sigma)
        ret += ">"
        return ret
